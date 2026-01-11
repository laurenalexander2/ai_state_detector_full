import sys
import time
import hashlib
from pathlib import Path
from io import BytesIO

import cv2
import numpy as np
import streamlit as st
from PIL import Image

# Optional: enables auto-animated swipe via timed reruns.
# IMPORTANT: we only call autorefresh when the swipe mode is active, otherwise it can look like an infinite loader.
try:
    from streamlit_autorefresh import st_autorefresh
except Exception:
    st_autorefresh = None


# -------------------------------------------------
# Path setup
# -------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.append(str(SRC))

from preprocess import preprocess
from align import align_images
from diffmaps import create_overlay
from utils import bgr_to_rgb


# -------------------------------------------------
# Page config
# -------------------------------------------------
st.set_page_config(page_title="Intaglio State Comparator", layout="centered")
st.title("Intaglio State Comparator")

st.markdown(
    """
Compare two impressions of the same intaglio plate.

- Examples-first
- Alignment swipe is a visual analysis mode
- Alignment swipe can autoplay if `streamlit-autorefresh` is installed
"""
)

# -------------------------------------------------
# Constants / paths
# -------------------------------------------------
IMG_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
UPLOAD_TYPES = ["jpg", "jpeg", "png", "tif", "tiff"]
EXAMPLES_DIR = ROOT / "app" / "examples"


# -------------------------------------------------
# Helpers
# -------------------------------------------------
def find_example_pairs(examples_dir: Path) -> dict[str, tuple[Path, Path]]:
    """
    app/examples/1/, 2/, 3/, ...
    Each folder must contain exactly two images.
    Alphabetically first filename = base, second = target.
    """
    pairs: dict[str, tuple[Path, Path]] = {}
    if not examples_dir.exists():
        return pairs

    for d in sorted(p for p in examples_dir.iterdir() if p.is_dir()):
        imgs = sorted(
            [p for p in d.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS],
            key=lambda p: p.name.lower(),
        )
        if len(imgs) == 2:
            pairs[d.name] = (imgs[0], imgs[1])
    return pairs


@st.cache_data(show_spinner=False)
def read_bytes(path_str: str) -> bytes:
    return Path(path_str).read_bytes()


@st.cache_data(show_spinner=False)
def decode_bgr(data: bytes) -> np.ndarray:
    arr = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Failed to decode image.")
    return img


@st.cache_data(show_spinner=False)
def run_preprocess(bgr: np.ndarray, max_width: int) -> np.ndarray:
    return preprocess(bgr, max_width=max_width)


@st.cache_data(show_spinner=False)
def run_align(base_gray: np.ndarray, target_gray: np.ndarray):
    return align_images(base_gray, target_gray)


def invert_luminance(rgb: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    hsv[:, :, 2] = 255 - hsv[:, :, 2]
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)


def make_color_overlay(base_gray: np.ndarray, target_gray: np.ndarray, swap: bool = False) -> np.ndarray:
    """
    Default: base=red, target=cyan (G+B).
    swap=True: base=cyan, target=red.
    """
    b = base_gray.astype(np.float32) / 255.0
    t = target_gray.astype(np.float32) / 255.0
    z = np.zeros_like(b)

    if not swap:
        base_rgb = np.stack([b, z, z], axis=2)  # red
        targ_rgb = np.stack([z, t, t], axis=2)  # cyan
    else:
        base_rgb = np.stack([z, b, b], axis=2)  # cyan
        targ_rgb = np.stack([t, z, z], axis=2)  # red

    out = np.clip(base_rgb + targ_rgb, 0, 1)
    return (out * 255).astype(np.uint8)


def wipe_base_to_target(base_img: np.ndarray, target_img: np.ndarray, pct: int) -> np.ndarray:
    """
    Base -> target reveal (left-to-right):
      pct=0   => 100% base
      pct=100 => 100% target
    Works for grayscale or RGB (shapes must match).
    """
    if base_img.shape[:2] != target_img.shape[:2]:
        raise ValueError(f"Wipe requires same shape. base={base_img.shape} target={target_img.shape}")

    h, w = base_img.shape[:2]
    x = int(w * pct / 100.0)
    out = base_img.copy()
    if x > 0:
        out[:, :x] = target_img[:, :x]
    return out


def stack_triptych(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    imgs = [a, b, c]
    h = max(img.shape[0] for img in imgs)
    resized = []
    for img in imgs:
        if img.shape[0] != h:
            s = h / img.shape[0]
            img = cv2.resize(img, (int(img.shape[1] * s), h), interpolation=cv2.INTER_AREA)
        resized.append(img)
    return np.hstack(resized)


def _bytes_fingerprint(data: bytes, take: int = 4096) -> bytes:
    if len(data) <= take * 2:
        return data
    return data[:take] + data[-take:]


def compute_token(base_bytes: bytes, target_bytes: bytes, max_width: int) -> str:
    h = hashlib.sha1()
    h.update(_bytes_fingerprint(base_bytes))
    h.update(_bytes_fingerprint(target_bytes))
    h.update(str(max_width).encode("utf-8"))
    return h.hexdigest()


# -------------------------------------------------
# Session state
# -------------------------------------------------
if "base_bytes" not in st.session_state:
    st.session_state.base_bytes = None
if "target_bytes" not in st.session_state:
    st.session_state.target_bytes = None

# derived cache keyed by (base_bytes, target_bytes, max_width)
if "derived_token" not in st.session_state:
    st.session_state.derived_token = None
if "base_gray" not in st.session_state:
    st.session_state.base_gray = None
if "target_gray" not in st.session_state:
    st.session_state.target_gray = None
if "aligned_target" not in st.session_state:
    st.session_state.aligned_target = None
if "H" not in st.session_state:
    st.session_state.H = None

# swipe state (prevents "always loading" and lets us stop/restart cleanly)
if "swipe_playing" not in st.session_state:
    st.session_state.swipe_playing = False
if "swipe_t0" not in st.session_state:
    st.session_state.swipe_t0 = 0.0
if "swipe_token" not in st.session_state:
    st.session_state.swipe_token = None


# -------------------------------------------------
# Inputs (examples-first)
# -------------------------------------------------
st.subheader("Inputs")
pairs = find_example_pairs(EXAMPLES_DIR)
tab_ex, tab_up = st.tabs(["Use an example", "Upload your own"])

with tab_ex:
    if not pairs:
        st.warning("No valid examples found in app/examples/ (each folder must contain exactly 2 images).")
    else:
        example_names = list(pairs.keys())
        chosen = st.selectbox("Choose an example folder", example_names, index=0)
        base_p, targ_p = pairs[chosen]

        with st.container(border=True):
            st.markdown("**Example preview (base + target)**")
            try:
                base_preview = cv2.cvtColor(decode_bgr(read_bytes(str(base_p))), cv2.COLOR_BGR2RGB)
                targ_preview = cv2.cvtColor(decode_bgr(read_bytes(str(targ_p))), cv2.COLOR_BGR2RGB)
                c1, c2 = st.columns(2)
                with c1:
                    st.image(base_preview, caption=f"Base: {base_p.name}", use_container_width=True)
                with c2:
                    st.image(targ_preview, caption=f"Target: {targ_p.name}", use_container_width=True)
            except Exception as e:
                st.warning(f"Could not preview example: {e}")

        if st.button("Use this example", type="primary"):
            st.session_state.base_bytes = read_bytes(str(base_p))
            st.session_state.target_bytes = read_bytes(str(targ_p))
            st.session_state.derived_token = None  # invalidate derived results
            st.session_state.swipe_playing = False  # stop swipe when switching input

with tab_up:
    u1 = st.file_uploader("Base image", type=UPLOAD_TYPES, key="upload_base")
    u2 = st.file_uploader("Target image", type=UPLOAD_TYPES, key="upload_target")
    st.caption("Uploads override the example currently loaded.")
    if u1 and u2:
        st.session_state.base_bytes = u1.getvalue()
        st.session_state.target_bytes = u2.getvalue()
        st.session_state.derived_token = None
        st.session_state.swipe_playing = False

if not st.session_state.base_bytes or not st.session_state.target_bytes:
    st.info("Choose an example (recommended) or upload two images to begin.")
    st.stop()

# Decode originals
base_bgr = decode_bgr(st.session_state.base_bytes)
target_bgr = decode_bgr(st.session_state.target_bytes)
base_rgb = cv2.cvtColor(base_bgr, cv2.COLOR_BGR2RGB)
target_rgb = cv2.cvtColor(target_bgr, cv2.COLOR_BGR2RGB)

with st.container(border=True):
    st.markdown("**Loaded pair**")
    c1, c2 = st.columns(2)
    with c1:
        st.image(base_rgb, caption="Base (loaded)", use_container_width=True)
    with c2:
        st.image(target_rgb, caption="Target (loaded)", use_container_width=True)


# -------------------------------------------------
# Processing settings
# -------------------------------------------------
st.markdown("---")
st.subheader("Processing settings")

max_width = st.slider(
    "Max processing width (px)",
    min_value=1200,
    max_value=6000,
    value=3000,
    step=200,
)

with st.expander("Advanced"):
    st.caption(
        "Higher values can improve fine detail but slow alignment and increase memory use. "
        "If things get slow or unstable, reduce this."
    )
    if st.checkbox("Enable extreme resolutions", value=False):
        max_width = st.slider("Extreme max width", 6000, 10000, 8000, 500)

# Compute preprocess + alignment lazily (only when token changes)
token = compute_token(st.session_state.base_bytes, st.session_state.target_bytes, max_width)
if st.session_state.derived_token != token:
    with st.spinner("Preprocessing and aligning..."):
        base_gray = run_preprocess(base_bgr, max_width)
        target_gray = run_preprocess(target_bgr, max_width)
        aligned_target, H = run_align(base_gray, target_gray)
    st.session_state.base_gray = base_gray
    st.session_state.target_gray = target_gray
    st.session_state.aligned_target = aligned_target
    st.session_state.H = H
    st.session_state.derived_token = token

    # Any processing change stops autoplay to prevent the "always loading" feel
    st.session_state.swipe_playing = False

base_gray = st.session_state.base_gray
target_gray = st.session_state.target_gray
aligned_target = st.session_state.aligned_target


# -------------------------------------------------
# Visual analysis modes
# -------------------------------------------------
st.markdown("---")
st.subheader("Visual analysis")

mode = st.radio(
    "Mode",
    ["Color overlay (red vs cyan)", "Tonal difference", "Alignment swipe (autoplay)"],
    index=0,
)

invert = st.checkbox("Invert black/white", value=False)

composite_rgb = None

if mode == "Color overlay (red vs cyan)":
    swap_colors = st.checkbox("Swap overlay colors (red ↔ cyan)", value=False)
    composite_rgb = make_color_overlay(base_gray, aligned_target, swap=swap_colors)
    if invert:
        composite_rgb = invert_luminance(composite_rgb)

    st.caption("Base original • Target original • Composite overlay")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.image(base_rgb, caption="Base original", use_container_width=True)
    with c2:
        st.image(target_rgb, caption="Target original", use_container_width=True)
    with c3:
        st.image(composite_rgb, caption="Overlay", use_container_width=True, clamp=True)

elif mode == "Tonal difference":
    thresh = st.slider("Mask sensitivity (higher = stricter mask)", 1, 255, 40, 1)
    diff = cv2.absdiff(base_gray, aligned_target)
    norm = cv2.normalize(diff, None, 0, 255, cv2.NORM_MINMAX)
    _, mask = cv2.threshold(diff, thresh, 255, cv2.THRESH_BINARY)

    overlay = bgr_to_rgb(create_overlay(base_gray, mask, alpha=0.5))
    if invert:
        overlay = invert_luminance(overlay)

    composite_rgb = overlay

    st.caption("Tonal difference • Binary mask • Overlay")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.image(norm, caption="Tonal difference", use_container_width=True, clamp=True)
    with c2:
        st.image(mask, caption="Binary mask", use_container_width=True, clamp=True)
    with c3:
        st.image(overlay, caption="Overlay", use_container_width=True, clamp=True)

else:
    # Fix: prevent "infinite loading" by ONLY autorefreshing while 'playing' is true,
    # and by keeping the swipe phase derived from (time.time() - swipe_t0), not raw time.time().
    # Also: render the swipe in a narrower center column so it's closer in feel to the other panels.
    WIPE_PERIOD_SEC = 10.0

    # If we entered swipe mode, keep a stable token so toggling modes doesn't desync.
    if st.session_state.swipe_token != token:
        st.session_state.swipe_token = token
        st.session_state.swipe_playing = False

    # Controls (minimal)
    autoplay_available = st_autorefresh is not None
    autoplay = st.checkbox("Autoplay", value=False, disabled=not autoplay_available)

    if autoplay and autoplay_available and not st.session_state.swipe_playing:
        st.session_state.swipe_playing = True
        st.session_state.swipe_t0 = time.time()

    if (not autoplay) and st.session_state.swipe_playing:
        st.session_state.swipe_playing = False

    if st.session_state.swipe_playing and autoplay_available:
        # Only rerun while actually playing (avoids global infinite spinner vibes)
        st_autorefresh(interval=120, key="swipe_refresh")  # ~8 fps (lighter on host)

        t = time.time() - st.session_state.swipe_t0
        phase = (t % WIPE_PERIOD_SEC) / WIPE_PERIOD_SEC
        wipe_pct = int(phase * 100)
    else:
        wipe_pct = st.slider("Swipe (base → aligned target)", 0, 100, 50, 1)

    try:
        swipe = wipe_base_to_target(base_gray, aligned_target, wipe_pct)
    except Exception as e:
        st.error(f"Swipe failed: {e}")
        st.stop()

    st.caption("Swipe viewer uses preprocessed images (grayscale).")

    # Smaller, centered rendering (matches the scale of the other panels better)
    a1, a2, a3 = st.columns([1.4, 2.0, 1.4])
    with a2:
        st.image(swipe, use_container_width=True, clamp=True)

    composite_rgb = cv2.cvtColor(swipe, cv2.COLOR_GRAY2RGB)


# -------------------------------------------------
# Download (originals + current view)
# -------------------------------------------------
st.markdown("---")
combined = stack_triptych(base_rgb, target_rgb, composite_rgb)
buf = BytesIO()
Image.fromarray(combined).save(buf, format="PNG")

st.download_button(
    "Download originals + current view (PNG)",
    data=buf.getvalue(),
    file_name="intaglio_state_comparison.png",
    mime="image/png",
)
