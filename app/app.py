import sys
import time
from pathlib import Path
from io import BytesIO

import cv2
import numpy as np
import streamlit as st
from PIL import Image

# Optional: smooth auto-refresh for the alignment wipe animation.
# If missing, the app still runs (wipe just won't auto-animate).
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

This tool:
1) Standardizes both images (crop/resize/normalize)
2) Aligns the second impression to the first
3) Visualizes differences to support close looking

"""
)

# -------------------------------------------------
# Example discovery
# -------------------------------------------------
IMG_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
UPLOAD_TYPES = ["jpg", "jpeg", "png", "tif", "tiff"]
EXAMPLES_DIR = ROOT / "app" / "examples"


def find_example_pairs(examples_dir: Path) -> dict[str, tuple[Path, Path]]:
    """
    app/examples/1/, 2/, 3/, ...
    Each folder must contain exactly two images.
    Alphabetically first filename = base
    Alphabetically second filename = target
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
        base_rgb = np.stack([b, z, z], axis=2)       # red
        targ_rgb = np.stack([z, t, t], axis=2)       # cyan
    else:
        base_rgb = np.stack([z, b, b], axis=2)       # cyan
        targ_rgb = np.stack([t, z, z], axis=2)       # red

    out = np.clip(base_rgb + targ_rgb, 0, 1)
    return (out * 255).astype(np.uint8)


def wipe_view(base: np.ndarray, aligned: np.ndarray, pct: int) -> np.ndarray:
    """
    Simple wipe: left portion base, right portion aligned target.
    pct 0..100 moves boundary.
    """
    h, w = base.shape
    x = int(w * pct / 100.0)
    out = np.zeros_like(base)
    out[:, :x] = base[:, :x]
    out[:, x:] = aligned[:, x:]
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


# -------------------------------------------------
# Session state
# -------------------------------------------------
if "base_bytes" not in st.session_state:
    st.session_state.base_bytes = None
if "target_bytes" not in st.session_state:
    st.session_state.target_bytes = None


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
        chosen = st.selectbox("Choose an example", example_names, index=0)

        # Preview base image (alphabetically first) immediately on selection
        base_p, targ_p = pairs[chosen]
        try:
            preview_bytes = read_bytes(str(base_p))
            preview_bgr = decode_bgr(preview_bytes)
            preview_rgb = cv2.cvtColor(preview_bgr, cv2.COLOR_BGR2RGB)
            st.image(
                preview_rgb,
                caption=f"Preview (base): {base_p.name}",
                use_container_width=True,
            )
        except Exception as e:
            st.warning(f"Could not preview example: {e}")

        c1, c2 = st.columns([1, 2])
        with c1:
            if st.button("Use example", type="primary"):
                st.session_state.base_bytes = read_bytes(str(base_p))
                st.session_state.target_bytes = read_bytes(str(targ_p))
        with c2:
            st.caption(f"Base: {base_p.name} • Target: {targ_p.name}")

with tab_up:
    u1 = st.file_uploader("Base image", type=UPLOAD_TYPES, key="upload_base")
    u2 = st.file_uploader("Target image", type=UPLOAD_TYPES, key="upload_target")
    st.caption("Uploads override the example currently loaded.")
    if u1 and u2:
        st.session_state.base_bytes = u1.getvalue()
        st.session_state.target_bytes = u2.getvalue()

if not st.session_state.base_bytes or not st.session_state.target_bytes:
    st.info("Select an example (recommended) or upload two images to begin.")
    st.stop()


# -------------------------------------------------
# Decode originals
# -------------------------------------------------
try:
    base_bgr = decode_bgr(st.session_state.base_bytes)
    target_bgr = decode_bgr(st.session_state.target_bytes)
except Exception as e:
    st.error(f"Failed to decode one or both images: {e}")
    st.stop()

base_rgb = cv2.cvtColor(base_bgr, cv2.COLOR_BGR2RGB)
target_rgb = cv2.cvtColor(target_bgr, cv2.COLOR_BGR2RGB)


# -------------------------------------------------
# Processing settings
# -------------------------------------------------
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
        "Higher values can improve fine detail but slow alignment a lot and increase memory use. "
        "If alignment becomes slow or unstable, reduce this."
    )
    if st.checkbox("Enable extreme resolutions", value=False):
        max_width = st.slider(
            "Extreme max width (use sparingly)",
            min_value=6000,
            max_value=10000,
            value=8000,
            step=500,
        )


# -------------------------------------------------
# Preprocessing
# -------------------------------------------------
show_pre = st.toggle("Show preprocessing preview (before / after)", value=False)

try:
    base_gray = run_preprocess(base_bgr, max_width)
    target_gray = run_preprocess(target_bgr, max_width)
except Exception as e:
    st.error(f"Error during preprocessing: {e}")
    st.stop()

if show_pre:
    st.caption("Original (RGB) and preprocessed (grayscale) used for alignment/comparison.")
    c1, c2 = st.columns(2)
    with c1:
        st.image(base_rgb, caption="Base original", use_container_width=True)
        st.image(base_gray, caption="Base preprocessed", use_container_width=True, clamp=True)
    with c2:
        st.image(target_rgb, caption="Target original", use_container_width=True)
        st.image(target_gray, caption="Target preprocessed", use_container_width=True, clamp=True)


# -------------------------------------------------
# Alignment (auto-looping wipe)
# -------------------------------------------------
st.subheader("Alignment")

try:
    aligned_target, H = run_align(base_gray, target_gray)
    st.success("Alignment succeeded.")
except Exception as e:
    st.error(f"Alignment failed: {e}")
    st.stop()

WIPE_PERIOD_SEC = 10.0  # full left->right sweep duration

if st_autorefresh is None:
    st.caption("Auto-wipe animation requires `streamlit-autorefresh` (optional).")
    st.caption("Install: `pip install streamlit-autorefresh` (then it will loop automatically).")
    wipe_pct = st.slider("Alignment wipe", 0, 100, 50, 1)
else:
    # ~12.5 fps refresh (80ms). Lower fps reduces CPU usage.
    st_autorefresh(interval=80, key="alignment_wipe_refresh")
    phase = (time.time() % WIPE_PERIOD_SEC) / WIPE_PERIOD_SEC
    wipe_pct = int(phase * 100)

wipe_img = wipe_view(base_gray, aligned_target, wipe_pct)
a1, a2, a3 = st.columns([1, 2, 1])
with a2:
    st.image(wipe_img, caption="Alignment wipe", use_container_width=True, clamp=True)


# -------------------------------------------------
# Comparison
# -------------------------------------------------
st.subheader("Comparison")

mode = st.radio(
    "View mode",
    ["Color overlay (red vs cyan)", "Tonal difference"],
    index=0,  # favor color overlay
)

invert = st.checkbox("Invert black/white", value=False)

if mode == "Color overlay (red vs cyan)":
    swap_colors = st.checkbox("Swap overlay colors (red ↔ cyan)", value=False)
    overlay = make_color_overlay(base_gray, aligned_target, swap=swap_colors)
    if invert:
        overlay = invert_luminance(overlay)

    st.caption("Base original • Target original • Composite overlay")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.image(base_rgb, caption="Base", use_container_width=True)
    with c2:
        st.image(target_rgb, caption="Target", use_container_width=True)
    with c3:
        st.image(overlay, caption="Overlay", use_container_width=True, clamp=True)

else:
    thresh = st.slider("Mask sensitivity (higher = stricter mask)", 1, 255, 40, 1)

    diff = cv2.absdiff(base_gray, aligned_target)
    norm = cv2.normalize(diff, None, 0, 255, cv2.NORM_MINMAX)
    _, mask = cv2.threshold(diff, thresh, 255, cv2.THRESH_BINARY)

    overlay = bgr_to_rgb(create_overlay(base_gray, mask, alpha=0.5))
    if invert:
        overlay = invert_luminance(overlay)

    st.caption("Tonal difference • Binary mask • Overlay")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.image(norm, caption="Tonal difference", use_container_width=True, clamp=True)
    with c2:
        st.image(mask, caption="Binary mask", use_container_width=True, clamp=True)
    with c3:
        st.image(overlay, caption="Overlay", use_container_width=True, clamp=True)


# -------------------------------------------------
# Download
# -------------------------------------------------
combined = stack_triptych(base_rgb, target_rgb, overlay)
buf = BytesIO()
Image.fromarray(combined).save(buf, format="PNG")

st.download_button(
    "Download originals + overlay (PNG)",
    data=buf.getvalue(),
    file_name="intaglio_state_comparison.png",
    mime="image/png",
)
