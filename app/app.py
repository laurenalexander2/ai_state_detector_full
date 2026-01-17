import sys
import hashlib
from pathlib import Path
from io import BytesIO

import cv2
import numpy as np
import streamlit as st
from PIL import Image

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
st.title("Computer-Vision Print Comparator")

st.markdown(
    """
Lauren Alexander / December 2025

For best view - trust me - try downloading!

Repo: [GitHub](https://github.com/laurenalexander2/ai_state_detector_full#)
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
def find_example_pairs(examples_dir: Path) -> list[tuple[str, Path, Path]]:
    """
    Returns a stable ordered list of (folder_name, base_path, target_path).
    app/examples/1/, 2/, 3/, ...
    Each folder must contain exactly two images.
    Alphabetically first filename = base, second = target.
    """
    out: list[tuple[str, Path, Path]] = []
    if not examples_dir.exists():
        return out

    for d in sorted((p for p in examples_dir.iterdir() if p.is_dir()), key=lambda p: p.name.lower()):
        imgs = sorted(
            [p for p in d.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS],
            key=lambda p: p.name.lower(),
        )
        if len(imgs) == 2:
            out.append((d.name, imgs[0], imgs[1]))
    return out


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


def make_download_sheet(
    base_img: np.ndarray,
    target_img: np.ndarray,
    overlay_img: np.ndarray | None,
    tonal_img: np.ndarray | None,
    mask_img: np.ndarray | None,
    tonal_overlay_img: np.ndarray | None,
) -> np.ndarray:
    """
    Build a 2-row sheet:
      Row 1: base | target | (optional overlay)
      Row 2: (optional) tonal_diff | mask | tonal_overlay
    Images are assumed RGB uint8 (mask/tonal_diff should be RGB already).
    """

    def resize_to_h(img: np.ndarray, h: int) -> np.ndarray:
        if img.shape[0] == h:
            return img
        s = h / img.shape[0]
        return cv2.resize(img, (int(img.shape[1] * s), h), interpolation=cv2.INTER_AREA)

    def hstack(imgs: list[np.ndarray]) -> np.ndarray:
        hh = max(im.shape[0] for im in imgs)
        rs = [resize_to_h(im, hh) for im in imgs]
        return np.hstack(rs)

    def pad_to_w(im: np.ndarray, w: int) -> np.ndarray:
        if im.shape[1] == w:
            return im
        pad = w - im.shape[1]
        return cv2.copyMakeBorder(
            im, 0, 0, 0, pad, borderType=cv2.BORDER_CONSTANT, value=(255, 255, 255)
        )

    row1 = [base_img, target_img] + ([overlay_img] if overlay_img is not None else [])
    row1_img = hstack(row1)

    rows = [row1_img]

    if tonal_img is not None and mask_img is not None and tonal_overlay_img is not None:
        row2_img = hstack([tonal_img, mask_img, tonal_overlay_img])
        w = max(row1_img.shape[1], row2_img.shape[1])
        rows = [pad_to_w(row1_img, w), pad_to_w(row2_img, w)]

    return np.vstack(rows)


# -------------------------------------------------
# Session state defaults
# -------------------------------------------------
st.session_state.setdefault("base_bytes", None)
st.session_state.setdefault("target_bytes", None)

st.session_state.setdefault("derived_token", None)
st.session_state.setdefault("base_gray", None)
st.session_state.setdefault("aligned_target", None)
st.session_state.setdefault("H", None)

# Input UI state
st.session_state.setdefault("sample_index", 0)
st.session_state.setdefault("sample_loaded_once", False)
st.session_state.setdefault("upload_nonce", 0)

# Settings state
st.session_state.setdefault("hide_overlay", False)
st.session_state.setdefault("invert_overlay", False)  # overlay-only
st.session_state.setdefault("swap_colors", False)  # overlay-only
st.session_state.setdefault("hide_tonal", False)
st.session_state.setdefault("mask_sensitivity", 30)  # 1..100 (tonal-only)
st.session_state.setdefault("max_width", 3000)


def reset_loaded_pair() -> None:
    st.session_state.base_bytes = None
    st.session_state.target_bytes = None
    st.session_state.derived_token = None
    st.session_state.base_gray = None
    st.session_state.aligned_target = None
    st.session_state.H = None


def clear_all() -> None:
    reset_loaded_pair()
    st.session_state.upload_nonce += 1  # reset uploaders by changing their keys


def load_next_sample(samples: list[tuple[str, Path, Path]]) -> None:
    if not samples:
        return
    name, base_p, targ_p = samples[st.session_state.sample_index]
    st.session_state.sample_index = (st.session_state.sample_index + 1) % len(samples)
    st.session_state.sample_loaded_once = True
    st.session_state.base_bytes = read_bytes(str(base_p))
    st.session_state.target_bytes = read_bytes(str(targ_p))
    st.session_state.derived_token = None


def render_actions_row(
    samples: list[tuple[str, Path, Path]],
    loaded: bool,
    show_download: bool,
    download_data: bytes | None,
    pos: str,
) -> None:
    """
    Renders: Try sample | Clear | Download view (PNG)
    - pos must be unique per row ("top" or "bottom") to keep widget keys stable.
    - show_download controls whether the download button appears (only after processing).
    """
    c1, c2, c3 = st.columns(3)

    with c1:
        sample_label = "Try a sample" if not st.session_state.sample_loaded_once else "Try a new sample"
        if st.button(
            sample_label,
            type="primary",
            use_container_width=True,
            disabled=(len(samples) == 0),
            key=f"btn_sample_{pos}",
        ):
            load_next_sample(samples)
            st.rerun()

    with c2:
        if st.button(
            "Clear",
            use_container_width=True,
            disabled=not loaded,
            key=f"btn_clear_{pos}",
        ):
            clear_all()
            st.rerun()

    with c3:
        if show_download and download_data is not None:
            st.download_button(
                "Download view (PNG)",
                data=download_data,
                file_name="intaglio_state_comparison.png",
                mime="image/png",
                use_container_width=True,
                key=f"dl_{pos}",
            )
        else:
            st.caption("Download enabled after processing.")


# -------------------------------------------------
# Inputs
# -------------------------------------------------
st.subheader("Inputs")

samples = find_example_pairs(EXAMPLES_DIR)
loaded = bool(st.session_state.base_bytes and st.session_state.target_bytes)

# Top actions row (download not available yet)
render_actions_row(
    samples=samples,
    loaded=loaded,
    show_download=False,
    download_data=None,
    pos="top",
)

# Uploaders (only show when nothing loaded)
if not loaded:
    nonce = st.session_state.upload_nonce
    u1 = st.file_uploader("Base image", type=UPLOAD_TYPES, key=f"upload_base_{nonce}")
    u2 = st.file_uploader("Target image", type=UPLOAD_TYPES, key=f"upload_target_{nonce}")
    st.caption("Loads only when both files are present.")

    if u1 and u2:
        st.session_state.base_bytes = u1.getvalue()
        st.session_state.target_bytes = u2.getvalue()
        st.session_state.derived_token = None
        st.rerun()

# Stop if still not loaded
loaded = bool(st.session_state.base_bytes and st.session_state.target_bytes)
if not loaded:
    if len(samples) == 0:
        st.info("Upload two images to begin. (No samples found in app/examples/.)")
    else:
        st.info("Upload two images to begin, or click “Try a sample”.")
    st.stop()

# -------------------------------------------------
# Decode originals
# -------------------------------------------------
base_bgr = decode_bgr(st.session_state.base_bytes)
target_bgr = decode_bgr(st.session_state.target_bytes)
base_rgb_full = cv2.cvtColor(base_bgr, cv2.COLOR_BGR2RGB)
target_rgb_full = cv2.cvtColor(target_bgr, cv2.COLOR_BGR2RGB)

# -------------------------------------------------
# Settings (collapsible, 3 columns)
# -------------------------------------------------
st.markdown("---")
with st.expander("Settings", expanded=False):
    col_overlay, col_tonal, col_meta = st.columns(3)

    with col_overlay:
        st.markdown("**Color overlay**")
        st.checkbox("Hide color overlay", key="hide_overlay")
        st.checkbox("Invert overlay black/white", key="invert_overlay")
        st.checkbox("Swap overlay colors (red ↔ cyan)", key="swap_colors")

    with col_tonal:
        st.markdown("**Tonal difference**")
        st.checkbox("Hide tonal difference", key="hide_tonal")
        st.slider(
            "Mask sensitivity",
            min_value=1,
            max_value=100,
            value=30,
            key="mask_sensitivity",
        )

    with col_meta:
        st.markdown("**Processing**")
        st.slider(
            "Max processing width (px)",
            min_value=3000,
            max_value=10000,
            step=200,
            value=2000,
            key="max_width",
        )

hide_overlay = bool(st.session_state.hide_overlay)
invert_overlay = bool(st.session_state.invert_overlay)
swap_colors = bool(st.session_state.swap_colors)
hide_tonal = bool(st.session_state.hide_tonal)
mask_sensitivity = int(st.session_state.mask_sensitivity)
max_width = int(st.session_state.max_width)

# Map 1..100 -> 1..255
thresh_255 = int(round(1 + (mask_sensitivity - 1) * (254 / 99)))

# -------------------------------------------------
# Compute preprocess + alignment lazily (tokened)
# -------------------------------------------------
token = compute_token(st.session_state.base_bytes, st.session_state.target_bytes, max_width)
if st.session_state.derived_token != token:
    with st.spinner("Preprocessing and aligning..."):
        base_gray = run_preprocess(base_bgr, max_width)
        target_gray = run_preprocess(target_bgr, max_width)
        aligned_target, H = run_align(base_gray, target_gray)

    st.session_state.base_gray = base_gray
    st.session_state.aligned_target = aligned_target
    st.session_state.H = H
    st.session_state.derived_token = token

base_gray = st.session_state.base_gray
aligned_target = st.session_state.aligned_target

# Match display originals to processed resolution
h, w = base_gray.shape[:2]
base_rgb = cv2.resize(base_rgb_full, (w, h), interpolation=cv2.INTER_AREA)
target_rgb = cv2.resize(target_rgb_full, (w, h), interpolation=cv2.INTER_AREA)

# -------------------------------------------------
# Compute outputs
# -------------------------------------------------
color_overlay = make_color_overlay(base_gray, aligned_target, swap=swap_colors)
if invert_overlay:
    color_overlay = invert_luminance(color_overlay)

diff = cv2.absdiff(base_gray, aligned_target)
tonal_diff = cv2.normalize(diff, None, 0, 255, cv2.NORM_MINMAX)
_, mask = cv2.threshold(diff, thresh_255, 255, cv2.THRESH_BINARY)
tonal_overlay = bgr_to_rgb(create_overlay(base_gray, mask, alpha=0.5))

tonal_diff_rgb = cv2.cvtColor(tonal_diff, cv2.COLOR_GRAY2RGB)
mask_rgb = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)

# -------------------------------------------------
# Display: 2 rows x 3 images
# -------------------------------------------------
st.markdown("---")
with st.container(border=True):
    st.markdown("**Loaded pair + analysis**")

    # Row 1
    if hide_overlay:
        c1, c2 = st.columns(2)
        with c1:
            st.image(base_rgb, caption="Base (loaded)", use_container_width=True)
        with c2:
            st.image(target_rgb, caption="Target (loaded)", use_container_width=True)
    else:
        c1, c2, c3 = st.columns(3)
        with c1:
            st.image(base_rgb, caption="Base (loaded)", use_container_width=True)
        with c2:
            st.image(target_rgb, caption="Target (loaded)", use_container_width=True)
        with c3:
            st.image(color_overlay, caption="Color overlay", use_container_width=True, clamp=True)

    # Row 2
    if not hide_tonal:
        d1, d2, d3 = st.columns(3)
        with d1:
            st.image(tonal_diff_rgb, caption="Tonal difference", use_container_width=True, clamp=True)
        with d2:
            st.image(mask_rgb, caption="Binary mask", use_container_width=True, clamp=True)
        with d3:
            st.image(tonal_overlay, caption="Tonal overlay", use_container_width=True, clamp=True)

# -------------------------------------------------
# Export + Download data
# -------------------------------------------------
overlay_for_export = None if hide_overlay else color_overlay
tonal_for_export = None if hide_tonal else tonal_diff_rgb
mask_for_export = None if hide_tonal else mask_rgb
tonal_overlay_for_export = None if hide_tonal else tonal_overlay

export_sheet = make_download_sheet(
    base_img=base_rgb,
    target_img=target_rgb,
    overlay_img=overlay_for_export,
    tonal_img=tonal_for_export,
    mask_img=mask_for_export,
    tonal_overlay_img=tonal_overlay_for_export,
)

buf = BytesIO()
Image.fromarray(export_sheet).save(buf, format="PNG")
download_bytes = buf.getvalue()

# -------------------------------------------------
# Bottom actions row (now includes download)
# -------------------------------------------------
st.markdown("")
render_actions_row(
    samples=samples,
    loaded=True,
    show_download=True,
    download_data=download_bytes,
    pos="bottom",
)
