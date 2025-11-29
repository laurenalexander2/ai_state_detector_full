
import sys
from pathlib import Path

import cv2
import numpy as np
import streamlit as st

# Make src importable when running `streamlit run app/app.py`
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.append(str(SRC))

from preprocess import preprocess
from align import align_images
from diffmaps import compute_diff_mask, create_overlay
from utils import bgr_to_rgb


st.set_page_config(page_title="Intaglio State Detector", layout="wide")

st.title("AI-Assisted State Detection for Intaglio Prints")

st.markdown(
    """
Upload two digitized impressions of the **same intaglio print**.  
The app will:

1. Preprocess and normalize both images  
2. Align the second image onto the first using feature-based registration  
3. Compute a difference mask (edge-based)  
4. Overlay the changes in red on the base impression  

This is a research prototype intended to support close looking and connoisseurship, not to replace it.
"""
)

col1, col2 = st.columns(2)

with col1:
    uploaded1 = st.file_uploader("Upload first impression (base)", type=["jpg", "jpeg", "png", "tif", "tiff"], key="img1")

with col2:
    uploaded2 = st.file_uploader("Upload second impression (to compare)", type=["jpg", "jpeg", "png", "tif", "tiff"], key="img2")

if uploaded1 is not None and uploaded2 is not None:
    # Read bytes into OpenCV images
    file_bytes1 = np.asarray(bytearray(uploaded1.read()), dtype=np.uint8)
    file_bytes2 = np.asarray(bytearray(uploaded2.read()), dtype=np.uint8)
    img1_bgr = cv2.imdecode(file_bytes1, cv2.IMREAD_COLOR)
    img2_bgr = cv2.imdecode(file_bytes2, cv2.IMREAD_COLOR)

    if img1_bgr is None or img2_bgr is None:
        st.error("Failed to decode one or both images.")
    else:
        max_width = st.slider("Max processing width (pixels)", 800, 3000, 2000, step=100)

        try:
            base_gray = preprocess(img1_bgr, max_width=max_width)
            target_gray = preprocess(img2_bgr, max_width=max_width)
        except Exception as e:
            st.error(f"Error during preprocessing: {e}")
            st.stop()

        st.subheader("Preprocessed inputs")
        c1, c2 = st.columns(2)
        with c1:
            st.image(base_gray, caption="Base (preprocessed)", use_container_width=True, clamp=True)
        with c2:
            st.image(target_gray, caption="Target (preprocessed)", use_container_width=True, clamp=True)

        st.markdown("---")
        st.subheader("Alignment")

        try:
            aligned_target, H = align_images(base_gray, target_gray)
            st.success("Alignment succeeded.")
        except Exception as e:
            st.error(f"Alignment failed: {e}")
            st.stop()

        st.image(aligned_target, caption="Target aligned to base", use_container_width=True, clamp=True)

        st.markdown("---")
        st.subheader("Difference map and overlay")

        use_edges = st.checkbox("Use edge-based difference (recommended)", value=True)

        try:
            norm_diff, mask = compute_diff_mask(base_gray, aligned_target, use_edges=use_edges)
        except Exception as e:
            st.error(f"Difference computation failed: {e}")
            st.stop()

        # Show raw difference and mask
        c3, c4 = st.columns(2)
        with c3:
            st.image(norm_diff, caption="Normalized difference", use_container_width=True, clamp=True)
        with c4:
            st.image(mask, caption="Binary change mask", use_container_width=True, clamp=True)

        overlay_bgr = create_overlay(base_gray, mask, alpha=0.5)
        overlay_rgb = bgr_to_rgb(overlay_bgr)

        st.markdown("---")
        st.subheader("Overlay visualization")

        st.image(overlay_rgb, caption="Changes highlighted in red on base impression", use_container_width=True)

        st.markdown(
            """
**Interpretation tip:**  
Red regions mark where the algorithm detects notable changes in edges or tone between these two impressions.
These may correspond to:

- Added or removed cross-hatching
- Reworked contours or shadows
- Areas of plate wear or burnishing

They are **starting points for close looking**, not definitive attributions of state.
"""
        )
else:
    st.info("Upload two impressions of the same print to begin.")
