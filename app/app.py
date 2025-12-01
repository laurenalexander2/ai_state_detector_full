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
from diffmaps import (
    compute_diff_mask,
    create_overlay,
    compute_edge_line_mask,
    color_overlay_two,
)
from utils import bgr_to_rgb


st.set_page_config(page_title="Intaglio State Comparator", layout="centered")

st.title("Intaglio State Comparator")

st.markdown(
    """
Upload two digitized impressions of the **same intaglio plate** – for example,
different states, different printings, or impressions from different collections.

This tool will:

1. **Standardize** both images (crop, resize, and normalize contrast)  
2. **Align** the second impression onto the first using feature-based registration  
3. **Compare** the aligned impressions using several visual modes:

   - **Tonal difference** – highlights regions where overall darkness, plate wear, or inking differ  
   - **Line-sensitive color overlay** – tints one impression red and the other cyan so added or strengthened lines appear as color fringes  
   - **Experimental line difference** – a work-in-progress mode that tries to isolate changed line segments

The goal is to **support close looking and connoisseurship**:

- surfacing areas where cross-hatching has been reinforced or effaced  
- drawing attention to retouching, plate wear, or damage  
- helping you move quickly between impressions without losing your place on the plate

It does **not** assign states or make attributions on its own.  
All visualizations should be read as prompts for expert interpretation, not as final judgments.
"""
)

col1, col2 = st.columns(2)

with col1:
    uploaded1 = st.file_uploader(
        "Upload first impression (base)",
        type=["jpg", "jpeg", "png", "tif", "tiff"],
        key="img1",
    )

with col2:
    uploaded2 = st.file_uploader(
        "Upload second impression (to compare)",
        type=["jpg", "jpeg", "png", "tif", "tiff"],
        key="img2",
    )

if uploaded1 is not None and uploaded2 is not None:
    # Read bytes into OpenCV images
    file_bytes1 = np.asarray(bytearray(uploaded1.read()), dtype=np.uint8)
    file_bytes2 = np.asarray(bytearray(uploaded2.read()), dtype=np.uint8)
    img1_bgr = cv2.imdecode(file_bytes1, cv2.IMREAD_COLOR)
    img2_bgr = cv2.imdecode(file_bytes2, cv2.IMREAD_COLOR)

    if img1_bgr is None or img2_bgr is None:
        st.error("Failed to decode one or both images.")
    else:
        max_width = st.slider(
            "Max processing width (pixels)", 800, 3000, 2000, step=100
        )

        try:
            base_gray = preprocess(img1_bgr, max_width=max_width)
            target_gray = preprocess(img2_bgr, max_width=max_width)
        except Exception as e:
            st.error(f"Error during preprocessing: {e}")
            st.stop()

        st.subheader("Preprocessed inputs")
        c1, c2 = st.columns(2)
        with c1:
            st.image(
                base_gray,
                caption="Base (preprocessed)",
                use_container_width=True,
                clamp=True,
            )
        with c2:
            st.image(
                target_gray,
                caption="Target (preprocessed)",
                use_container_width=True,
                clamp=True,
            )

        st.markdown("---")
        st.subheader("Alignment")

        try:
            aligned_target, H = align_images(base_gray, target_gray)
            st.success("Alignment succeeded.")
        except Exception as e:
            st.error(f"Alignment failed: {e}")
            st.stop()

        # Center the alignment image in a narrower middle column
        a1, a2, a3 = st.columns([1, 2, 1])
        with a2:
            st.image(
                aligned_target,
                caption="Target aligned to base",
                use_container_width=True,
                clamp=True,
            )

        # ===== Comparison modes =====
        st.markdown("---")
        st.subheader("Difference map and overlay")

        mode = st.radio(
            "Choose view / algorithm",
            (
                "Tonal difference (current implementation)",
                "Experimental line difference (edge XOR)",
                "Color overlay (red vs cyan)",
            ),
        )

        if mode == "Tonal difference (current implementation)":
            # Original compute_diff_mask behavior; use_edges=False usually looks best
            try:
                norm_diff, mask = compute_diff_mask(
                    base_gray, aligned_target, use_edges=False
                )
            except Exception as e:
                st.error(f"Difference computation failed: {e}")
                st.stop()

            c3, c4 = st.columns(2)
            with c3:
                st.image(
                    norm_diff,
                    caption="Tonal / intensity difference",
                    use_container_width=True,
                    clamp=True,
                )
            with c4:
                st.image(
                    mask,
                    caption="Binary mask (tonal diff)",
                    use_container_width=True,
                    clamp=True,
                )

            overlay_bgr = create_overlay(base_gray, mask, alpha=0.5)
            overlay_rgb = bgr_to_rgb(overlay_bgr)

            st.subheader("Overlay visualization")
            # Center the overlay image like the alignment image
            o1, o2, o3 = st.columns([1, 2, 1])
            with o2:
                st.image(
                    overlay_rgb,
                    caption="Overlay (tonal difference)",
                    use_container_width=True,
                    clamp=True,
                )

        elif mode == "Experimental line difference (edge XOR)":
            try:
                edge_diff, line_mask = compute_edge_line_mask(
                    base_gray, aligned_target
                )
            except Exception as e:
                st.error(f"Edge-line difference failed: {e}")
                st.stop()

            c3, c4 = st.columns(2)
            with c3:
                st.image(
                    edge_diff,
                    caption="Changed line segments (edge XOR, experimental)",
                    use_container_width=True,
                    clamp=True,
                )
            with c4:
                st.image(
                    line_mask,
                    caption="Binary mask of changed lines",
                    use_container_width=True,
                    clamp=True,
                )

            overlay_bgr = create_overlay(base_gray, line_mask, alpha=0.6)
            overlay_rgb = bgr_to_rgb(overlay_bgr)

            st.subheader("Overlay visualization")
            o1, o2, o3 = st.columns([1, 2, 1])
            with o2:
                st.image(
                    overlay_rgb,
                    caption="Changed lines highlighted in red (experimental)",
                    use_container_width=True,
                    clamp=True,
                )

        else:  # "Color overlay (red vs cyan)"
            color_bgr = color_overlay_two(base_gray, aligned_target)
            color_rgb = bgr_to_rgb(color_bgr)

            st.subheader("Color overlay of both impressions")
            o1, o2, o3 = st.columns([1, 2, 1])
            with o2:
                st.image(
                    color_rgb,
                    caption="Base = red, Target = cyan (unique lines tinted)",
                    use_container_width=True,
                    clamp=True,
                )

else:
    st.info("Upload two impressions of the same print to begin.")
