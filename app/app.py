import sys
from pathlib import Path
from io import BytesIO

import cv2
import numpy as np
import streamlit as st
from PIL import Image

# Make src importable when running `streamlit run app/app.py`
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.append(str(SRC))

from preprocess import preprocess
from align import align_images
from diffmaps import (
    compute_diff_mask,      # still imported in case you want it later
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

# ------------------------------------------------------------------
# Session state for swap & visibility toggles
# ------------------------------------------------------------------
if "swap" not in st.session_state:
    st.session_state.swap = False
if "hide_base" not in st.session_state:
    st.session_state.hide_base = False
if "hide_target" not in st.session_state:
    st.session_state.hide_target = False

# -----------------------------
# Uploaders + top swap button
# -----------------------------
col1, col2, col3 = st.columns([2, 2, 1])

with col1:
    uploaded1 = st.file_uploader(
        "Upload first impression",
        type=["jpg", "jpeg", "png", "tif", "tiff"],
        key="img1",
    )

with col2:
    uploaded2 = st.file_uploader(
        "Upload second impression",
        type=["jpg", "jpeg", "png", "tif", "tiff"],
        key="img2",
    )

with col3:
    st.write(" ")
    st.write(" ")
    if st.button("Swap base/target"):
        st.session_state.swap = not st.session_state.swap

# ----------------------------------------------------------
# Proceed once both uploads are present (in either order)
# ----------------------------------------------------------
if uploaded1 is not None and uploaded2 is not None:
    # Decide which uploaded file is base vs target according to swap flag
    if not st.session_state.swap:
        base_uploaded = uploaded1
        target_uploaded = uploaded2
    else:
        base_uploaded = uploaded2
        target_uploaded = uploaded1

    # Read bytes into OpenCV images (base, then target)
    file_bytes_base = np.asarray(bytearray(base_uploaded.read()), dtype=np.uint8)
    file_bytes_target = np.asarray(bytearray(target_uploaded.read()), dtype=np.uint8)

    base_bgr = cv2.imdecode(file_bytes_base, cv2.IMREAD_COLOR)
    target_bgr = cv2.imdecode(file_bytes_target, cv2.IMREAD_COLOR)

    if base_bgr is None or target_bgr is None:
        st.error("Failed to decode one or both images.")
        st.stop()

    # Save original RGB versions for display + download
    base_rgb_original = cv2.cvtColor(base_bgr, cv2.COLOR_BGR2RGB)
    target_rgb_original = cv2.cvtColor(target_bgr, cv2.COLOR_BGR2RGB)

    # ------------------------------
    # Show original uploaded images
    # ------------------------------
    st.subheader("Original uploaded images")
    oc1, oc2 = st.columns(2)
    with oc1:
        st.image(
            base_rgb_original,
            caption="Base (original uploaded)",
            use_container_width=True,
        )
    with oc2:
        st.image(
            target_rgb_original,
            caption="Target (original uploaded)",
            use_container_width=True,
        )

    st.markdown("---")

    # ------------------------------
    # Preprocessing
    # ------------------------------
    max_width = st.slider(
        "Max processing width (pixels)", 800, 3000, 2000, step=100
    )

    try:
        base_gray = preprocess(base_bgr, max_width=max_width)
        target_gray = preprocess(target_bgr, max_width=max_width)
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

    # ------------------------------
    # Alignment
    # ------------------------------
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

    # -------------------------------------------------
    # Helper: invert luminance while preserving colors
    # -------------------------------------------------
    def invert_luminance(rgb_img: np.ndarray) -> np.ndarray:
        """Invert brightness only; keep hue/saturation (red/cyan stay red/cyan)."""
        hsv = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2HSV)
        hsv[:, :, 2] = 255 - hsv[:, :, 2]
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

    # -------------------------------------------------
    # Helper: stack originals + composite side by side
    # -------------------------------------------------
    def stack_triptych(base_rgb: np.ndarray,
                       target_rgb: np.ndarray,
                       composite_rgb: np.ndarray) -> np.ndarray:
        """Resize all to same height and stack horizontally."""
        imgs = [base_rgb, target_rgb, composite_rgb]
        # Ensure uint8
        imgs = [img.astype("uint8") for img in imgs]

        max_h = max(img.shape[0] for img in imgs)
        resized = []
        for img in imgs:
            h, w = img.shape[:2]
            if h != max_h:
                scale = max_h / h
                new_w = int(w * scale)
                img_resized = cv2.resize(img, (new_w, max_h), interpolation=cv2.INTER_AREA)
            else:
                img_resized = img
            resized.append(img_resized)

        combined = np.hstack(resized)
        return combined

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

    # This will hold the currently displayed composite
    composite_rgb = None

    # -------------------------------------------------
    # 1) Tonal difference mode (with threshold slider)
    # -------------------------------------------------
    if mode == "Tonal difference (current implementation)":
        # Slider to adjust how sensitive the binary mask is
        threshold = st.slider(
            "Mask sensitivity (higher = stricter mask)",
            min_value=1,
            max_value=255,
            value=40,
            step=1,
        )

        try:
            # Raw absolute difference between base and aligned target
            diff_raw = cv2.absdiff(base_gray, aligned_target)

            # Normalize for visualization
            norm_diff = cv2.normalize(diff_raw, None, 0, 255, cv2.NORM_MINMAX)

            # Manual threshold -> binary mask
            _, mask = cv2.threshold(diff_raw, threshold, 255, cv2.THRESH_BINARY)
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
                caption="Binary mask (tonal diff, thresholded)",
                use_container_width=True,
                clamp=True,
            )

        overlay_bgr = create_overlay(base_gray, mask, alpha=0.5)
        overlay_rgb = bgr_to_rgb(overlay_bgr)

        invert_bw = st.checkbox(
            "Invert black/white in composite",
            key="invert_bw_tonal",
        )
        if invert_bw:
            overlay_rgb = invert_luminance(overlay_rgb)

        composite_rgb = overlay_rgb  # track current composite

        st.subheader("Overlay visualization")
        o1, o2, o3 = st.columns([1, 2, 1])
        with o2:
            st.image(
                overlay_rgb,
                caption="Overlay (tonal difference)",
                use_container_width=True,
                clamp=True,
            )

    # -------------------------------------------------
    # 2) Line difference mode
    # -------------------------------------------------
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

        invert_bw = st.checkbox(
            "Invert black/white in composite",
            key="invert_bw_line",
        )
        if invert_bw:
            overlay_rgb = invert_luminance(overlay_rgb)

        composite_rgb = overlay_rgb  # track current composite

        st.subheader("Overlay visualization")
        o1, o2, o3 = st.columns([1, 2, 1])
        with o2:
            st.image(
                overlay_rgb,
                caption="Changed lines highlighted in red (experimental)",
                use_container_width=True,
                clamp=True,
            )

    # -------------------------------------------------
    # 3) Color overlay mode (with hide/show + invert)
    # -------------------------------------------------
    else:  # "Color overlay (red vs cyan)"

        # Buttons to hide/show base/target and swap again
        cA, cB, cC = st.columns(3)
        with cA:
            if st.button("Hide Base (red)"):
                st.session_state.hide_base = not st.session_state.hide_base
        with cB:
            if st.button("Hide Target (cyan)"):
                st.session_state.hide_target = not st.session_state.hide_target
        with cC:
            if st.button("Swap base/target (here)"):
                st.session_state.swap = not st.session_state.swap

        # Normalize to 0–1
        base_norm = base_gray.astype(np.float32) / 255.0
        target_norm = aligned_target.astype(np.float32) / 255.0

        # Tint base (red) and target (cyan)
        zeros_base = np.zeros_like(base_norm, dtype=np.float32)
        zeros_target = np.zeros_like(target_norm, dtype=np.float32)

        base_rgb = np.stack([base_norm, zeros_base, zeros_base], axis=2)
        target_rgb = np.stack([zeros_target, target_norm, target_norm], axis=2)

        # Apply hide toggles
        if st.session_state.hide_base:
            base_rgb[:] = 0
        if st.session_state.hide_target:
            target_rgb[:] = 0

        # Combine and scale back to 0–255 uint8
        color_rgb = base_rgb + target_rgb
        color_rgb = np.clip(color_rgb, 0.0, 1.0)
        color_rgb = (color_rgb * 255).astype(np.uint8)

        invert_bw = st.checkbox(
            "Invert black/white in composite",
            key="invert_bw_color",
        )
        if invert_bw:
            color_rgb = invert_luminance(color_rgb)

        composite_rgb = color_rgb  # track current composite

        st.subheader("Color overlay of both impressions")
        o1, o2, o3 = st.columns([1, 2, 1])
        with o2:
            st.image(
                color_rgb,
                caption="Base = red, Target = cyan (toggles below)",
                use_container_width=True,
                clamp=True,
            )

    # -------------------------------------------------
    # Download button: originals + current composite
    # -------------------------------------------------
    if composite_rgb is not None:
        combined = stack_triptych(
            base_rgb_original,
            target_rgb_original,
            composite_rgb,
        )
        pil_img = Image.fromarray(combined)
        buf = BytesIO()
        pil_img.save(buf, format="PNG")
        byte_im = buf.getvalue()

        st.markdown("---")
        st.download_button(
            label="📥 Download originals + current composite (PNG)",
            data=byte_im,
            file_name="intaglio_state_comparison.png",
            mime="image/png",
        )
