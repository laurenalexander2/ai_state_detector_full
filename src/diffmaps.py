import cv2
import numpy as np
from typing import Tuple


def edge_difference(
    base: np.ndarray,
    aligned: np.ndarray,
    low_threshold: int = 80,
    high_threshold: int = 220,
    blur_ksize: int = 5,
) -> np.ndarray:
    """
    Compute an edge-based absolute difference map between two aligned grayscale images.
    This version first blurs the images to suppress paper grain / noise, then runs Canny.

    Returns a uint8 image with high values where edges differ.
    """
    if base.shape != aligned.shape:
        raise ValueError("Images must have the same shape for difference computation.")

    # Smooth to reduce tiny paper-texture edges
    base_blur = cv2.GaussianBlur(base, (blur_ksize, blur_ksize), 0)
    aligned_blur = cv2.GaussianBlur(aligned, (blur_ksize, blur_ksize), 0)

    edges1 = cv2.Canny(base_blur, low_threshold, high_threshold)
    edges2 = cv2.Canny(aligned_blur, low_threshold, high_threshold)

    diff = cv2.absdiff(edges1, edges2)
    return diff


def compute_diff_mask(
    base: np.ndarray,
    aligned: np.ndarray,
    use_edges: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute a normalized difference image and a binary mask highlighting
    significant changes between base and aligned images.

    Two modes:

    - use_edges = False  → raw grayscale difference (what you already had working)
    - use_edges = True   → *ink-line difference*: focuses only on very dark marks
                           (intaglio ink) and compares those binary "ink masks".

    Parameters
    ----------
    base : np.ndarray
        Grayscale base image.
    aligned : np.ndarray
        Grayscale aligned image.
    use_edges : bool
        If True, use ink-based difference on very dark marks.
        If False, use raw pixel difference.

    Returns
    -------
    norm_diff : np.ndarray
        Difference image (uint8) used for the left panel.
    mask : np.ndarray
        Binary mask of significant differences (0 or 255) used for overlay.
    """
    if base.shape != aligned.shape:
        raise ValueError("Base and aligned images must have the same shape.")

    if use_edges:
        # --- NEW MODE: "ink-only" line difference -------------------------
        # We ignore most paper tone and only look at the darkest pixels
        # (where the etched lines really bite).

        # Adaptive threshold so dark ink becomes white (255), paper becomes 0.
        # THRESH_BINARY_INV because ink is dark.
        ink1 = cv2.adaptiveThreshold(
            base, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            35,   # block size (odd)
            10    # C: tweak to be a bit stricter/looser
        )
        ink2 = cv2.adaptiveThreshold(
            aligned, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            35,
            10
        )

        # Symmetric difference: pixels where one has ink and the other doesn't.
        diff_ink = cv2.bitwise_xor(ink1, ink2)

        # Light morphological opening to remove single-pixel junk.
        kernel = np.ones((3, 3), np.uint8)
        opened = cv2.morphologyEx(diff_ink, cv2.MORPH_OPEN, kernel, iterations=1)

        # Remove tiny connected components (e.g. stray dots)
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            opened, connectivity=8
        )
        clean_mask = opened.copy()
        min_area = 30  # you can tune this
        for label in range(1, num_labels):  # skip background
            area = stats[label, cv2.CC_STAT_AREA]
            if area < min_area:
                clean_mask[labels == label] = 0

        # For the left panel we can just show the cleaned ink-difference map.
        norm_diff = clean_mask.copy()

        return norm_diff.astype("uint8"), clean_mask.astype("uint8")

    else:
        # --- ORIGINAL RAW GRAYSCALE DIFFERENCE MODE -----------------------
        diff = cv2.absdiff(base, aligned)

        # Normalize difference for thresholding.
        norm_diff = cv2.normalize(diff, None, 0, 255, cv2.NORM_MINMAX)

        # Otsu threshold to get binary mask.
        _, mask = cv2.threshold(
            norm_diff, 0, 255,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )

        # Morphological opening to remove small specks.
        kernel = np.ones((3, 3), np.uint8)
        clean_mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

        return norm_diff.astype("uint8"), clean_mask.astype("uint8")




def create_overlay(
    base: np.ndarray,
    mask: np.ndarray,
    alpha: float = 0.5,
) -> np.ndarray:
    """
    Create a red overlay visualization of the difference mask on top of a base image.

    Parameters
    ----------
    base : np.ndarray
        Grayscale base image.
    mask : np.ndarray
        Binary mask (0 or 255) of changed regions.
    alpha : float
        Opacity of the red overlay.

    Returns
    -------
    overlay_bgr : np.ndarray
        BGR image suitable for saving or display.
    """
    if base.ndim != 2:
        raise ValueError("Base image must be grayscale.")
    if mask.shape != base.shape:
        raise ValueError("Mask must have same shape as base image.")

    base_u8 = base.astype("uint8")
    base_bgr = cv2.cvtColor(base_u8, cv2.COLOR_GRAY2BGR)

    # Paint changed pixels red
    overlay = base_bgr.copy()
    overlay[mask == 255] = (0, 0, 255)  # BGR red

    # Blend overlay with original
    blended = cv2.addWeighted(overlay, alpha, base_bgr, 1 - alpha, 0)
    return blended

def compute_edge_line_mask(
    base: np.ndarray,
    aligned: np.ndarray,
    blur_ksize: int = 3,
    low_threshold: int = 60,
    high_threshold: int = 180,
    min_component_area: int = 40,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    More precise *line-based* difference.

    Steps:
    - light blur to reduce paper noise
    - Canny edges on both images
    - XOR of the two edge maps (lines present in one but not the other)
    - remove tiny connected components

    Returns
    -------
    edge_diff : np.ndarray
        Edge difference image for display (uint8).
    mask : np.ndarray
        Cleaned binary mask (0 or 255) of changed line segments.
    """
    if base.shape != aligned.shape:
        raise ValueError("Base and aligned images must have the same shape.")

    # Light blur to kill micro-paper texture but keep lines sharp
    base_blur = cv2.GaussianBlur(base, (blur_ksize, blur_ksize), 0)
    aligned_blur = cv2.GaussianBlur(aligned, (blur_ksize, blur_ksize), 0)

    # Canny edges (binary, 0 or 255)
    edges1 = cv2.Canny(base_blur, low_threshold, high_threshold)
    edges2 = cv2.Canny(aligned_blur, low_threshold, high_threshold)

    # XOR: pixels where only one image has an edge → changed lines
    edge_diff = cv2.bitwise_xor(edges1, edges2)

    # Optionally thicken very thin edges a bit
    kernel = np.ones((3, 3), np.uint8)
    thick = cv2.dilate(edge_diff, kernel, iterations=1)

    # Remove tiny specks via connected components
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        thick, connectivity=8
    )
    clean_mask = np.zeros_like(thick)
    for label in range(1, num_labels):  # skip background
        area = stats[label, cv2.CC_STAT_AREA]
        if area >= min_component_area:
            clean_mask[labels == label] = 255

    # For the left panel we can just show the cleaned XOR edges.
    return clean_mask.astype("uint8"), clean_mask.astype("uint8")


def color_overlay_two(
    base: np.ndarray,
    aligned: np.ndarray,
) -> np.ndarray:
    """
    Create a color overlay where:

    - base impression is tinted RED
    - aligned impression is tinted CYAN (G+B)

    Lines present in both appear dark/black,
    lines unique to one impression are colored.

    Returns a BGR image (for OpenCV/Streamlit).
    """
    # Normalize both to full 0–255 range
    base_norm = cv2.normalize(base, None, 0, 255, cv2.NORM_MINMAX).astype("uint8")
    aligned_norm = cv2.normalize(aligned, None, 0, 255, cv2.NORM_MINMAX).astype("uint8")

    # Invert so dark ink = high values (easier to see as bright color)
    base_inv = 255 - base_norm
    aligned_inv = 255 - aligned_norm

    # base -> red channel, aligned -> green+blue (cyan)
    r = base_inv
    g = aligned_inv
    b = aligned_inv

    overlay_bgr = cv2.merge([b, g, r])
    return overlay_bgr
