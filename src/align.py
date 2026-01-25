# align.py

import cv2
import numpy as np


def _apply_clahe(gray: np.ndarray) -> np.ndarray:
    """Apply CLAHE for better feature detection in varying contrast regions."""
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(gray)


def _phase_correlate_align(
    base_gray: np.ndarray, target_gray: np.ndarray
) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Use phase correlation to find translation offset.
    Very robust even when content differs significantly.
    Returns (aligned_image, homography, response_value).
    """
    h, w = base_gray.shape[:2]

    # Resize target to match base
    if target_gray.shape[:2] != (h, w):
        tgt = cv2.resize(target_gray, (w, h), interpolation=cv2.INTER_LINEAR)
    else:
        tgt = target_gray.copy()

    # Convert to float for phase correlation
    base_f = base_gray.astype(np.float32)
    tgt_f = tgt.astype(np.float32)

    # Apply windowing to reduce edge effects
    hann_y = np.hanning(h).reshape(-1, 1).astype(np.float32)
    hann_x = np.hanning(w).reshape(1, -1).astype(np.float32)
    window = hann_y * hann_x

    base_windowed = base_f * window
    tgt_windowed = tgt_f * window

    # Phase correlation
    (dx, dy), response = cv2.phaseCorrelate(tgt_windowed, base_windowed)

    # Create translation matrix
    M = np.float32([[1, 0, dx], [0, 1, dy]])

    aligned = cv2.warpAffine(
        tgt,
        M,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )

    # Convert to 3x3 homography
    H = np.eye(3, dtype=np.float32)
    H[0, 2] = dx
    H[1, 2] = dy

    return aligned, H, response


def _ecc_refine(
    base_gray: np.ndarray,
    target_warped: np.ndarray,
    initial_warp: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Refine alignment using ECC for subpixel accuracy.
    Returns (refined_image, warp_matrix as 3x3 homography).
    """
    h, w = base_gray.shape[:2]

    # ECC expects float32 in [0,1]
    im1 = base_gray.astype(np.float32) / 255.0
    im2 = target_warped.astype(np.float32) / 255.0

    # Use translation-only for refinement (we already have coarse alignment)
    warp_mode = cv2.MOTION_TRANSLATION
    warp_matrix = np.eye(2, 3, dtype=np.float32)

    criteria = (
        cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
        200,  # max iterations
        1e-5,  # convergence epsilon
    )

    try:
        cc, warp_matrix = cv2.findTransformECC(
            im1, im2, warp_matrix, warp_mode, criteria
        )

        refined = cv2.warpAffine(
            target_warped,
            warp_matrix,
            (w, h),
            flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )

        # Embed translation in a 3x3 homography
        H_refine = np.eye(3, dtype=np.float32)
        H_refine[:2, :3] = warp_matrix
        return refined, H_refine
    except cv2.error:
        # If ECC fails, return input unchanged
        return target_warped, np.eye(3, dtype=np.float32)


def _ecc_fallback(base_gray: np.ndarray, target_gray: np.ndarray):
    """
    Fallback alignment using ECC with an affine transform.
    This will never produce the starburst warp – at worst it gives a gentle
    global alignment (rotation/scale/translation) or just a resized image.
    """
    h, w = base_gray.shape[:2]

    # Resize target to base size for ECC
    if target_gray.shape[:2] != (h, w):
        tgt = cv2.resize(target_gray, (w, h), interpolation=cv2.INTER_LINEAR)
    else:
        tgt = target_gray.copy()

    # ECC expects float32 in [0,1]
    im1 = base_gray.astype(np.float32) / 255.0
    im2 = tgt.astype(np.float32) / 255.0

    warp_mode = cv2.MOTION_AFFINE
    warp_matrix = np.eye(2, 3, dtype=np.float32)

    criteria = (
        cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
        200,  # max iterations
        1e-5,  # convergence epsilon
    )

    try:
        cc, warp_matrix = cv2.findTransformECC(
            im1, im2, warp_matrix, warp_mode, criteria
        )

        aligned = cv2.warpAffine(
            tgt,
            warp_matrix,
            (w, h),
            flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )

        # Embed affine in a 3x3 homography for compatibility
        H = np.eye(3, dtype=np.float32)
        H[:2, :3] = warp_matrix
        return aligned, H
    except cv2.error:
        # ECC can fail for very weird inputs; in that case just resize + identity H
        H = np.eye(3, dtype=np.float32)
        return tgt, H


def _edge_based_alignment(
    base_gray: np.ndarray, target_gray: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """
    Edge-based alignment fallback. Uses Canny edges which are more stable
    across different print states than texture features.
    """
    h, w = base_gray.shape[:2]

    # Resize target to base size
    if target_gray.shape[:2] != (h, w):
        tgt = cv2.resize(target_gray, (w, h), interpolation=cv2.INTER_LINEAR)
    else:
        tgt = target_gray.copy()

    # Extract edges
    base_edges = cv2.Canny(base_gray, 50, 150)
    target_edges = cv2.Canny(tgt, 50, 150)

    # Dilate edges slightly to improve matching
    kernel = np.ones((3, 3), np.uint8)
    base_edges = cv2.dilate(base_edges, kernel, iterations=1)
    target_edges = cv2.dilate(target_edges, kernel, iterations=1)

    # Use ECC on edge images
    im1 = base_edges.astype(np.float32) / 255.0
    im2 = target_edges.astype(np.float32) / 255.0

    warp_mode = cv2.MOTION_AFFINE
    warp_matrix = np.eye(2, 3, dtype=np.float32)

    criteria = (
        cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
        200,
        1e-5,
    )

    try:
        cc, warp_matrix = cv2.findTransformECC(
            im1, im2, warp_matrix, warp_mode, criteria
        )

        aligned = cv2.warpAffine(
            tgt,
            warp_matrix,
            (w, h),
            flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )

        H = np.eye(3, dtype=np.float32)
        H[:2, :3] = warp_matrix
        return aligned, H
    except cv2.error:
        H = np.eye(3, dtype=np.float32)
        return tgt, H


def _try_feature_alignment(
    base_gray: np.ndarray,
    target_gray: np.ndarray,
    detector_name: str = "sift",
    min_matches: int = 20,
    ratio_thresh: float = 0.8,
) -> tuple[np.ndarray | None, np.ndarray | None, int]:
    """
    Try feature-based alignment with specified detector.
    Returns (aligned_image, homography, num_inliers) or (None, None, 0) on failure.
    """
    h, w = base_gray.shape[:2]

    # Apply CLAHE for better feature detection
    base_clahe = _apply_clahe(base_gray)
    target_clahe = _apply_clahe(target_gray)

    # Create detector
    if detector_name == "sift":
        detector = cv2.SIFT_create(nfeatures=5000)
        norm_type = cv2.NORM_L2
    else:  # akaze
        detector = cv2.AKAZE_create()
        norm_type = cv2.NORM_HAMMING

    kp1, des1 = detector.detectAndCompute(base_clahe, None)
    kp2, des2 = detector.detectAndCompute(target_clahe, None)

    if des1 is None or des2 is None or len(kp1) < 10 or len(kp2) < 10:
        return None, None, 0

    # Match with BFMatcher + Lowe ratio test
    bf = cv2.BFMatcher(norm_type, crossCheck=False)
    matches = bf.knnMatch(des2, des1, k=2)

    good = []
    for match_pair in matches:
        if len(match_pair) == 2:
            m, n = match_pair
            if m.distance < ratio_thresh * n.distance:
                good.append(m)

    if len(good) < min_matches:
        return None, None, 0

    # Estimate homography with USAC_MAGSAC (more robust than RANSAC)
    src_pts = np.float32([kp2[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp1[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

    try:
        H, inlier_mask = cv2.findHomography(
            src_pts, dst_pts, cv2.USAC_MAGSAC, ransacReprojThreshold=3.0
        )
    except cv2.error:
        # USAC_MAGSAC not available in older OpenCV, fall back to RANSAC
        H, inlier_mask = cv2.findHomography(
            src_pts, dst_pts, cv2.RANSAC, ransacReprojThreshold=3.0
        )

    # Validate homography
    if H is None or not np.all(np.isfinite(H)):
        return None, None, 0

    det = np.linalg.det(H)
    if det <= 0 or det < 0.1 or det > 10:
        return None, None, 0

    if inlier_mask is None:
        return None, None, 0

    num_inliers = int(inlier_mask.sum())
    if num_inliers < min_matches * 0.5:
        return None, None, 0

    # Warp target
    aligned = cv2.warpPerspective(
        target_gray,
        H,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )

    return aligned, H, num_inliers


def _multi_scale_phase_correlate(
    base_gray: np.ndarray, target_gray: np.ndarray, levels: int = 3
) -> tuple[np.ndarray, np.ndarray]:
    """
    Multi-scale phase correlation for robust translation detection.
    Works from coarse to fine for better convergence.
    """
    h, w = base_gray.shape[:2]

    # Resize target to match base
    if target_gray.shape[:2] != (h, w):
        tgt = cv2.resize(target_gray, (w, h), interpolation=cv2.INTER_LINEAR)
    else:
        tgt = target_gray.copy()

    total_dx, total_dy = 0.0, 0.0

    # Build pyramids
    base_pyr = [base_gray]
    tgt_pyr = [tgt]

    for _ in range(levels - 1):
        base_pyr.append(
            cv2.pyrDown(base_pyr[-1])
        )
        tgt_pyr.append(
            cv2.pyrDown(tgt_pyr[-1])
        )

    # Process from coarsest to finest
    for level in range(levels - 1, -1, -1):
        scale = 2 ** level
        base_level = base_pyr[level]
        tgt_level = tgt_pyr[level]

        # Apply accumulated shift at this level
        if total_dx != 0 or total_dy != 0:
            M = np.float32([[1, 0, total_dx / scale], [0, 1, total_dy / scale]])
            tgt_level = cv2.warpAffine(
                tgt_level,
                M,
                (tgt_level.shape[1], tgt_level.shape[0]),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
            )

        # Phase correlate at this level
        base_f = base_level.astype(np.float32)
        tgt_f = tgt_level.astype(np.float32)

        lh, lw = base_level.shape[:2]
        hann_y = np.hanning(lh).reshape(-1, 1).astype(np.float32)
        hann_x = np.hanning(lw).reshape(1, -1).astype(np.float32)
        window = hann_y * hann_x

        (dx, dy), response = cv2.phaseCorrelate(
            tgt_f * window, base_f * window
        )

        # Accumulate at full scale
        total_dx += dx * scale
        total_dy += dy * scale

    # Apply final translation
    M = np.float32([[1, 0, total_dx], [0, 1, total_dy]])
    aligned = cv2.warpAffine(
        tgt,
        M,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )

    H = np.eye(3, dtype=np.float32)
    H[0, 2] = total_dx
    H[1, 2] = total_dy

    return aligned, H


def align_images(
    base_gray: np.ndarray,
    target_gray: np.ndarray,
    min_matches: int = 20,
    ratio_thresh: float = 0.8,
):
    """
    Align target_gray onto base_gray using a multi-stage robust pipeline.

    Pipeline:
    1. Multi-scale phase correlation (robust translation detection)
    2. Try SIFT feature matching for rotation/scale correction
    3. If SIFT fails, try AKAZE features
    4. Edge-based alignment fallback
    5. ECC refinement for subpixel accuracy
    """
    h, w = base_gray.shape[:2]

    # Resize target to base size if needed
    if target_gray.shape[:2] != (h, w):
        target_resized = cv2.resize(
            target_gray, (w, h), interpolation=cv2.INTER_LINEAR
        )
    else:
        target_resized = target_gray.copy()

    # Stage 0: Multi-scale phase correlation for initial translation
    phase_aligned, H_phase = _multi_scale_phase_correlate(base_gray, target_resized)

    # Stage 1: Try SIFT on phase-aligned image (for any rotation/scale)
    aligned, H_feat, inliers = _try_feature_alignment(
        base_gray, phase_aligned, "sift", min_matches, ratio_thresh
    )

    if aligned is not None and inliers >= min_matches:
        # Refine with ECC for subpixel accuracy
        refined, H_refine = _ecc_refine(base_gray, aligned)
        H_combined = H_refine @ H_feat @ H_phase
        return refined, H_combined

    # Stage 2: Try AKAZE on phase-aligned image
    aligned, H_feat, inliers = _try_feature_alignment(
        base_gray, phase_aligned, "akaze", min_matches, ratio_thresh
    )

    if aligned is not None and inliers >= min_matches:
        refined, H_refine = _ecc_refine(base_gray, aligned)
        H_combined = H_refine @ H_feat @ H_phase
        return refined, H_combined

    # Stage 3: Check if phase correlation alone was good enough
    corr_original = cv2.matchTemplate(
        base_gray, target_resized, cv2.TM_CCOEFF_NORMED
    )[0, 0]
    corr_phase = cv2.matchTemplate(
        base_gray, phase_aligned, cv2.TM_CCOEFF_NORMED
    )[0, 0]

    if corr_phase > corr_original + 0.01:
        # Phase correlation helped, refine with ECC
        refined, H_refine = _ecc_refine(base_gray, phase_aligned)
        H_combined = H_refine @ H_phase
        return refined, H_combined

    # Stage 4: Try edge-based alignment
    aligned, H_edge = _edge_based_alignment(base_gray, target_resized)
    corr_edge = cv2.matchTemplate(base_gray, aligned, cv2.TM_CCOEFF_NORMED)[0, 0]

    if corr_edge > corr_original + 0.01:
        refined, H_refine = _ecc_refine(base_gray, aligned)
        H_combined = H_refine @ H_edge
        return refined, H_combined

    # Stage 5: Final fallback - pure ECC affine
    return _ecc_fallback(base_gray, target_resized)
