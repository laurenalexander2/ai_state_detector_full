# align.py

import cv2
import numpy as np


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
        100,      # max iterations
        1e-4,     # convergence epsilon
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


def align_images(
    base_gray: np.ndarray,
    target_gray: np.ndarray,
    min_matches: int = 40,
    ratio_thresh: float = 0.75,
):
    """
    Align target_gray onto base_gray.

    1. Try AKAZE + BFMatcher + RANSAC homography with strict validation.
    2. If that fails or looks degenerate, fall back to ECC affine alignment.
    """

    h, w = base_gray.shape[:2]

    # ---- STEP 1: AKAZE features ----
    akaze = cv2.AKAZE_create()

    kp1, des1 = akaze.detectAndCompute(base_gray, None)
    kp2, des2 = akaze.detectAndCompute(target_gray, None)

    # If we don't have enough features at all, go straight to ECC
    if des1 is None or des2 is None or len(kp1) < 10 or len(kp2) < 10:
        return _ecc_fallback(base_gray, target_gray)

    # ---- STEP 2: Match with BFMatcher (Hamming) + Lowe ratio test ----
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    matches = bf.knnMatch(des2, des1, k=2)  # query: target, train: base

    good = []
    for m, n in matches:
        if m.distance < ratio_thresh * n.distance:
            good.append(m)

    if len(good) < min_matches:
        # Not enough reliable correspondences → use ECC
        return _ecc_fallback(base_gray, target_gray)

    # ---- STEP 3: Estimate homography with RANSAC ----
    src_pts = np.float32(
        [kp2[m.queryIdx].pt for m in good]
    ).reshape(-1, 1, 2)  # target
    dst_pts = np.float32(
        [kp1[m.trainIdx].pt for m in good]
    ).reshape(-1, 1, 2)  # base

    H, inlier_mask = cv2.findHomography(
        src_pts, dst_pts, cv2.RANSAC, ransacReprojThreshold=1.5
    )

    # ---- STEP 4: Validate homography ----
    def _invalid_H(H_mat, mask):
        if H_mat is None:
            return True
        if not np.all(np.isfinite(H_mat)):
            return True

        det = np.linalg.det(H_mat)
        # Reject crazy scales / flips
        if det <= 0 or det < 0.1 or det > 10:
            return True

        if mask is None:
            return True
        inliers = int(mask.sum())
        # Require most of the good matches to be inliers
        if inliers < 0.7 * min_matches:
            return True

        return False

    if _invalid_H(H, inlier_mask):
        # Fall back to a safer, non-crazy alignment
        return _ecc_fallback(base_gray, target_gray)

    # ---- STEP 5: Warp target using the validated homography ----
    aligned = cv2.warpPerspective(
        target_gray,
        H,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )

    return aligned, H
