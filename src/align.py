
import cv2
import numpy as np
from typing import Tuple


def align_images(base: np.ndarray,
                 target: np.ndarray,
                 max_features: int = 5000,
                 good_match_percent: float = 0.15) -> Tuple[np.ndarray, np.ndarray]:
    """
    Align `target` image to `base` image using ORB keypoints and homography.

    Parameters
    ----------
    base : np.ndarray
        Preprocessed grayscale base image.
    target : np.ndarray
        Preprocessed grayscale target image to be warped onto base.
    max_features : int
        Maximum number of ORB features to detect.
    good_match_percent : float
        Fraction of top matches to keep for homography estimation.

    Returns
    -------
    aligned_target : np.ndarray
        Target image warped into the coordinate system of base.
    H : np.ndarray
        3x3 homography matrix mapping target -> base.
    """
    if base.ndim != 2 or target.ndim != 2:
        raise ValueError("align_images expects grayscale images.")

    # Detect ORB features and descriptors.
    orb = cv2.ORB_create(max_features)
    keypoints1, descriptors1 = orb.detectAndCompute(base, None)
    keypoints2, descriptors2 = orb.detectAndCompute(target, None)

    if descriptors1 is None or descriptors2 is None:
        raise RuntimeError("Could not find features in one or both images.")

    # Match features.
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = matcher.match(descriptors1, descriptors2)
    if len(matches) < 4:
        raise RuntimeError("Not enough matches to compute homography.")

    # Sort matches by score and take top fraction.
    matches = sorted(matches, key=lambda x: x.distance)
    num_good = max(4, int(len(matches) * good_match_percent))
    matches = matches[:num_good]

    # Extract matched keypoints.
    points1 = np.zeros((len(matches), 2), dtype=np.float32)
    points2 = np.zeros((len(matches), 2), dtype=np.float32)

    for i, match in enumerate(matches):
        points1[i, :] = keypoints1[match.queryIdx].pt
        points2[i, :] = keypoints2[match.trainIdx].pt

    # Find homography.
    H, mask = cv2.findHomography(points2, points1, cv2.RANSAC)
    if H is None:
        raise RuntimeError("Homography estimation failed.")

    height, width = base.shape
    aligned = cv2.warpPerspective(target, H, (width, height))
    return aligned, H
