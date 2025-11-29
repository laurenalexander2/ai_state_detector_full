
import cv2
import numpy as np
from typing import Tuple


def grid_change_features(mask: np.ndarray,
                         grid_shape: Tuple[int, int] = (4, 4)) -> np.ndarray:
    """
    Compute a simple feature vector from a binary change mask by dividing the image
    into a grid and computing the ratio of changed pixels in each cell.

    Parameters
    ----------
    mask : np.ndarray
        Binary mask (0 or 255) of changed regions.
    grid_shape : (rows, cols)
        Number of grid rows and columns.

    Returns
    -------
    features : np.ndarray
        1D feature vector of length rows*cols with change ratios.
    """
    if mask.ndim != 2:
        raise ValueError("Mask must be a 2D array.")
    rows, cols = grid_shape
    h, w = mask.shape
    features = []

    for r in range(rows):
        for c in range(cols):
            r0 = int(r * h / rows)
            r1 = int((r + 1) * h / rows)
            c0 = int(c * w / cols)
            c1 = int((c + 1) * w / cols)
            cell = mask[r0:r1, c0:c1]
            # ratio of changed pixels (mask==255)
            if cell.size == 0:
                features.append(0.0)
            else:
                change_ratio = float((cell == 255).sum()) / float(cell.size)
                features.append(change_ratio)

    return np.array(features, dtype=np.float32)


def global_change_ratio(mask: np.ndarray) -> float:
    """
    Global ratio of changed pixels in the mask.
    """
    if mask.size == 0:
        return 0.0
    return float((mask == 255).sum()) / float(mask.size)
