
import cv2
import numpy as np
from pathlib import Path
from typing import Union


def _load_image(path_or_array: Union[str, Path, np.ndarray]) -> np.ndarray:
    """
    Load an image from a path or return the array unchanged.
    Always returns a uint8 BGR image as read by OpenCV.
    """
    if isinstance(path_or_array, np.ndarray):
        img = path_or_array
        if img.ndim == 2:
            # grayscale, convert to BGR for consistency
            return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        return img
    p = Path(path_or_array)
    if not p.exists():
        raise FileNotFoundError(f"Image not found: {p}")
    img = cv2.imread(str(p), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Failed to read image: {p}")
    return img


def to_grayscale(img_bgr: np.ndarray) -> np.ndarray:
    """Convert BGR image to grayscale uint8."""
    if img_bgr.ndim == 2:
        # already grayscale
        return img_bgr
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)


def resize_max_width(img: np.ndarray, max_width: int = 2000) -> np.ndarray:
    """
    Resize image so that its width is at most max_width,
    preserving aspect ratio. If width is already <= max_width,
    the image is returned unchanged.
    """
    h, w = img.shape[:2]
    if w <= max_width:
        return img
    scale = max_width / float(w)
    new_size = (max_width, int(round(h * scale)))
    return cv2.resize(img, new_size, interpolation=cv2.INTER_AREA)


def normalize_contrast(gray: np.ndarray) -> np.ndarray:
    """
    Apply histogram equalization to mitigate paper tone and
    scanning differences. Input/Output: uint8 grayscale.
    """
    if gray.dtype != np.uint8:
        gray_u8 = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX).astype("uint8")
    else:
        gray_u8 = gray
    return cv2.equalizeHist(gray_u8)


def preprocess(path_or_array: Union[str, Path, np.ndarray],
               max_width: int = 2000) -> np.ndarray:
    """
    Full preprocessing pipeline:
    - load image (if path)
    - convert to grayscale
    - resize to max_width
    - normalize contrast

    Returns a uint8 grayscale image.
    """
    img = _load_image(path_or_array)
    gray = to_grayscale(img)
    resized = resize_max_width(gray, max_width=max_width)
    norm = normalize_contrast(resized)
    return norm
