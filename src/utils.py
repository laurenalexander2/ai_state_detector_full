
import cv2
import numpy as np
from pathlib import Path
from typing import Union


def save_image(path: Union[str, Path], img: np.ndarray) -> None:
    """
    Save an image (grayscale or BGR) to disk. Path parent directories
    are created if needed.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(p), img)


def bgr_to_rgb(img: np.ndarray) -> np.ndarray:
    """
    Convert BGR (OpenCV) to RGB (for matplotlib/Streamlit).
    """
    if img.ndim == 2:
        return img
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
