"""Reusable image helpers for SmartPark Maroc.

Beginner note:
OpenCV (``cv2``) loads images as NumPy arrays in BGR order. Streamlit,
Matplotlib and Pillow usually expect RGB, so use ``bgr_to_rgb`` before
displaying an image with those tools.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import cv2
import numpy as np

Box = tuple[int, int, int, int]

SUPPORTED_IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
    ".tif",
    ".tiff",
}


def load_image(path: str | Path, *, grayscale: bool = False) -> np.ndarray:
    """Read an image from disk.

    Raises a clear error instead of returning ``None`` when the file is
    missing or cannot be decoded.
    """
    image_path = Path(path).expanduser()
    if not image_path.is_file():
        raise FileNotFoundError(f"Image not found: {image_path}")

    flag = cv2.IMREAD_GRAYSCALE if grayscale else cv2.IMREAD_COLOR
    image = cv2.imread(str(image_path), flag)
    if image is None:
        raise ValueError(f"OpenCV could not decode the image: {image_path}")
    return image


def decode_image_bytes(data: bytes, *, grayscale: bool = False) -> np.ndarray:
    """Decode an in-memory image (for example a Streamlit upload).

    This avoids writing temporary files to disk.
    """
    if not data:
        raise ValueError("Cannot decode an empty image buffer.")

    buffer = np.frombuffer(data, dtype=np.uint8)
    flag = cv2.IMREAD_GRAYSCALE if grayscale else cv2.IMREAD_COLOR
    image = cv2.imdecode(buffer, flag)
    if image is None:
        raise ValueError("OpenCV could not decode the uploaded image bytes.")
    return image


def save_image(path: str | Path, image: np.ndarray) -> Path:
    """Write an image to disk, creating parent directories if needed."""
    if not isinstance(image, np.ndarray) or image.size == 0:
        raise ValueError("save_image expects a non-empty NumPy array.")

    output_path = Path(path).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), image):
        raise OSError(f"Could not write image: {output_path}")
    return output_path


def resize_image(
    image: np.ndarray,
    *,
    width: int | None = None,
    height: int | None = None,
) -> np.ndarray:
    """Resize an image while preserving its aspect ratio.

    Provide exactly one of ``width`` or ``height``.
    """
    if (width is None) == (height is None):
        raise ValueError("Provide exactly one of 'width' or 'height'.")

    original_height, original_width = image.shape[:2]
    if width is not None:
        scale = width / original_width
    else:
        scale = height / original_height

    new_size = (
        max(1, round(original_width * scale)),
        max(1, round(original_height * scale)),
    )
    return cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)


def crop_region(image: np.ndarray, box: Sequence[int], padding: int = 0) -> np.ndarray:
    """Crop a ``(x1, y1, x2, y2)`` region, optionally adding padding.

    Coordinates are clamped to the image boundaries so bad boxes do not
    crash the application.
    """
    if len(box) != 4:
        raise ValueError("A box must contain four values: (x1, y1, x2, y2).")

    image_height, image_width = image.shape[:2]
    x1, y1, x2, y2 = (int(value) for value in box)

    x1 = max(0, x1 - padding)
    y1 = max(0, y1 - padding)
    x2 = min(image_width, x2 + padding)
    y2 = min(image_height, y2 + padding)

    if x2 <= x1 or y2 <= y1:
        raise ValueError(f"Invalid crop box after clamping: {(x1, y1, x2, y2)}")

    return image[y1:y2, x1:x2].copy()


def draw_boxes(
    image: np.ndarray,
    boxes: Iterable[Sequence[int]],
    color: tuple[int, int, int] = (0, 200, 0),
    thickness: int = 2,
) -> np.ndarray:
    """Return a copy of the image with rectangles drawn on it."""
    canvas = image.copy()
    for box in boxes:
        if len(box) != 4:
            raise ValueError("A box must contain four values: (x1, y1, x2, y2).")
        x1, y1, x2, y2 = (int(value) for value in box)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, thickness)
    return canvas


def bgr_to_rgb(image: np.ndarray) -> np.ndarray:
    """Convert an OpenCV BGR image to RGB for display."""
    if image.ndim == 2:
        return image
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def to_grayscale(image: np.ndarray) -> np.ndarray:
    """Convert a BGR image to a single-channel grayscale image."""
    if image.ndim == 2:
        return image
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def iter_image_paths(directory: str | Path) -> Iterable[Path]:
    """Yield supported image files from a directory, sorted by name."""
    folder = Path(directory).expanduser()
    if not folder.is_dir():
        raise NotADirectoryError(f"Not a directory: {folder}")

    for path in sorted(folder.iterdir()):
        if path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS:
            yield path
