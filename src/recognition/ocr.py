"""Licence-plate OCR scaffold.

No OCR engine or trained character-recognition model is bundled with this
project. This module provides:

1. A clean image-preprocessing pipeline for plate crops.
2. A pluggable interface (``recognizer`` callable) for your future model.
3. Small text-normalisation helpers.

Beginners can start by wiring Tesseract, EasyOCR, PaddleOCR or a custom
PyTorch CRNN into ``recognizer`` later.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import cv2
import numpy as np

Recognizer = Callable[[np.ndarray], str]


def clean_plate_text(text: str) -> str:
    """Uppercase a plate string and keep letters, digits and hyphens only."""
    if not isinstance(text, str):
        raise TypeError("clean_plate_text expects a string.")

    cleaned = []
    for character in text.upper().strip():
        if character.isalnum() or character == "-":
            cleaned.append(character)
    return "".join(cleaned)


class PlateOCR:
    """Preprocess plate crops and call an injected recognition function."""

    def __init__(self, recognizer: Recognizer | None = None, *, min_height: int = 64) -> None:
        self.recognizer = recognizer
        self.min_height = int(min_height)

    def preprocess(self, plate_image: np.ndarray) -> np.ndarray:
        """Prepare a plate crop for OCR.

        Steps: grayscale -> upscale small crops -> improve contrast ->
        reduce noise -> binarise with Otsu's method.
        """
        if not isinstance(plate_image, np.ndarray) or plate_image.size == 0:
            raise TypeError("preprocess expects a non-empty NumPy image array.")

        if plate_image.ndim == 3:
            gray = cv2.cvtColor(plate_image, cv2.COLOR_BGR2GRAY)
        else:
            gray = plate_image

        height, width = gray.shape[:2]
        if height < self.min_height:
            scale = self.min_height / height
            gray = cv2.resize(
                gray,
                (max(1, round(width * scale)), self.min_height),
                interpolation=cv2.INTER_CUBIC,
            )

        gray = cv2.bilateralFilter(gray, d=5, sigmaColor=50, sigmaSpace=50)

        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)

        _, binary = cv2.threshold(
            enhanced,
            0,
            255,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU,
        )
        return binary

    def recognize(self, plate_image: np.ndarray) -> str:
        """Run the configured recogniser on one plate crop."""
        if self.recognizer is None:
            raise RuntimeError(
                "No OCR recognizer configured. Pass recognizer=<callable> to "
                "PlateOCR. No OCR model or engine is bundled with this project."
            )

        prepared = self.preprocess(plate_image)
        raw_text = self.recognizer(prepared)
        return clean_plate_text(raw_text)

    def recognize_detections(
        self,
        image: np.ndarray,
        detections: list[dict[str, Any]],
        *,
        padding: int = 4,
    ) -> list[dict[str, Any]]:
        """Run OCR on each detection and return the enriched dictionaries."""
        from src.processing.image_utils import crop_region

        results: list[dict[str, Any]] = []
        for detection in detections:
            crop = crop_region(image, detection["box"], padding=padding)
            enriched = dict(detection)
            enriched["text"] = self.recognize(crop)
            results.append(enriched)
        return results
