"""YOLO licence-plate detector wrapper.

Important:
This project NEVER downloads model weights automatically. Put a trained
YOLO weight file at ``models/plate_detector.pt`` (or pass another path).
See ``README.md`` for details.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from src.processing.image_utils import crop_region

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "plate_detector.pt"


class PlateDetector:
    """Small wrapper around Ultralytics YOLO that always runs on CPU.

    The Ultralytics import is delayed until a model is actually loaded so
    that the rest of the project stays fast and testable.
    """

    def __init__(
        self,
        model_path: str | Path | None = None,
        *,
        confidence: float = 0.25,
        iou: float = 0.45,
        device: str = "cpu",
    ) -> None:
        self.model_path = Path(model_path) if model_path else DEFAULT_MODEL_PATH
        self.confidence = float(confidence)
        self.iou = float(iou)
        self.device = device

        if device != "cpu":
            raise ValueError(
                "SmartPark Maroc is configured for CPU only. "
                "Remove the CUDA device setting to avoid unsupported NVIDIA dependencies."
            )

        self._model: Any | None = None

    @property
    def is_loaded(self) -> bool:
        """True once the YOLO weights have been loaded into memory."""
        return self._model is not None

    def load(self) -> "PlateDetector":
        """Load the YOLO weights lazily."""
        if self._model is not None:
            return self

        if not self.model_path.is_file():
            raise FileNotFoundError(
                f"Detection model not found: {self.model_path}\n"
                "Place a YOLO weight file there, or pass another path. "
                "Model weights are not downloaded automatically."
            )

        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise RuntimeError(
                "Ultralytics is not installed. Run: pip install -r requirements.txt"
            ) from exc

        self._model = YOLO(str(self.model_path))
        return self

    def predict(self, image: np.ndarray) -> list[dict[str, Any]]:
        """Detect plates in a BGR image.

        Returns a list of dictionaries with ``box``, ``confidence``,
        ``class_id`` and ``class_name`` keys.
        """
        if not isinstance(image, np.ndarray) or image.size == 0:
            raise TypeError("predict expects a non-empty NumPy image array.")

        self.load()

        results = self._model.predict(
            source=image,
            conf=self.confidence,
            iou=self.iou,
            device="cpu",
            verbose=False,
        )

        detections: list[dict[str, Any]] = []
        for result in results:
            if result.boxes is None:
                continue

            names = result.names or {}
            for box in result.boxes:
                x1, y1, x2, y2 = (int(value) for value in box.xyxy[0].tolist())
                class_id = int(box.cls[0])

                if isinstance(names, dict):
                    class_name = names.get(class_id, str(class_id))
                else:
                    class_name = names[class_id]

                detections.append(
                    {
                        "box": (x1, y1, x2, y2),
                        "confidence": float(box.conf[0]),
                        "class_id": class_id,
                        "class_name": class_name,
                    }
                )

        return detections

    def detect_and_crop(
        self,
        image: np.ndarray,
        *,
        padding: int = 4,
    ) -> list[tuple[np.ndarray, dict[str, Any]]]:
        """Return ``(cropped_plate, detection)`` pairs."""
        crops: list[tuple[np.ndarray, dict[str, Any]]] = []
        for detection in self.predict(image):
            crop = crop_region(image, detection["box"], padding=padding)
            crops.append((crop, detection))
        return crops
