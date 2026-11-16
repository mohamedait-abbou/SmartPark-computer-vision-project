"""Environment and smoke tests for SmartPark Maroc.

Run from the project root with:

    pytest

These tests check that Python, the CPU-only ML stack and the project layout
are ready. They do NOT require model weights or a dataset.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]

EXPECTED_DIRECTORIES = [
    "data/raw",
    "data/processed",
    "models",
    "notebooks",
    "src/detection",
    "src/recognition",
    "src/processing",
    "app",
    "tests",
    "docs",
]

EXPECTED_FILES = [
    "requirements.txt",
    "README.md",
    ".gitignore",
    ".env.example",
    "src/processing/image_utils.py",
    "src/detection/detector.py",
    "src/recognition/ocr.py",
    "app/app.py",
    "tests/test_environment.py",
]


def test_python_version_is_311() -> None:
    """The project is pinned to Python 3.11."""
    assert sys.version_info[:2] == (3, 11), f"Expected Python 3.11, got {sys.version}"


def test_project_structure() -> None:
    """Every required folder and starter file must exist."""
    missing_dirs = [name for name in EXPECTED_DIRECTORIES if not (PROJECT_ROOT / name).is_dir()]
    missing_files = [name for name in EXPECTED_FILES if not (PROJECT_ROOT / name).is_file()]

    assert not missing_dirs, f"Missing directories: {missing_dirs}"
    assert not missing_files, f"Missing files: {missing_files}"


def test_core_dependencies_import() -> None:
    """Import the main libraries and check that versions are readable."""
    import cv2
    import fastapi
    import matplotlib
    import numpy
    import pandas
    import PIL
    import pytest as pytest_module
    import streamlit
    import torch
    import torchvision
    import uvicorn
    from ultralytics import YOLO

    assert cv2.__version__
    assert numpy.__version__
    assert pandas.__version__
    assert matplotlib.__version__
    assert PIL.__version__
    assert fastapi.__version__
    assert uvicorn.__version__
    assert streamlit.__version__
    assert pytest_module.__version__
    assert torch.__version__
    assert torchvision.__version__
    assert YOLO is not None


def test_torch_is_cpu_only() -> None:
    """CUDA must not be available and no NVIDIA runtime must be installed."""
    import torch

    assert torch.version.cuda is None, "PyTorch was built with CUDA support."
    assert not torch.cuda.is_available(), "CUDA is available; this setup must stay CPU-only."
    assert "+cpu" in torch.__version__, f"Expected a +cpu wheel, got {torch.__version__}"

    assert importlib.util.find_spec("nvidia") is None, "NVIDIA Python packages are installed."
    assert importlib.util.find_spec("pynvml") is None, "NVIDIA management bindings are installed."


def test_requirements_pin_cpu_torch() -> None:
    """Guard against accidentally switching requirements back to CUDA wheels."""
    requirements = (PROJECT_ROOT / "requirements.txt").read_text(encoding="utf-8")

    assert "https://download.pytorch.org/whl/cpu" in requirements
    assert "torch==2.6.0+cpu" in requirements
    assert "torchvision==0.21.0+cpu" in requirements
    assert "cu121" not in requirements
    assert "cu124" not in requirements


def test_image_utils_roundtrip(tmp_path: Path) -> None:
    """Save, load, resize, crop and convert a synthetic image."""
    from src.processing.image_utils import (
        bgr_to_rgb,
        crop_region,
        load_image,
        resize_image,
        save_image,
        to_grayscale,
    )

    image = np.zeros((40, 80, 3), dtype=np.uint8)
    image[:, :, 1] = 200  # Green channel only.

    image_path = save_image(tmp_path / "sample.png", image)
    loaded = load_image(image_path)

    assert loaded.shape == (40, 80, 3)
    assert resize_image(loaded, width=40).shape == (20, 40, 3)
    assert resize_image(loaded, height=20).shape == (20, 40, 3)
    assert crop_region(loaded, (10, 5, 30, 25)).shape == (20, 20, 3)
    assert bgr_to_rgb(loaded).shape == loaded.shape
    assert to_grayscale(loaded).shape == (40, 80)


def test_image_utils_validation() -> None:
    """Bad inputs should fail with clear errors, not silently."""
    from src.processing.image_utils import resize_image

    image = np.zeros((10, 10, 3), dtype=np.uint8)

    with pytest.raises(ValueError, match="exactly one"):
        resize_image(image)

    with pytest.raises(ValueError, match="exactly one"):
        resize_image(image, width=10, height=10)


def test_decode_image_bytes() -> None:
    """In-memory uploads should decode correctly."""
    import cv2

    from src.processing.image_utils import decode_image_bytes

    image = np.full((15, 25, 3), 127, dtype=np.uint8)
    success, encoded = cv2.imencode(".png", image)
    assert success

    decoded = decode_image_bytes(encoded.tobytes())
    assert decoded.shape == (15, 25, 3)


def test_detector_requires_model_weights(tmp_path: Path) -> None:
    """The detector must fail clearly and must not download weights."""
    from src.detection.detector import PlateDetector

    detector = PlateDetector(model_path=tmp_path / "missing_weights.pt")
    image = np.zeros((32, 32, 3), dtype=np.uint8)

    with pytest.raises(FileNotFoundError, match="not downloaded automatically"):
        detector.predict(image)


def test_detector_rejects_cuda() -> None:
    """The project intentionally refuses CUDA devices."""
    from src.detection.detector import PlateDetector

    with pytest.raises(ValueError, match="CPU only"):
        PlateDetector(device="cuda:0")


def test_ocr_requires_recognizer() -> None:
    """OCR must explain how to configure a recogniser instead of guessing."""
    from src.recognition.ocr import PlateOCR

    plate = np.full((30, 100, 3), 255, dtype=np.uint8)

    with pytest.raises(RuntimeError, match="No OCR recognizer configured"):
        PlateOCR().recognize(plate)


def test_ocr_preprocessing_and_injected_recognizer() -> None:
    """The OCR pipeline should accept a custom recogniser callable."""
    from src.recognition.ocr import PlateOCR, clean_plate_text

    plate = np.full((30, 100, 3), 255, dtype=np.uint8)

    def fake_recognizer(prepared: np.ndarray) -> str:
        assert prepared.ndim == 2
        assert prepared.shape[0] >= 64
        return " 12345-a-6 "

    ocr = PlateOCR(recognizer=fake_recognizer)
    assert ocr.recognize(plate) == "12345-A-6"
    assert clean_plate_text("AB 123 / CD") == "AB123CD"
