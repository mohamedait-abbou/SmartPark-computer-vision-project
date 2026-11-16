"""SmartPark Maroc - beginner-friendly Streamlit demo.

Run from the project root:

    streamlit run app/app.py

The app works without model weights: it can display and preprocess an
uploaded image. Plate detection requires a YOLO weight file placed in
``models/plate_detector.pt`` (see README.md). Weights are not downloaded
automatically.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st

from src.detection.detector import DEFAULT_MODEL_PATH, PlateDetector
from src.processing.image_utils import (
    bgr_to_rgb,
    decode_image_bytes,
    draw_boxes,
    to_grayscale,
)
from src.recognition.ocr import PlateOCR

st.set_page_config(
    page_title="SmartPark Maroc",
    page_icon="P",
    layout="wide",
)


def _show_detection_results(
    image: np.ndarray,
    model_path: Path,
    confidence: float,
    run_ocr: bool,
) -> None:
    """Run YOLO detection and render the results."""
    detector = PlateDetector(model_path=model_path, confidence=confidence)

    try:
        detections = detector.predict(image)
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        st.error(str(exc))
        st.info(
            "Place your trained YOLO weights in the models/ directory, or "
            "change the model path in the sidebar. No weights are downloaded "
            "automatically."
        )
        return

    if not detections:
        st.warning("No licence plates were detected in this image.")
        return

    st.success(f"Detected {len(detections)} plate(s).")

    boxes = [detection["box"] for detection in detections]
    annotated = draw_boxes(image, boxes)
    st.subheader("Detections")
    st.image(bgr_to_rgb(annotated), width="stretch")

    ocr = PlateOCR()
    for index, detection in enumerate(detections, start=1):
        x1, y1, x2, y2 = detection["box"]
        crop = image[y1:y2, x1:x2].copy()

        with st.expander(
            f"Plate {index} - confidence {detection['confidence']:.2f}",
            expanded=True,
        ):
            st.image(bgr_to_rgb(crop), width="stretch")

            if run_ocr:
                try:
                    st.write(f"Recognised text: `{ocr.recognize(crop)}`")
                except RuntimeError as exc:
                    st.warning(str(exc))


def main() -> None:
    st.title("SmartPark Maroc")
    st.caption(
        "AI-powered Moroccan parking management - licence-plate detection demo. "
        "CPU-only, designed for a laptop with 8 GB RAM."
    )

    with st.sidebar:
        st.header("Settings")
        model_path = st.text_input(
            "YOLO weights path",
            value=str(DEFAULT_MODEL_PATH),
            help="Relative to the project root, or an absolute path.",
        )
        confidence = st.slider(
            "Detection confidence",
            min_value=0.05,
            max_value=0.95,
            value=0.25,
            step=0.05,
        )
        run_detection = st.checkbox("Run plate detection", value=False)
        run_ocr = st.checkbox(
            "Run OCR (requires a configured OCR recogniser)",
            value=False,
        )

    uploaded_file = st.file_uploader(
        "Upload a vehicle image",
        type=["jpg", "jpeg", "png", "bmp", "webp"],
    )

    if uploaded_file is None:
        st.info(
            "Upload an image to begin. The preview and preprocessing work "
            "without any AI model."
        )
        st.code(
            "conda activate smartpark\n"
            "cd ~/projects/cv_project/SmartPark-Maroc\n"
            "streamlit run app/app.py",
            language="bash",
        )
        return

    try:
        image = decode_image_bytes(uploaded_file.read())
    except ValueError as exc:
        st.error(str(exc))
        return

    left, right = st.columns(2)
    with left:
        st.subheader("Original image")
        st.image(bgr_to_rgb(image), width="stretch")
    with right:
        st.subheader("Grayscale preprocessing")
        st.image(to_grayscale(image), width="stretch", clamp=True)

    if run_detection:
        settings_path = Path(model_path).expanduser()
        if not settings_path.is_absolute():
            settings_path = PROJECT_ROOT / settings_path
        _show_detection_results(image, settings_path, confidence, run_ocr)


if __name__ == "__main__":
    main()
