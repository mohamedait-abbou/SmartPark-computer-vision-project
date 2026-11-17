"""Smoke-test the trained plate detector on a local image.

Place the Colab checkpoint at ``models/plate_detector.pt``, then run from the
project root:

    python scripts/verify_model.py
    python scripts/verify_model.py --image path/to/car.jpg
    python scripts/verify_model.py --image path/to/car.jpg --confidence 0.40

The default image is the first file in the validation split. The annotated
result is written to ``runs/local_check.jpg`` (ignored by Git).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.detection.detector import DEFAULT_MODEL_PATH, PlateDetector
from src.processing.image_utils import (
    draw_boxes,
    iter_image_paths,
    load_image,
    save_image,
)

DEFAULT_IMAGE_DIR = PROJECT_ROOT / "data" / "raw" / "plates" / "images" / "val"
DEFAULT_OUTPUT = PROJECT_ROOT / "runs" / "local_check.jpg"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--image", type=Path, default=None)
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def pick_image(image_path: Path | None) -> Path:
    if image_path is not None:
        path = image_path.expanduser().resolve()
        if not path.is_file():
            raise SystemExit(f"Image not found: {path}")
        return path

    if not DEFAULT_IMAGE_DIR.is_dir():
        raise SystemExit(
            f"Validation images not found: {DEFAULT_IMAGE_DIR}\n"
            "Pass an image explicitly with --image path/to/car.jpg"
        )

    for path in iter_image_paths(DEFAULT_IMAGE_DIR):
        return path

    raise SystemExit(f"No images found in: {DEFAULT_IMAGE_DIR}")


def main() -> int:
    args = parse_args()
    model_path = args.model.expanduser().resolve()
    image_path = pick_image(args.image)

    if not model_path.is_file():
        raise SystemExit(
            f"Model weights not found: {model_path}\n"
            "Download plate_detector.pt from Colab and place it there."
        )

    print(f"Model: {model_path} ({model_path.stat().st_size / 1e6:.1f} MB)")
    print(f"Image: {image_path}")

    image = load_image(image_path)
    print(f"Image size: {image.shape[1]}x{image.shape[0]}")

    detector = PlateDetector(
        model_path=model_path,
        confidence=args.confidence,
        iou=args.iou,
    )
    detections = detector.predict(image)

    print(f"Detections: {len(detections)}")
    for index, detection in enumerate(detections, start=1):
        x1, y1, x2, y2 = detection["box"]
        print(
            f"  {index}. {detection['class_name']} "
            f"conf={detection['confidence']:.3f} "
            f"box=({x1},{y1},{x2},{y2})"
        )

    annotated = draw_boxes(image, [detection["box"] for detection in detections])
    output_path = save_image(args.output, annotated)
    print(f"Saved annotated image: {output_path}")

    if not detections:
        print("WARNING: no plate detected. Try --confidence 0.10 or another image.")
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
