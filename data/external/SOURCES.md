# External test datasets (metadata only)

Images and labels in this folder are **not committed**. They are independent
test data, used once to evaluate the trained detector honestly. Do not train,
tune, or select thresholds on these splits.

## Algerian License Plates Detection Dataset

- Source: <https://huggingface.co/datasets/tobni/algerian-license-plates>
- Author: Tobni Mohamed Islam (2026)
- License: **CC BY 4.0** — attribution required if results are published.
- Downloaded: `images/test/` (195 JPG) + `labels/test/` (195 YOLO txt),
  class 0 = plate, normalized `cx cy w h` boxes.
- Local path: `data/external/algerian_license_plates/`
- Why: unseen, Arabic-script North-African plates with a publisher-provided
  test split. Domain shift warning: Algerian plate layouts differ from
  Moroccan ones, so failures may reflect format, not only robustness.

## License Plate Detection (cleaned Roboflow COCO export)

- Source: <https://huggingface.co/datasets/justjuu/license-plate-detection>
- License: **CC BY 4.0** — attribution required if results are published.
- Downloaded: `data/test-00000-of-00001.parquet` (882 test images, COCO bboxes,
  class `license_plate`).
- Local path: `data/external/justjuu_license_plate_detection/`
- Why: generic, unseen cameras/countries; extra generalization stress test.
  Not Moroccan — report separately from the North-African test.
