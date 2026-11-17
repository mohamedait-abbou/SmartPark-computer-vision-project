# Detailed Setup and Troubleshooting

This is the full environment and setup guide. For the project overview and
current status, see the [README](../README.md).

---

# SmartPark Maroc

AI-powered Moroccan parking management: licence-plate **detection** (YOLO)
and **character recognition** (OCR).

This repository contains the local development environment, starter code,
tests and documentation. It is optimised for a **CPU-only laptop** with
8 GB RAM (HP EliteBook 840 G3, Intel Core i5-6300U).

---

## Project location

```text
/home/mhd/projects/cv_project/SmartPark-Maroc
```

## Machine profile and design choices

| Item | Value |
| --- | --- |
| OS | Ubuntu 26.04.1 LTS |
| CPU | Intel Core i5-6300U (4 threads) |
| RAM | 8 GB (7.1 GiB usable) |
| GPU | Intel integrated graphics (no CUDA) |
| Python | 3.11 (Conda environment `smartpark`) |
| PyTorch | `2.6.0+cpu` from the official PyTorch CPU index |
| Detection | Ultralytics YOLO, CPU inference |

Why CPU-only PyTorch? The default Linux PyPI wheel pulls in large NVIDIA
CUDA libraries that cannot be used on this laptop. The `+cpu` wheel avoids
those downloads and saves both disk space and RAM.

---

## Project structure

```text
SmartPark-Maroc/
├── app/                  # Streamlit demo UI
│   └── app.py
├── data/
│   ├── raw/              # original images (never committed)
│   └── processed/        # resized/cropped images (never committed)
├── docs/                 # project documentation
├── models/               # model weights (never committed)
├── notebooks/            # Jupyter experiments
│   └── 01_environment_check.ipynb
├── src/
│   ├── detection/        # YOLO plate detector wrapper
│   │   └── detector.py
│   ├── processing/       # reusable OpenCV helpers
│   │   └── image_utils.py
│   └── recognition/      # OCR preprocessing + pluggable recogniser
│       └── ocr.py
├── tests/
│   └── test_environment.py
├── .env.example
├── .gitignore
├── pytest.ini
├── README.md
└── requirements.txt
```

---

## 1. Activate the environment

```bash
conda activate smartpark
python --version        # expected: Python 3.11.x
```

If `conda activate` is not available in your shell, load Conda first:

```bash
source ~/anaconda3/etc/profile.d/conda.sh
conda activate smartpark
```

> Your Anaconda installation prints plugin warnings
> (`conda-anaconda-telemetry`, `repo`, `conda-anaconda-tos`) because some
> plugins are older than Conda itself. These warnings are **pre-existing**
> and harmless; environment creation and activation still work.

---

## 2. Install or reinstall dependencies

Everything is already installed. To reproduce the environment on another
machine:

```bash
conda create -y -n smartpark python=3.11 pip --no-default-packages
conda activate smartpark
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

`requirements.txt` points pip at `https://download.pytorch.org/whl/cpu`
for `torch==2.6.0+cpu` and `torchvision==0.21.0+cpu`. Do not replace these
with plain `torch` from PyPI: that wheel would download CUDA/NVIDIA
libraries.

---

## 3. Verify the installation

```bash
conda activate smartpark
python -c "import sys; print(sys.version)"
python -c "import torch; print(torch.__version__, 'CUDA:', torch.cuda.is_available())"
python -c "import cv2; print('OpenCV', cv2.__version__)"
python -c "from ultralytics import YOLO; import ultralytics; print('Ultralytics', ultralytics.__version__)"
pytest
```

Expected results:

- Python is **3.11.x**.
- PyTorch version ends with **`+cpu`**.
- `torch.cuda.is_available()` prints **`False`**.
- All tests pass.

---

## 4. First experiment

The environment-check notebook works without any dataset or model weights:

```bash
conda activate smartpark
cd ~/projects/cv_project/SmartPark-Maroc
jupyter lab notebooks/01_environment_check.ipynb
```

Run all cells. It prints versions, confirms that CUDA is disabled and
exercises the image-processing utilities on a synthetic image.

You can also launch the Streamlit preview:

```bash
conda activate smartpark
cd ~/projects/cv_project/SmartPark-Maroc
streamlit run app/app.py
```

Then open <http://localhost:8501> and upload a vehicle photo. Image preview
and preprocessing work immediately; detection needs model weights (below).

---

## 5. Adding AI models (manual step)

No model weights are included or downloaded automatically.

### Plate detector

1. Train or obtain a YOLO model that detects licence plates.
2. Copy the weights to `models/plate_detector.pt`.
3. Pass the path in the app sidebar or to `PlateDetector(model_path=...)`.

The detector always runs on CPU and refuses `cuda` devices on purpose.

### OCR recogniser

`src/recognition/ocr.py` contains the preprocessing pipeline and a clean
plug-in point. No OCR engine is bundled. To use Tesseract, EasyOCR,
PaddleOCR or a custom CRNN later, pass a callable:

```python
from src.recognition.ocr import PlateOCR

def my_recognizer(prepared_gray_image):
    # Call your OCR engine here and return the raw text.
    return "12345-A-6"

ocr = PlateOCR(recognizer=my_recognizer)
text = ocr.recognize(plate_crop)
```

---

## 6. Data and Git rules

The `.gitignore` protects the repository. Never commit:

- datasets in `data/raw/` or `data/processed/`;
- model weights (`*.pt`, `*.pth`, `*.onnx`, …) or `models/*`;
- virtual/conda environments;
- `.env`, API keys or any secrets;
- notebook checkpoints, logs and coverage output.

Keep small sample images only if they are legally shareable and contain no
personal data. Moroccan licence plates are personal data: anonymise images
before sharing them.

Useful Git checks:

```bash
git status
git check-ignore -v models/plate_detector.pt
git check-ignore -v .env
```

---

## 7. Performance tips for 8 GB RAM / Intel CPU

- Run **one** heavy tool at a time (Jupyter, Streamlit or training, not all).
- Use YOLO inference at `imgsz=640` or `416` and batch size `1`.
- Resize very large phone photos before processing (e.g. max width 1280).
- Keep `device="cpu"`; never try to force CUDA.
- Close unused browser tabs; Chromium-based browsers can use >1 GB.
- The machine already swaps heavily under load; expect slower first runs
  while Python imports PyTorch and Ultralytics.

---

## 8. Troubleshooting

| Problem | Fix |
| --- | --- |
| `conda: command not found` | `source ~/anaconda3/etc/profile.d/conda.sh` |
| Conda plugin warnings | Pre-existing, harmless; do not modify `base` to fix them |
| `torch.cuda.is_available()` is `True` | Wrong wheel installed; reinstall with `pip install -r requirements.txt` |
| `ModuleNotFoundError: src` | Run commands from the project root, or use the provided app/test entry points |
| `libGL.so.1: cannot open shared object file` | System OpenGL library is missing; OpenCV GUI features need it (ask before using `sudo`) |
| Detection says model not found | Put weights at `models/plate_detector.pt`; nothing is downloaded automatically |
| OCR says no recogniser | Pass `recognizer=<callable>` to `PlateOCR` |

If an install fails, read the **last 30 lines** of the pip output first.
Most failures are network timeouts, a full disk, or a version conflict.
