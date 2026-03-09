# How to Run — Multimodal HAR (UTD-MHAD)

## Prerequisites

- Python 3.9+
- A CUDA-capable GPU (tested on NVIDIA RTX 6000 Ada, 48 GB VRAM)
- `uv` package manager ([installation guide](https://docs.astral.sh/uv/getting-started/installation/))

## 1. Install `uv`

If you don't have `uv` installed:

```bash
# Linux / macOS
curl -LsSf https://astral.sh/uv/install.sh | sh

# Or via pip
pip install uv
```

After installation, verify:

```bash
uv --version
```

## 2. Set Up the Environment

From the project root (`multimodal_har/`):

```bash
cd multimodal_har

# Install all dependencies from pyproject.toml (creates .venv/ and uv.lock)
uv sync
```

This creates a virtual environment in `.venv/` and installs all dependencies
(PyTorch, torchvision, OpenCV, scikit-learn, matplotlib, etc.) as specified in
`pyproject.toml`. The lock file `uv.lock` ensures reproducible builds.

## 3. Download the Dataset

Download the UTD-MHAD dataset from [Google Drive](https://drive.google.com/drive/folders/1iMZ68ASWTvBomP2U10qhAIsaQ2z-XTta)
and place the files so the structure is:

```
multimodal_har/
  data/
    RGB/
      a1_s1_t1_color.avi
      a1_s1_t2_color.avi
      ...
    Inertial/
      a1_s1_t1_inertial.mat
      a1_s1_t2_inertial.mat
      ...
```

There should be 861 `.avi` files in `data/RGB/` and 861 `.mat` files in `data/Inertial/`.

## 4. Run the Notebooks

### Part 1 — Exploratory Data Analysis

```bash
uv run jupyter notebook notebooks/part1_eda.ipynb
```

### Parts 2, 3 & 4 — Models, Fusion & Missing Modality

**Important:** If multiple GPUs are present, specify the target GPU to avoid
out-of-memory errors on busy devices:

```bash
CUDA_VISIBLE_DEVICES=0 uv run jupyter notebook notebooks/parts2_3_4.ipynb
```

To execute a notebook non-interactively (e.g., for CI or reproducibility):

```bash
CUDA_VISIBLE_DEVICES=0 uv run jupyter nbconvert \
  --to notebook --execute \
  notebooks/parts2_3_4.ipynb \
  --output parts2_3_4.ipynb \
  --ExecutePreprocessor.timeout=7200
```

## 5. Project Structure

```
multimodal_har/
├── pyproject.toml          # Dependencies (uv-compatible)
├── uv.lock                 # Reproducible lock file
├── HOW_TO_RUN.md           # This file
├── data/
│   ├── RGB/                # 861 .avi video files
│   └── Inertial/           # 861 .mat sensor files
├── notebooks/
│   ├── part1_eda.ipynb     # Part 1: EDA & data issues
│   └── parts2_3_4.ipynb    # Parts 2-4: unimodal, fusion, missing modality
└── src/
    ├── __init__.py
    ├── data/
    │   ├── __init__.py
    │   ├── dataset.py      # UTDMHADDataset (PyTorch Dataset + DataLoaders)
    │   └── preprocessing.py # Video frame sampling, inertial loading
    ├── models/
    │   ├── __init__.py
    │   ├── inertial_model.py # InertialCNN (1D-CNN for sensor data)
    │   ├── video_model.py    # VideoClassifier (R3D-18 pretrained)
    │   └── fusion_model.py   # MultimodalFusion (late fusion)
    └── utils/
        ├── __init__.py
        ├── config.py         # All constants and hyperparameters
        └── training.py       # Training loops, evaluation, grid search
```

## 6. Estimated Runtimes (RTX 6000 Ada)

| Task | Approximate Time |
|------|-----------------|
| Part 1 EDA | ~2 min |
| Part 2A Inertial grid search (4 configs × 20 epochs) | ~3 min |
| Part 2B Video grid search (4 configs × 15 epochs) | ~8 min |
| Part 3 Fusion training (20 epochs) | ~2 min |
| Part 4 Robust fusion training (20 epochs) | ~2 min |
| **Total (Parts 2-4)** | **~15 min** |
