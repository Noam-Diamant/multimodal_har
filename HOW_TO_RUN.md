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

## 4. Results — All Notebooks

All results, experiments, and analysis are contained in the **4 notebooks** inside the `notebooks/` folder:

| Notebook | Contents |
|---|---|
| `part1_eda.ipynb` | Part 1: Exploratory Data Analysis & data issues |
| `parts2_3_4.ipynb` | Parts 2–4: Unimodal models, late fusion, missing modality |
| `parts_5_6_7.ipynb` | Parts 5–7: Class imbalance handling & VAE-based augmentation |
| `parts_8_9.ipynb` | Parts 8–9: Conformal prediction & uncertainty quantification |

> **Note on training time:** Training all models across all four notebooks can take **several hours** in total. To keep runtimes manageable, experiments use a **subset of 10 classes** (out of the full 27) from the UTD-MHAD dataset.

> **Pre-run outputs:** All notebooks have been **pre-executed and their outputs saved**, so you can browse all results, plots, and metrics directly without running anything. Only re-run cells if you want to reproduce or modify the experiments.

## 5. Run the Notebooks

Notebooks are executed **headlessly** using `nbconvert` — this runs all cells in the terminal, saves the outputs back into the notebook file, and prints a message when done. No browser or Jupyter server is needed.

```bash
CUDA_VISIBLE_DEVICES=0 uv run jupyter nbconvert \
  --to notebook --execute \
  notebooks/<notebook_name>.ipynb \
  --output <notebook_name>.ipynb \
  --ExecutePreprocessor.timeout=7200
```

> **`--ExecutePreprocessor.timeout`**: This sets the maximum number of seconds allowed for any single cell to finish. The default is only 30 seconds, which is far too short for training cells. If the notebook finishes before the timeout is reached, `nbconvert` exits immediately — the timeout is a ceiling, not a wait. If a cell exceeds the timeout, execution is aborted with a `TimeoutError`. Set it high enough to cover your longest-running cell — `7200` (2 hours) is a safe default for training notebooks, and `600` (10 minutes) is enough for EDA.

Once finished, open the `.ipynb` file in any notebook viewer (VS Code, JupyterLab, etc.) to inspect all outputs, plots, and metrics.

### Part 1 — Exploratory Data Analysis

No training involved — safe to run all cells directly.

```bash
uv run jupyter nbconvert \
  --to notebook --execute \
  notebooks/part1_eda.ipynb \
  --output part1_eda.ipynb \
  --ExecutePreprocessor.timeout=600
```

### Parts 2, 3 & 4 — Models, Fusion & Missing Modality

**Recommended:** Run the training script first to pre-train the models, then execute the notebook:

```bash
# Pre-train unimodal and fusion models
CUDA_VISIBLE_DEVICES=0 uv run python scripts/train_models.py
```

```bash
CUDA_VISIBLE_DEVICES=0 uv run jupyter nbconvert \
  --to notebook --execute \
  notebooks/parts2_3_4.ipynb \
  --output parts2_3_4.ipynb \
  --ExecutePreprocessor.timeout=7200
```

### Parts 5, 6 & 7 — Class Imbalance & VAE Augmentation

**Recommended:** Run the imbalance and VAE training scripts first:

```bash
# Train models under class imbalance conditions
CUDA_VISIBLE_DEVICES=0 uv run python scripts/train_imbalance.py

# Train the VAE for data augmentation
CUDA_VISIBLE_DEVICES=0 uv run python scripts/train_vae.py
```

```bash
CUDA_VISIBLE_DEVICES=0 uv run jupyter nbconvert \
  --to notebook --execute \
  notebooks/parts_5_6_7.ipynb \
  --output parts_5_6_7.ipynb \
  --ExecutePreprocessor.timeout=7200
```

### Parts 8 & 9 — Conformal Prediction & Uncertainty

Depends on models trained in parts 2–4. Run `train_models.py` first (see above), then:

```bash
CUDA_VISIBLE_DEVICES=0 uv run jupyter nbconvert \
  --to notebook --execute \
  notebooks/parts_8_9.ipynb \
  --output parts_8_9.ipynb \
  --ExecutePreprocessor.timeout=3600
```

## 6. Project Structure

```
multimodal_har/
├── pyproject.toml          # Dependencies (uv-compatible)
├── uv.lock                 # Reproducible lock file
├── HOW_TO_RUN.md           # This file
├── data/                   # The raw data, images and outputs from the notebooks
├── notebooks/
│   ├── part1_eda.ipynb       # Part 1: EDA & data issues
│   ├── parts2_3_4.ipynb      # Parts 2-4: unimodal, fusion, missing modality
│   ├── parts_5_6_7.ipynb     # Parts 5-7: class imbalance & VAE augmentation
│   └── parts_8_9.ipynb       # Parts 8-9: conformal prediction & uncertainty
├── scripts/
│   ├── train_models.py       # Train unimodal & fusion models (parts 2-4)
│   ├── train_imbalance.py    # Train under class imbalance (parts 5-7)
│   └── train_vae.py          # Train VAE for augmentation (parts 5-7)
└── src/
    ├── __init__.py
    ├── data/
    │   ├── __init__.py
    │   ├── dataset.py          # UTDMHADDataset (PyTorch Dataset + DataLoaders)
    │   └── preprocessing.py    # Video frame sampling, inertial loading
    ├── models/
    │   ├── __init__.py
    │   ├── inertial_model.py   # InertialCNN (1D-CNN for sensor data)
    │   ├── video_model.py      # VideoClassifier (R3D-18 pretrained)
    │   ├── fusion_model.py     # MultimodalFusion (late fusion)
    │   └── vae.py              # VAE for inertial data augmentation
    └── utils/
        ├── __init__.py
        ├── config.py           # All constants and hyperparameters
        ├── training.py         # Training loops, evaluation, grid search
        ├── imbalance_helpers.py # Imbalance strategies & metrics
        └── conformal.py        # Conformal prediction utilities
```
