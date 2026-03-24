# Multimodal HAR (UTD-MHAD)

Multimodal human activity recognition on **UTD-MHAD** using **RGB video** and **inertial** sensor streams. The codebase includes PyTorch datasets and models (unimodal CNNs, late fusion), training utilities, and Jupyter notebooks that walk through exploratory analysis, fusion under missing modalities, class-imbalance strategies, VAE-based augmentation, and conformal prediction.

Experiments in the notebooks use a **10-class subset** of the full 27 actions for practical runtimes; notebooks are **pre-executed** so plots and metrics can be inspected without re-running.

## Quick start

- **Python** 3.9+, **CUDA** GPU recommended (see [HOW_TO_RUN.md](HOW_TO_RUN.md) for details).
- **Install** dependencies with [uv](https://docs.astral.sh/uv/):

```bash
cd multimodal_har
uv sync
```

- **Data**: download UTD-MHAD and place **861** `.avi` files under `data/RGB/` and **861** `.mat` files under `data/Inertial/` (links and layout in [HOW_TO_RUN.md](HOW_TO_RUN.md)).

- **Training scripts** (optional; speeds up notebook re-runs):

```bash
CUDA_VISIBLE_DEVICES=0 uv run python scripts/train_models.py      # unimodal + fusion
CUDA_VISIBLE_DEVICES=0 uv run python scripts/train_imbalance.py    # imbalance experiments
CUDA_VISIBLE_DEVICES=0 uv run python scripts/train_vae.py        # VAE for augmentation
```

- **Notebooks** (`notebooks/`): EDA, models & fusion, imbalance & VAE, conformal prediction. To execute headlessly, see the `jupyter nbconvert` commands in [HOW_TO_RUN.md](HOW_TO_RUN.md).

## Repository layout

| Path | Role |
|------|------|
| `src/data/` | `UTDMHADDataset`, preprocessing |
| `src/models/` | Inertial CNN, R3D-18 video classifier, fusion, VAE |
| `src/utils/` | Config, training loops, imbalance helpers, conformal utilities |
| `scripts/` | Long-running training entry points |
| `notebooks/` | End-to-end experiments and figures |

## Documentation

Full prerequisites, dataset paths, notebook index, and execution commands: **[HOW_TO_RUN.md](HOW_TO_RUN.md)**.

## Dataset

**UTD-MHAD** — use the dataset according to its license and citation requirements from the [official source](https://sites.google.com/view/utd-mhad-dataset).
