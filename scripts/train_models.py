#!/usr/bin/env python3
"""Standalone training script that runs all model training and saves checkpoints.

This script performs:
    1. Inertial CNN grid search  → selects best on val → saves model + grid results
    2. Video R3D-18 grid search  → selects best on val → saves model + grid results
    3. Standard fusion training  → monitors val accuracy → saves model
    4. Robust fusion training (modality dropout) → monitors val accuracy → saves model

Data split (subject-based, no leakage):
    * Train: subjects 1–4  (50 %)
    * Val:   subjects 5–6  (25 %) — used for grid search selection & early monitoring
    * Test:  subjects 7–8  (25 %) — held-out, never used during training

All artefacts are saved under ``checkpoints/``.  The notebook can then
simply load the saved checkpoints instead of re-training.

Usage:
    cd multimodal_har/
    CUDA_VISIBLE_DEVICES=0 uv run python scripts/train_models.py
"""

import sys
import os
import json
from copy import deepcopy
from pathlib import Path

# Ensure the project root is on sys.path so ``import src`` works
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
import torch.nn as nn

from src.utils.config import (
    NUM_CLASSES, SEED, CHECKPOINTS_DIR,
    TRAIN_SUBJECTS, VAL_SUBJECTS, TEST_SUBJECTS, 
    EPOCHS,
)
from src.data.dataset import UTDMHADDataset, get_dataloaders
from src.models.inertial_model import InertialCNN
from src.models.video_model import VideoClassifier
from src.models.fusion_model import MultimodalFusion
from src.utils.training import train_epoch, evaluate, run_grid_search

# ── Reproducibility ──────────────────────────────────────────────────────────
torch.manual_seed(SEED)
np.random.seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
if device.type == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# ── Create checkpoint directory ──────────────────────────────────────────────
CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)

# ── Data loaders (train / val / test) ────────────────────────────────────────
print("\nLoading data...")
print(f"  Train subjects: {TRAIN_SUBJECTS}  |  Val subjects: {VAL_SUBJECTS}  |  Test subjects: {TEST_SUBJECTS}")

train_loader_both, val_loader_both, test_loader_both = get_dataloaders(modality="both", batch_size=8, num_workers=4)
train_loader_iner, val_loader_iner, test_loader_iner = get_dataloaders(modality="inertial", batch_size=16, num_workers=4)
train_loader_vid, val_loader_vid, test_loader_vid = get_dataloaders(modality="video", batch_size=8, num_workers=4)

print(f"  Inertial  — train: {len(train_loader_iner)} batches, val: {len(val_loader_iner)}, test: {len(test_loader_iner)}")
print(f"  Video     — train: {len(train_loader_vid)}, val: {len(val_loader_vid)}, test: {len(test_loader_vid)}")
print(f"  Both      — train: {len(train_loader_both)}, val: {len(val_loader_both)}, test: {len(test_loader_both)}")


def save_grid_results(all_results, path):
    """Serialise grid-search results (strip numpy arrays) to JSON."""
    serialisable = []
    for r in all_results:
        entry = {}
        for k, v in r.items():
            if isinstance(v, (list, float, int, str)):
                entry[k] = v
            elif isinstance(v, np.floating):
                entry[k] = float(v)
        serialisable.append(entry)
    with open(path, "w") as f:
        json.dump(serialisable, f, indent=2)


# ══════════════════════════════════════════════════════════════════════════════
# 1.  Inertial CNN Grid Search (select best on val)
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("1. INERTIAL CNN — Grid Search (select on val)")
print("=" * 60)


def build_inertial(params):
    return InertialCNN(n_classes=NUM_CLASSES, dropout=params["dropout"])


# Grid search uses val_loader for model selection
best_iner_params, best_iner_model, iner_results = run_grid_search(
    build_model_fn=build_inertial,
    train_loader=train_loader_iner,
    val_loader=val_loader_iner,          # ← val, NOT test
    param_grid={"lr": [1e-3, 1e-4], "dropout": [0.3, 0.5]},
    device=device,
    model_type="inertial",
    epochs=EPOCHS,
)

# Report held-out test accuracy (never seen during selection)
criterion = nn.CrossEntropyLoss()
iner_test = evaluate(best_iner_model, test_loader_iner, criterion, device, model_type="inertial")
print(f"  Held-out test accuracy: {iner_test['accuracy']:.3f}")

# Save checkpoint
torch.save({
    "model_state_dict": best_iner_model.state_dict(),
    "best_params": best_iner_params,
}, CHECKPOINTS_DIR / "inertial_best.pt")

save_grid_results(iner_results, CHECKPOINTS_DIR / "inertial_grid_results.json")
print(f"  Saved → {CHECKPOINTS_DIR / 'inertial_best.pt'}")

# ══════════════════════════════════════════════════════════════════════════════
# 2.  Video R3D-18 Grid Search (select best on val)
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("2. VIDEO R3D-18 — Grid Search (select on val)")
print("=" * 60)


def build_video(params):
    return VideoClassifier(n_classes=NUM_CLASSES, dropout=params["dropout"])


best_vid_params, best_vid_model, vid_results = run_grid_search(
    build_model_fn=build_video,
    train_loader=train_loader_vid,
    val_loader=val_loader_vid,           # ← val, NOT test
    param_grid={"lr": [1e-3, 1e-4], "dropout": [0.3, 0.5]},
    device=device,
    model_type="video",
    epochs=EPOCHS,
)

vid_test = evaluate(best_vid_model, test_loader_vid, criterion, device, model_type="video")
print(f"  Held-out test accuracy: {vid_test['accuracy']:.3f}")

torch.save({
    "model_state_dict": best_vid_model.state_dict(),
    "best_params": best_vid_params,
}, CHECKPOINTS_DIR / "video_best.pt")

save_grid_results(vid_results, CHECKPOINTS_DIR / "video_grid_results.json")
print(f"  Saved → {CHECKPOINTS_DIR / 'video_best.pt'}")

# ══════════════════════════════════════════════════════════════════════════════
# 3.  Standard Fusion (no modality dropout) — monitor val
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("3. STANDARD FUSION — Training (monitor val)")
print("=" * 60)

# Initialise fusion encoders from the best unimodal models
video_enc = deepcopy(best_vid_model)
inertial_enc = deepcopy(best_iner_model)

fusion_model = MultimodalFusion(
    n_classes=NUM_CLASSES,
    dropout=0.3,
    modality_dropout=0.0,
    video_encoder=video_enc,
    inertial_encoder=inertial_enc,
).to(device)

# Differential learning rates: classifier head gets higher LR,
# pre-trained encoders get much lower LR to avoid catastrophic forgetting
optimizer_fusion = torch.optim.Adam([
    {"params": fusion_model.classifier.parameters(), "lr": 1e-3},
    {"params": fusion_model.video_encoder.parameters(), "lr": 1e-5},
    {"params": fusion_model.inertial_encoder.parameters(), "lr": 1e-4},
], weight_decay=1e-4)

fusion_train_losses = []
fusion_val_accs = []

for epoch in range(EPOCHS):
    loss = train_epoch(fusion_model, train_loader_both, optimizer_fusion, criterion, device, model_type="fusion")
    res = evaluate(fusion_model, val_loader_both, criterion, device, model_type="fusion")  # ← val
    fusion_train_losses.append(loss)
    fusion_val_accs.append(res["accuracy"])
    if (epoch + 1) % 5 == 0:
        print(f"  Epoch {epoch+1}/{EPOCHS}  loss={loss:.4f}  val_acc={res['accuracy']:.3f}")

fusion_test = evaluate(fusion_model, test_loader_both, criterion, device, model_type="fusion")
print(f"  Held-out test accuracy: {fusion_test['accuracy']:.3f}")

torch.save({
    "model_state_dict": fusion_model.state_dict(),
    "train_losses": fusion_train_losses,
    "val_accs": fusion_val_accs,
}, CHECKPOINTS_DIR / "fusion_standard.pt")
print(f"  Saved → {CHECKPOINTS_DIR / 'fusion_standard.pt'}")

# ══════════════════════════════════════════════════════════════════════════════
# 4.  Robust Fusion (modality dropout = 0.3) — monitor val
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("4. ROBUST FUSION (modality dropout) — Training (monitor val)")
print("=" * 60)

video_enc_b = deepcopy(best_vid_model)
inertial_enc_b = deepcopy(best_iner_model)

fusion_robust = MultimodalFusion(
    n_classes=NUM_CLASSES,
    dropout=0.3,
    modality_dropout=0.3,  # 30 % chance of dropping one modality per batch
    video_encoder=video_enc_b,
    inertial_encoder=inertial_enc_b,
).to(device)

optimizer_robust = torch.optim.Adam([
    {"params": fusion_robust.classifier.parameters(), "lr": 1e-3},
    {"params": fusion_robust.video_encoder.parameters(), "lr": 1e-5},
    {"params": fusion_robust.inertial_encoder.parameters(), "lr": 1e-4},
], weight_decay=1e-4)

robust_train_losses = []
robust_val_accs = []

for epoch in range(EPOCHS):
    loss = train_epoch(fusion_robust, train_loader_both, optimizer_robust, criterion, device, model_type="fusion")
    res = evaluate(fusion_robust, val_loader_both, criterion, device, model_type="fusion")  # ← val
    robust_train_losses.append(loss)
    robust_val_accs.append(res["accuracy"])
    if (epoch + 1) % 5 == 0:
        print(f"  Epoch {epoch+1}/{EPOCHS}  loss={loss:.4f}  val_acc={res['accuracy']:.3f}")

robust_test = evaluate(fusion_robust, test_loader_both, criterion, device, model_type="fusion")
print(f"  Held-out test accuracy: {robust_test['accuracy']:.3f}")

torch.save({
    "model_state_dict": fusion_robust.state_dict(),
    "train_losses": robust_train_losses,
    "val_accs": robust_val_accs,
}, CHECKPOINTS_DIR / "fusion_robust.pt")
print(f"  Saved → {CHECKPOINTS_DIR / 'fusion_robust.pt'}")

print("\n" + "=" * 60)
print("ALL DONE — checkpoints saved in:", CHECKPOINTS_DIR)
print("=" * 60)
