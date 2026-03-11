#!/usr/bin/env python3
"""Part 5 training script — Class Imbalance Experiments.

Trains three MultimodalFusion models from scratch on different versions of
the imbalanced training set and saves their checkpoints to ``checkpoints/``.

The notebook (``notebooks/parts_5_6_7.ipynb``) will try to load these
checkpoints first; only if loading fails does it fall back to inline training.

Experiments
-----------
1. **Naive** (``imbalance_naive.pt``)
   - Walk sabotaged to 10% of original samples
   - No imbalance compensation — plain CrossEntropyLoss

2. **Technique A** (``imbalance_wrs.pt``)
   - Same sabotaged dataset
   - ``WeightedRandomSampler`` to equalize sampling frequency

3. **Technique B** (``imbalance_wce.pt``)
   - Same sabotaged dataset (plain DataLoader, no sampler)
   - Class-balanced ``CrossEntropyLoss(weight=...)``

A metadata JSON (``imbalance_meta.json``) is saved alongside, recording
training curves and final metrics for all three experiments.

Usage
-----
    cd multimodal_har/
    CUDA_VISIBLE_DEVICES=0 uv run python scripts/train_imbalance.py
"""

import sys
import json
from pathlib import Path

# Ensure the project root (multimodal_har/) is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.utils.config import (
    NUM_CLASSES, CLASS_NAMES, SEED, CHECKPOINTS_DIR,
    TRAIN_SUBJECTS, VAL_SUBJECTS, TEST_SUBJECTS,
)
from src.data.dataset import UTDMHADDataset, get_dataloaders
from src.utils.training import evaluate
from src.utils.imbalance_helpers import (
    subsample_class, get_class_counts,
    get_loss_weights, make_weighted_loader,
    train_fusion_from_scratch,
    extract_macro_f1, get_minority_recall,
)

# ── Reproducibility ───────────────────────────────────────────────────────────
torch.manual_seed(SEED)
np.random.seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
if device.type == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")

CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)

# ── The minority class to sabotage ────────────────────────────────────────────
MINORITY_CLASS_IDX  = CLASS_NAMES.index("Boxing")  # 0-based index
MINORITY_CLASS_NAME = CLASS_NAMES[MINORITY_CLASS_IDX]
KEEP_FRACTION = 0.15   # retain predetermined fraction of minority class training samples
EPOCHS = 30

print(f"\nMinority class: \"{MINORITY_CLASS_NAME}\" (index {MINORITY_CLASS_IDX})")
print(f"Keep fraction: {KEEP_FRACTION:.0%} → ~{max(1, int(16 * KEEP_FRACTION))} minority class training samples")

# ── Load validation and test loaders (always balanced) ────────────────────────
# Only the training set is imbalanced; val/test remain as-is for fair evaluation
print("\nLoading val/test loaders...")
_, val_loader, test_loader = get_dataloaders(modality="both", batch_size=8, num_workers=4)
criterion_eval = nn.CrossEntropyLoss()

# ── Build the imbalanced training dataset ─────────────────────────────────────
print("Building imbalanced training dataset...")
train_ds_full = UTDMHADDataset(subjects=TRAIN_SUBJECTS, modality="both", augment=True)
train_ds_imbal = subsample_class(
    train_ds_full,
    target_class_idx=MINORITY_CLASS_IDX,
    keep_fraction=KEEP_FRACTION,
    seed=SEED,
)

counts_imbal = get_class_counts(train_ds_imbal)
n_walk_orig   = get_class_counts(train_ds_full).get(MINORITY_CLASS_IDX, 0)
n_walk_after  = counts_imbal.get(MINORITY_CLASS_IDX, 0)
print(f"Walk samples: {n_walk_orig} → {n_walk_after}")
print(f"Total train: {len(train_ds_full)} → {len(train_ds_imbal)}")

# ── Shared plain DataLoader for naive + Technique B ───────────────────────────
# (Technique A uses make_weighted_loader instead)
g = torch.Generator()
g.manual_seed(SEED)
train_loader_naive = DataLoader(
    train_ds_imbal, batch_size=8, shuffle=True,
    num_workers=4, pin_memory=True, generator=g,
)

meta = {}  # will hold training curves and final metrics for all experiments


# ══════════════════════════════════════════════════════════════════════════════
# Experiment 1 — Naive retraining (no compensation)
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("EXPERIMENT 1: Naive retraining — no imbalance compensation")
print("=" * 60)

model_naive, losses_naive, accs_naive = train_fusion_from_scratch(
    train_loader=train_loader_naive,
    val_loader=val_loader,
    device=device,
    criterion=nn.CrossEntropyLoss(),
    epochs=EPOCHS,
    num_classes=NUM_CLASSES,
    verbose=True,
)

# Evaluate on the held-out balanced test set
eval_naive = evaluate(model_naive, test_loader, criterion_eval, device, model_type="fusion")
print(f"\nNaive — test accuracy:  {eval_naive['accuracy']:.3f}")
print(f"Naive — macro-F1:       {extract_macro_f1(eval_naive['report']):.3f}")
print(f"Naive — {MINORITY_CLASS_NAME} recall: {get_minority_recall(eval_naive['report'], MINORITY_CLASS_NAME):.3f}")

# Save checkpoint (model weights + training curves)
torch.save({
    "model_state_dict": model_naive.state_dict(),
    "train_losses":     losses_naive,
    "val_accs":         accs_naive,
}, CHECKPOINTS_DIR / "imbalance_naive.pt")
print(f"Saved → {CHECKPOINTS_DIR / 'imbalance_naive.pt'}")

meta["naive"] = {
    "train_losses": losses_naive,
    "val_accs":     accs_naive,
    "test_accuracy": eval_naive["accuracy"],
    "macro_f1":      extract_macro_f1(eval_naive["report"]),
    f"{MINORITY_CLASS_NAME}_recall": get_minority_recall(eval_naive["report"], MINORITY_CLASS_NAME),
    "report": eval_naive["report"],
    "confusion": eval_naive["confusion"].tolist(),
}


# ══════════════════════════════════════════════════════════════════════════════
# Experiment 2 — Technique A: WeightedRandomSampler
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("EXPERIMENT 2: Technique A — WeightedRandomSampler")
print("=" * 60)

# make_weighted_loader wraps the same imbalanced dataset but samples each
# class at equal frequency; augment=True on the base dataset ensures minority
# samples receive different augmentations on each re-sample.
train_loader_wrs = make_weighted_loader(
    train_ds_imbal, batch_size=8, num_workers=4, seed=SEED,
)

model_wrs, losses_wrs, accs_wrs = train_fusion_from_scratch(
    train_loader=train_loader_wrs,
    val_loader=val_loader,
    device=device,
    criterion=nn.CrossEntropyLoss(),
    epochs=EPOCHS,
    num_classes=NUM_CLASSES,
    verbose=True,
)

eval_wrs = evaluate(model_wrs, test_loader, criterion_eval, device, model_type="fusion")
print(f"\nTechnique A — test accuracy:  {eval_wrs['accuracy']:.3f}")
print(f"Technique A — macro-F1:       {extract_macro_f1(eval_wrs['report']):.3f}")
print(f"Technique A — {MINORITY_CLASS_NAME} recall: {get_minority_recall(eval_wrs['report'], MINORITY_CLASS_NAME):.3f}")

torch.save({
    "model_state_dict": model_wrs.state_dict(),
    "train_losses":     losses_wrs,
    "val_accs":         accs_wrs,
}, CHECKPOINTS_DIR / "imbalance_wrs.pt")
print(f"Saved → {CHECKPOINTS_DIR / 'imbalance_wrs.pt'}")

meta["wrs"] = {
    "train_losses": losses_wrs,
    "val_accs":     accs_wrs,
    "test_accuracy": eval_wrs["accuracy"],
    "macro_f1":      extract_macro_f1(eval_wrs["report"]),
    f"{MINORITY_CLASS_NAME}_recall": get_minority_recall(eval_wrs["report"], MINORITY_CLASS_NAME),
    "report": eval_wrs["report"],
    "confusion": eval_wrs["confusion"].tolist(),
}


# ══════════════════════════════════════════════════════════════════════════════
# Experiment 3 — Technique B: Weighted CrossEntropyLoss
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("EXPERIMENT 3: Technique B — Weighted CrossEntropyLoss")
print("=" * 60)

# Compute inverse-frequency weights and embed them in the loss criterion
class_weights = get_loss_weights(counts_imbal, num_classes=NUM_CLASSES, device=device)
print("Class weights (higher = rarer class):")
for name, w in zip(CLASS_NAMES, class_weights.cpu().numpy()):
    marker = "  ← minority" if name == MINORITY_CLASS_NAME else ""
    print(f"  {name:<22} {w:.4f}{marker}")

criterion_wce = nn.CrossEntropyLoss(weight=class_weights)

# Same plain imbalanced DataLoader as in the naive experiment — the only
# change is the loss function, not the sampling strategy.
model_wce, losses_wce, accs_wce = train_fusion_from_scratch(
    train_loader=train_loader_naive,   # same DataLoader, no WeightedRandomSampler
    val_loader=val_loader,
    device=device,
    criterion=criterion_wce,
    epochs=EPOCHS,
    num_classes=NUM_CLASSES,
    verbose=True,
)

eval_wce = evaluate(model_wce, test_loader, criterion_eval, device, model_type="fusion")
print(f"\nTechnique B — test accuracy:  {eval_wce['accuracy']:.3f}")
print(f"Technique B — macro-F1:       {extract_macro_f1(eval_wce['report']):.3f}")
print(f"Technique B — {MINORITY_CLASS_NAME} recall: {get_minority_recall(eval_wce['report'], MINORITY_CLASS_NAME):.3f}")

torch.save({
    "model_state_dict": model_wce.state_dict(),
    "train_losses":     losses_wce,
    "val_accs":         accs_wce,
}, CHECKPOINTS_DIR / "imbalance_wce.pt")
print(f"Saved → {CHECKPOINTS_DIR / 'imbalance_wce.pt'}")

meta["wce"] = {
    "train_losses": losses_wce,
    "val_accs":     accs_wce,
    "test_accuracy": eval_wce["accuracy"],
    "macro_f1":      extract_macro_f1(eval_wce["report"]),
    f"{MINORITY_CLASS_NAME}_recall": get_minority_recall(eval_wce["report"], MINORITY_CLASS_NAME),
    "report": eval_wce["report"],
    "confusion": eval_wce["confusion"].tolist(),
}


# ── Save combined metadata JSON ───────────────────────────────────────────────
meta_path = CHECKPOINTS_DIR / "imbalance_meta.json"
with open(meta_path, "w") as f:
    json.dump(meta, f, indent=2)
print(f"\nMetadata saved → {meta_path}")


# ── Final comparison table ────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("FINAL COMPARISON")
print("=" * 60)
header = f"{'Experiment':<30} {'Accuracy':>8} {'Macro-F1':>8} {MINORITY_CLASS_NAME+' Recall':>12}"
print(header)
print("-" * len(header))
for exp_name, ev in [("Naive (no compensation)", eval_naive),
                     ("Technique A (WRS)",       eval_wrs),
                     ("Technique B (WCE)",       eval_wce)]:
    acc = ev["accuracy"]
    f1  = extract_macro_f1(ev["report"])
    rec = get_minority_recall(ev["report"], MINORITY_CLASS_NAME)
    print(f"{exp_name:<30} {acc:>8.3f} {f1:>8.3f} {rec:>12.3f}")

print("\nALL DONE — checkpoints saved in:", CHECKPOINTS_DIR)
