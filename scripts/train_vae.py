#!/usr/bin/env python3
"""Train the FusionVAE for OOD detection (Part 7) and prepare conformal calibration data (Part 6).

Steps
-----
1.  Load the trained MultimodalFusion model (robust checkpoint).
2.  Extract 640-d fusion vectors for every sample in train, val, and test splits.
3.  Train the FusionVAE on train-split fusion vectors.
4.  Compute the OOD threshold as the 65th-percentile reconstruction error on
    the val-split fusion vectors.
5.  Save all artefacts to checkpoints/:
        vae.pt              — trained VAE weights + hparams
        vae_threshold.json  — OOD threshold value
        fusion_vectors.pt   — train/val/test fusion vectors + labels (for notebook use)

For Part 6 (Conformal Prediction) the notebook reads the val fusion vectors
directly; the VAE threshold file is consumed by Part 7.

Usage
-----
    cd multimodal_har/
    CUDA_VISIBLE_DEVICES=0 uv run python scripts/train_vae.py
"""

import sys
import json
from pathlib import Path

# Ensure project root is on sys.path so `import src` works
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.utils.config import (
    NUM_CLASSES, SEED, CHECKPOINTS_DIR,
    TRAIN_SUBJECTS, VAL_SUBJECTS, TEST_SUBJECTS,
    SUBSET_ACTIONS, ACTION_NAMES,
    RGB_DIR, INERTIAL_DIR,
)
from src.data.dataset import UTDMHADDataset, get_dataloaders
from src.data.preprocessing import parse_filename, sample_video_frames, load_inertial
from src.models.fusion_model import MultimodalFusion
from src.models.vae import FusionVAE

# ── Reproducibility ──────────────────────────────────────────────────────────
torch.manual_seed(SEED)
np.random.seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
if device.type == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")

CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)

# ── OOD action classes (actions present in the dataset but NOT in SUBSET_ACTIONS) ──
# These are used to build the OOD evaluation dataset.
ALL_ACTIONS = list(range(1, 28))
OOD_ACTIONS = [a for a in ALL_ACTIONS if a not in SUBSET_ACTIONS]
print(f"\nIn-distribution actions ({len(SUBSET_ACTIONS)}): {SUBSET_ACTIONS}")
print(f"OOD actions ({len(OOD_ACTIONS)}): {OOD_ACTIONS}")
print(f"OOD action names: {[ACTION_NAMES[a] for a in OOD_ACTIONS]}")

# ══════════════════════════════════════════════════════════════════════════════
# 1.  Load the trained MultimodalFusion model
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("1. Loading MultimodalFusion (robust checkpoint)")
print("=" * 60)

fusion_ckpt = CHECKPOINTS_DIR / "fusion_robust.pt"
if not fusion_ckpt.exists():
    raise FileNotFoundError(
        f"Fusion checkpoint not found at {fusion_ckpt}. "
        "Run scripts/train_models.py first."
    )

fusion_model = MultimodalFusion(n_classes=NUM_CLASSES, dropout=0.3, modality_dropout=0.3)
ckpt = torch.load(fusion_ckpt, map_location=device, weights_only=False)
fusion_model.load_state_dict(ckpt["model_state_dict"])
fusion_model = fusion_model.to(device)
fusion_model.eval()
print(f"  Loaded {fusion_ckpt}")

# ══════════════════════════════════════════════════════════════════════════════
# 2.  Extract fusion vectors for train / val / test (ID) splits
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("2. Extracting fusion vectors (ID splits)")
print("=" * 60)


def extract_vectors(subjects, desc=""):
    """Extract 640-d fusion vectors and labels for a given set of subjects."""
    ds = UTDMHADDataset(subjects=subjects, modality="both", augment=False)
    loader = DataLoader(ds, batch_size=8, shuffle=False, num_workers=4)
    all_vecs, all_labels = [], []
    with torch.no_grad():
        for video, inertial, labels in loader:
            video = video.to(device)
            inertial = inertial.to(device)
            # extract_fusion_vector returns (B, 640) without classifier
            fvec = fusion_model.extract_fusion_vector(video, inertial)
            all_vecs.append(fvec.cpu())
            all_labels.append(labels)
    vecs = torch.cat(all_vecs, dim=0)   # (N, 640)
    labs = torch.cat(all_labels, dim=0) # (N,)
    print(f"  {desc}: {vecs.shape[0]} samples, vector dim={vecs.shape[1]}")
    return vecs, labs


train_vecs, train_labels = extract_vectors(TRAIN_SUBJECTS, "Train (ID)")
val_vecs,   val_labels   = extract_vectors(VAL_SUBJECTS,   "Val   (ID)")
test_vecs,  test_labels  = extract_vectors(TEST_SUBJECTS,  "Test  (ID)")

# ── Extract fusion vectors for OOD samples (same test subjects, unseen actions) ──
# UTDMHADDataset uses ACTION_TO_IDX which only knows ID classes, so we build
# the OOD file index manually using the preprocessing helpers.
print("\n  Extracting OOD fusion vectors (unseen action classes, test subjects)...")

# Build a lookup of (action, subject) → (rgb_path, inertial_path) for OOD actions
ood_file_pairs = []  # list of (rgb_path, inertial_path) tuples

rgb_index = {}
for f in sorted(RGB_DIR.glob("*.avi")):
    key = parse_filename(f.name)
    if key:
        rgb_index[key] = f

iner_index = {}
for f in sorted(INERTIAL_DIR.glob("*.mat")):
    key = parse_filename(f.name)
    if key:
        iner_index[key] = f

# Collect files where action is OOD and subject is in TEST_SUBJECTS
# Use all subjects as fallback if test subjects have no OOD recordings
for key in sorted(set(rgb_index) | set(iner_index)):
    action, subject, trial = key
    if action not in OOD_ACTIONS:
        continue
    if subject not in TEST_SUBJECTS:
        continue
    rgb_path  = rgb_index.get(key)
    iner_path = iner_index.get(key)
    if rgb_path and iner_path:
        ood_file_pairs.append((rgb_path, iner_path))

# Fallback: use all subjects if the test-subject restriction yields nothing
if len(ood_file_pairs) == 0:
    print("  WARNING: no OOD samples found for TEST_SUBJECTS, using all subjects.")
    for key in sorted(set(rgb_index) | set(iner_index)):
        action, subject, trial = key
        if action not in OOD_ACTIONS:
            continue
        rgb_path  = rgb_index.get(key)
        iner_path = iner_index.get(key)
        if rgb_path and iner_path:
            ood_file_pairs.append((rgb_path, iner_path))

print(f"  Found {len(ood_file_pairs)} OOD file pairs")

# Extract fusion vectors batch by batch
ood_vecs_list = []
BATCH = 8
for i in range(0, len(ood_file_pairs), BATCH):
    batch_pairs = ood_file_pairs[i:i + BATCH]
    videos   = torch.stack([sample_video_frames(rgb)  for rgb,  _ in batch_pairs])
    inertials = torch.stack([load_inertial(iner)       for _,  iner in batch_pairs])
    with torch.no_grad():
        fvec = fusion_model.extract_fusion_vector(
            videos.to(device), inertials.to(device)
        )
    ood_vecs_list.append(fvec.cpu())

ood_vecs = torch.cat(ood_vecs_list, dim=0) if ood_vecs_list else torch.zeros(0, 640)
print(f"  OOD samples: {ood_vecs.shape[0]}")

# Save all vectors so the notebook can load them without re-running the fusion model
torch.save({
    "train_vecs": train_vecs,
    "train_labels": train_labels,
    "val_vecs": val_vecs,
    "val_labels": val_labels,
    "test_vecs": test_vecs,
    "test_labels": test_labels,
    "ood_vecs": ood_vecs,
    "ood_actions": OOD_ACTIONS,
}, CHECKPOINTS_DIR / "fusion_vectors.pt")
print(f"  Saved → {CHECKPOINTS_DIR / 'fusion_vectors.pt'}")

# ══════════════════════════════════════════════════════════════════════════════
# 3.  Train the FusionVAE on train-split fusion vectors
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("3. Training FusionVAE on train-split fusion vectors")
print("=" * 60)

# Hyperparameters
VAE_LATENT_DIM = 64
VAE_HIDDEN_DIM = 256
VAE_BETA = 1.0
VAE_LR = 1e-3
VAE_EPOCHS = 60
VAE_BATCH_SIZE = 32

vae = FusionVAE(
    input_dim=train_vecs.shape[1],
    latent_dim=VAE_LATENT_DIM,
    hidden_dim=VAE_HIDDEN_DIM,
    beta=VAE_BETA,
).to(device)

optimizer = torch.optim.Adam(vae.parameters(), lr=VAE_LR, weight_decay=1e-5)
# Cosine annealing to smoothly reduce LR over training
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=VAE_EPOCHS)

# Wrap train vectors in a simple TensorDataset
train_vae_ds = TensorDataset(train_vecs)
train_vae_loader = DataLoader(
    train_vae_ds, batch_size=VAE_BATCH_SIZE, shuffle=True, drop_last=False,
)

vae_train_losses = []
for epoch in range(VAE_EPOCHS):
    vae.train()
    epoch_loss = 0.0
    n_batches = 0
    for (x_batch,) in train_vae_loader:
        x_batch = x_batch.to(device)
        x_hat, mu, log_var = vae(x_batch)
        loss, recon, kl = vae.loss(x_batch, x_hat, mu, log_var)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        n_batches += 1
    scheduler.step()
    avg_loss = epoch_loss / max(n_batches, 1)
    vae_train_losses.append(avg_loss)
    if (epoch + 1) % 10 == 0:
        print(f"  Epoch {epoch+1}/{VAE_EPOCHS}  loss={avg_loss:.4f}  lr={scheduler.get_last_lr()[0]:.2e}")

# Save VAE checkpoint
torch.save({
    "model_state_dict": vae.state_dict(),
    "hparams": {
        "input_dim": train_vecs.shape[1],
        "latent_dim": VAE_LATENT_DIM,
        "hidden_dim": VAE_HIDDEN_DIM,
        "beta": VAE_BETA,
    },
    "train_losses": vae_train_losses,
}, CHECKPOINTS_DIR / "vae.pt")
print(f"  Saved → {CHECKPOINTS_DIR / 'vae.pt'}")

# ══════════════════════════════════════════════════════════════════════════════
# 4.  Compute OOD threshold from val-split reconstruction errors
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("4. Computing OOD threshold (65th pct on val errors)")
print("=" * 60)

vae.eval()
val_errors = vae.reconstruction_error(val_vecs.to(device)).cpu().numpy()

# Use the 65th percentile of ID val errors as the detection boundary.
# Samples with reconstruction error above this threshold are flagged as OOD.
# 80th pct gives a 20% false-positive rate on ID data, which is a reasonable
# trade-off for higher true-positive detection of OOD samples.
ood_threshold = float(np.percentile(val_errors, 65))
print(f"  Val error — mean={val_errors.mean():.4f}  std={val_errors.std():.4f}")
print(f"  OOD threshold (65th pct): {ood_threshold:.4f}")

# Quick sanity check: how well does this threshold separate ID test vs OOD?
test_errors = vae.reconstruction_error(test_vecs.to(device)).cpu().numpy()
ood_errors  = vae.reconstruction_error(ood_vecs.to(device)).cpu().numpy() if len(ood_vecs) > 0 else np.array([])

id_flagged_pct  = float((test_errors > ood_threshold).mean() * 100)
ood_flagged_pct = float((ood_errors  > ood_threshold).mean() * 100) if len(ood_errors) > 0 else 0.0
print(f"  ID test samples flagged as OOD: {id_flagged_pct:.1f}%  (ideally ≤ 5%)")
print(f"  True OOD samples flagged:       {ood_flagged_pct:.1f}%  (ideally high)")

# Save threshold and evaluation metadata
with open(CHECKPOINTS_DIR / "vae_threshold.json", "w") as f:
    json.dump({
        "ood_threshold": ood_threshold,
        "val_error_mean": float(val_errors.mean()),
        "val_error_std": float(val_errors.std()),
        "id_false_positive_pct": id_flagged_pct,
        "ood_detection_pct": ood_flagged_pct,
        "ood_actions": OOD_ACTIONS,
        "ood_action_names": [ACTION_NAMES[a] for a in OOD_ACTIONS],
    }, f, indent=2)
print(f"  Saved → {CHECKPOINTS_DIR / 'vae_threshold.json'}")

print("\n" + "=" * 60)
print("ALL DONE — VAE artefacts saved in:", CHECKPOINTS_DIR)
print("=" * 60)
