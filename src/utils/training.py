"""Training loops, evaluation, grid search, and failure-case visualisation.

This module provides the building blocks that both the standalone training
scripts (``scripts/train_models.py``) and the notebook use:

* ``train_epoch`` — one pass over the training DataLoader.
* ``evaluate`` — full evaluation with loss, accuracy, per-class report.
* ``run_grid_search`` — exhaustive 2-D grid search returning the best model.
* ``plot_confusion`` / ``plot_training_curves`` — matplotlib helpers.
* ``show_failure_cases`` — find and display misclassified test samples.
"""

from __future__ import annotations

import copy
import itertools
from typing import Any, Dict, List, Tuple, Type

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
)
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

from src.utils.config import CLASS_NAMES


# ── Training / evaluation ──────────────────────────────────────────────────────

def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    model_type: str = "inertial",
) -> float:
    """Run one training epoch and return the average batch loss.

    ``model_type`` selects which inputs are forwarded to the model:
        * ``"inertial"`` → ``model(inertial)``
        * ``"video"``    → ``model(video)``
        * ``"fusion"``   → ``model(video, inertial)``
    """
    model.train()
    total_loss = 0.0
    n_batches = 0

    for video, inertial, labels in loader:
        # Move tensors to the training device (GPU or CPU)
        video = video.to(device)
        inertial = inertial.to(device)
        labels = labels.to(device)

        # Forward pass — pick the right input based on model_type
        if model_type == "inertial":
            logits = model(inertial)
        elif model_type == "video":
            logits = model(video)
        else:  # fusion
            logits = model(video, inertial)

        # Backward pass and parameter update
        loss = criterion(logits, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    model_type: str = "inertial",
    mask_video: bool = False,
    mask_inertial: bool = False,
) -> Dict[str, Any]:
    """Evaluate model on the full DataLoader.

    Returns a dict with keys: loss, accuracy, preds, labels, report, confusion.

    The ``mask_video`` / ``mask_inertial`` flags are only used for fusion
    models to simulate missing modalities at test time (Part 4A).
    """
    model.eval()
    all_preds, all_labels = [], []
    total_loss = 0.0
    n_batches = 0

    for video, inertial, labels in loader:
        video = video.to(device)
        inertial = inertial.to(device)
        labels = labels.to(device)

        if model_type == "inertial":
            logits = model(inertial)
        elif model_type == "video":
            logits = model(video)
        else:  # fusion — forward masking flags for missing-modality eval
            logits = model(
                video, inertial,
                mask_video=mask_video,
                mask_inertial=mask_inertial,
            )

        loss = criterion(logits, labels)
        total_loss += loss.item()
        n_batches += 1

        preds = logits.argmax(dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    acc = accuracy_score(all_labels, all_preds)

    return {
        "loss": total_loss / max(n_batches, 1),
        "accuracy": acc,
        "preds": all_preds,
        "labels": all_labels,
        "report": classification_report(
            all_labels, all_preds,
            target_names=CLASS_NAMES, zero_division=0,
        ),
        "confusion": confusion_matrix(all_labels, all_preds),
    }


# ── Visualisation helpers ──────────────────────────────────────────────────────

def plot_confusion(cm: np.ndarray, title: str = "Confusion Matrix"):
    """Plot a confusion matrix heatmap with adaptive sizing for many classes."""
    n = len(CLASS_NAMES)
    # Scale figure to the number of classes so labels remain readable
    size = max(7, n * 0.65)
    fig, ax = plt.subplots(figsize=(size, size - 0.5))
    fontsize = 10 if n <= 8 else 8 if n <= 15 else 6
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=ax,
        annot_kws={"size": fontsize},
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    ax.tick_params(axis="x", labelsize=fontsize, rotation=45)
    ax.tick_params(axis="y", labelsize=fontsize, rotation=0)
    plt.tight_layout()
    return fig


def plot_training_curves(
    train_losses: List[float],
    val_accs: List[float],
    title: str = "",
    acc_ylim: Tuple[float, float] = None,
):
    """Plot loss and accuracy curves side-by-side.

    Parameters
    ----------
    acc_ylim : tuple or None
        If provided, sets the y-axis limits for the accuracy subplot.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # Left: training loss over epochs
    ax1.plot(train_losses)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Train Loss")
    ax1.set_title(f"{title} — Loss")

    # Right: validation accuracy over epochs
    ax2.plot(val_accs)
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Val Accuracy")
    ax2.set_title(f"{title} — Accuracy")
    if acc_ylim is not None:
        ax2.set_ylim(acc_ylim)

    plt.tight_layout()
    return fig


# ── Grid search ────────────────────────────────────────────────────────────────

def run_grid_search(
    build_model_fn,
    train_loader: DataLoader,
    val_loader: DataLoader,
    param_grid: Dict[str, List],
    device: torch.device,
    model_type: str = "inertial",
    epochs: int = 15,
    verbose: bool = True,
) -> Tuple[Dict, nn.Module, List[Dict]]:
    """Exhaustive grid search over all combinations in ``param_grid``.

    Returns
    -------
    best_params : dict
        Hyper-parameter dict of the best configuration.
    best_model : nn.Module
        Model reloaded with the best checkpoint weights.
    all_results : list of dict
        Per-configuration results including training curves.

    ``build_model_fn(params)`` must return a fresh ``nn.Module``.
    """
    keys = list(param_grid.keys())
    combos = list(itertools.product(*[param_grid[k] for k in keys]))

    all_results: List[Dict] = []
    best_acc = -1.0
    best_model_state = None
    best_params: Dict = {}

    for combo in combos:
        params = dict(zip(keys, combo))
        if verbose:
            print(f"\n  Grid: {params}")

        # Build a fresh model for this configuration
        model = build_model_fn(params).to(device)
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=params.get("lr", 1e-3),
            weight_decay=1e-4,
        )
        criterion = nn.CrossEntropyLoss()

        # Train for the specified number of epochs, recording curves
        train_losses = []
        val_accs = []
        for epoch in range(epochs):
            loss = train_epoch(model, train_loader, optimizer, criterion, device, model_type)
            train_losses.append(loss)
            res = evaluate(model, val_loader, criterion, device, model_type)
            val_accs.append(res["accuracy"])
            if verbose and (epoch + 1) % 5 == 0:
                print(f"    Epoch {epoch+1}/{epochs}  loss={loss:.4f}  val_acc={res['accuracy']:.3f}")

        # Final evaluation on the validation set
        final = evaluate(model, val_loader, criterion, device, model_type)
        result = {**params, "accuracy": final["accuracy"], "loss": final["loss"],
                  "train_losses": train_losses, "val_accs": val_accs}
        all_results.append(result)

        # Track the best configuration by validation accuracy
        if final["accuracy"] > best_acc:
            best_acc = final["accuracy"]
            best_model_state = copy.deepcopy(model.state_dict())
            best_params = params

    # Reload best model weights into a fresh model instance
    best_model = build_model_fn(best_params).to(device)
    best_model.load_state_dict(best_model_state)

    if verbose:
        print(f"\n  Best: {best_params}  acc={best_acc:.3f}")
    return best_params, best_model, all_results


# ── Failure case visualisation ─────────────────────────────────────────────────

@torch.no_grad()
def show_failure_cases(
    model: nn.Module,
    dataset,
    device: torch.device,
    model_type: str = "inertial",
    n: int = 3,
):
    """Find and display the first ``n`` misclassified test samples.

    For video/fusion models, shows the middle frame of the clip.
    For inertial/fusion models, plots the 6-axis sensor signal.
    """
    model.eval()
    failures = []

    # Iterate sample-by-sample (not batched) to preserve sample identity
    for idx in range(len(dataset)):
        video, inertial, label = dataset[idx]
        v = video.unsqueeze(0).to(device)   # add batch dim
        i = inertial.unsqueeze(0).to(device)

        if model_type == "inertial":
            logits = model(i)
        elif model_type == "video":
            logits = model(v)
        else:
            logits = model(v, i)

        pred = logits.argmax(dim=1).item()
        if pred != label:
            failures.append((idx, label, pred, video, inertial))
        if len(failures) >= n:
            break

    if not failures:
        print("No misclassifications found!")
        return

    for idx, true_label, pred_label, video, inertial in failures:
        true_name = CLASS_NAMES[true_label]
        pred_name = CLASS_NAMES[pred_label]
        print(f"Sample {idx}: TRUE={true_name}  PRED={pred_name}")

        if model_type in ("video", "fusion"):
            # Display the middle frame of the video clip
            frame = video[:, video.shape[1] // 2]  # (C, H, W)
            frame = frame.permute(1, 2, 0).numpy()
            # Undo ImageNet normalisation for visualisation
            frame = frame * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
            frame = np.clip(frame, 0, 1)
            fig, ax = plt.subplots(figsize=(4, 4))
            ax.imshow(frame)
            ax.set_title(f"True: {true_name} | Pred: {pred_name}")
            ax.axis("off")
            plt.show()

        if model_type in ("inertial", "fusion"):
            # Plot accelerometer (channels 0-2) and gyroscope (channels 3-5)
            sig = inertial.numpy()  # (6, T)
            fig, axes = plt.subplots(1, 2, figsize=(10, 3))
            t = np.arange(sig.shape[1]) / 50.0  # time axis in seconds
            for ch in range(3):
                axes[0].plot(t, sig[ch], alpha=0.8)
            axes[0].set_title(f"Accel — True: {true_name} | Pred: {pred_name}")
            axes[0].set_xlabel("Time (s)")
            for ch in range(3, 6):
                axes[1].plot(t, sig[ch], alpha=0.8)
            axes[1].set_title("Gyro")
            axes[1].set_xlabel("Time (s)")
            plt.tight_layout()
            plt.show()
