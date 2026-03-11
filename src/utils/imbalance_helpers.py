"""Helper functions for Part 5: Class Imbalance Analysis & Mitigation.

Provides utilities to:
- Artificially imbalance a dataset (``subsample_class``)
- Compute class / sample weights for the two mitigation techniques
- Build a DataLoader with ``WeightedRandomSampler`` (Technique A)
- Train a fresh ``MultimodalFusion`` from scratch with a custom criterion (Technique B)
- Visualize normalized confusion matrices
- Extract macro-F1 from sklearn's classification report
"""

from __future__ import annotations

import re
import copy
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler

from src.utils.config import NUM_CLASSES, SEED, WEIGHT_DECAY
from src.models.fusion_model import MultimodalFusion
from src.utils.training import train_epoch, evaluate


# ── Dataset imbalance utilities ───────────────────────────────────────────────

def get_labels(dataset) -> List[int]:
    """Extract the integer label list from a Dataset or Subset.

    Works with ``UTDMHADDataset`` (stores labels in ``samples`` tuples) and
    with any ``torch.utils.data.Subset`` that wraps such a dataset.
    """
    base = dataset.dataset if isinstance(dataset, Subset) else dataset
    indices = dataset.indices if isinstance(dataset, Subset) else range(len(base))
    # Each entry in base.samples is (rgb_path, iner_path, label, key)
    return [base.samples[i][2] for i in indices]


def get_class_counts(dataset) -> Dict[int, int]:
    """Return a {class_index: sample_count} mapping for a dataset or Subset."""
    labels = get_labels(dataset)
    counts: Dict[int, int] = {}
    for lbl in labels:
        counts[lbl] = counts.get(lbl, 0) + 1
    return counts


def subsample_class(
    dataset,
    target_class_idx: int,
    keep_fraction: float = 0.10,
    seed: int = SEED,
) -> Subset:
    """Return a Subset that retains only ``keep_fraction`` of ``target_class_idx`` samples.

    All other classes keep all their samples.  This simulates a real-world
    scenario where one class is severely under-represented in the training set.

    Parameters
    ----------
    dataset : UTDMHADDataset (or any dataset whose items expose a label)
    target_class_idx : int
        0-based class index of the minority class to thin out.
    keep_fraction : float
        Fraction of minority-class samples to retain (default 0.10 = 10%).
    seed : int
        Random seed for reproducible sub-sampling.

    Returns
    -------
    Subset wrapping the original dataset with the thinned index list.
    """
    rng = np.random.default_rng(seed)
    base = dataset.dataset if isinstance(dataset, Subset) else dataset
    all_indices = list(dataset.indices) if isinstance(dataset, Subset) else list(range(len(base)))

    # Split indices by class
    minority_idx = [i for i in all_indices if base.samples[i][2] == target_class_idx]
    majority_idx = [i for i in all_indices if base.samples[i][2] != target_class_idx]

    # Randomly retain keep_fraction of minority samples
    n_keep = max(1, int(len(minority_idx) * keep_fraction))
    kept_minority = rng.choice(minority_idx, size=n_keep, replace=False).tolist()

    combined = sorted(majority_idx + kept_minority)
    return Subset(base, combined)


# ── Weight computation ────────────────────────────────────────────────────────

def get_loss_weights(
    class_counts: Dict[int, int],
    num_classes: int = NUM_CLASSES,
    device: torch.device = torch.device("cpu"),
) -> torch.Tensor:
    """Compute inverse-frequency class weights for ``nn.CrossEntropyLoss``.

    Weight formula: ``w_c = total_samples / (num_classes × count_c)``

    This normalises so that the average weight is 1.0, preserving the scale
    of the loss relative to the unweighted case.

    Parameters
    ----------
    class_counts : dict mapping class index → count
    num_classes : int
    device : torch.device

    Returns
    -------
    weights : torch.Tensor of shape (num_classes,)
    """
    total = sum(class_counts.values())
    weights = []
    for c in range(num_classes):
        count = class_counts.get(c, 1)  # avoid division by zero
        weights.append(total / (num_classes * count))
    return torch.tensor(weights, dtype=torch.float32, device=device)


def get_sample_weights(dataset) -> List[float]:
    """Compute per-sample weights for ``WeightedRandomSampler``.

    Each sample's weight is the inverse of its class frequency, so all
    classes are sampled at the same expected rate during training.

    Parameters
    ----------
    dataset : Dataset or Subset

    Returns
    -------
    weights : list of float, one per sample in ``dataset``
    """
    counts = get_class_counts(dataset)
    labels = get_labels(dataset)
    # weight for each sample = 1 / count of its class
    return [1.0 / counts[lbl] for lbl in labels]


def make_weighted_loader(
    dataset,
    batch_size: int = 8,
    num_workers: int = 4,
    seed: int = SEED,
) -> DataLoader:
    """Build a DataLoader that uses ``WeightedRandomSampler`` to balance classes.

    The sampler draws ``len(dataset)`` samples (with replacement) per epoch,
    assigning higher probability to under-represented classes.  Combined with
    ``augment=True`` on the underlying dataset, minority samples are seen
    repeatedly but with different random augmentations each time, preventing
    overfitting to the few minority examples.

    Parameters
    ----------
    dataset : Dataset or Subset with ``augment=True`` on the base UTDMHADDataset
    batch_size : int
    num_workers : int
    seed : int

    Returns
    -------
    DataLoader with WeightedRandomSampler
    """
    weights = get_sample_weights(dataset)
    g = torch.Generator()
    g.manual_seed(seed)

    # num_samples = len(dataset) so the DataLoader epoch length is unchanged
    sampler = WeightedRandomSampler(
        weights=weights,
        num_samples=len(dataset),
        replacement=True,
        generator=g,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,          # mutually exclusive with shuffle=True
        num_workers=num_workers,
        pin_memory=True,
    )


# ── Full-model training from scratch ─────────────────────────────────────────

def train_fusion_from_scratch(
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    criterion: nn.Module = None,
    epochs: int = 30,
    num_classes: int = NUM_CLASSES,
    verbose: bool = True,
) -> Tuple[MultimodalFusion, List[float], List[float]]:
    """Build and train a fresh MultimodalFusion model from random initialisation.

    All parameters (video encoder, inertial encoder, classifier head) are
    updated.  Differential learning rates are used to balance the larger
    pretrained backbone with the smaller new classifier:
        - Classifier head : lr = 1e-3
        - Inertial encoder : lr = 1e-4  (pretrained weights, fine-tune gently)
        - Video encoder    : lr = 1e-5  (large R3D-18, avoid catastrophic forgetting)

    Parameters
    ----------
    train_loader : DataLoader over the (possibly imbalanced) training split
    val_loader   : DataLoader over the validation split
    device       : torch.device
    criterion    : loss function (default: plain CrossEntropyLoss).
                   Pass a weighted CrossEntropyLoss for Technique B.
    epochs       : number of training epochs
    num_classes  : number of output classes
    verbose      : if True, print loss/acc every 5 epochs

    Returns
    -------
    model        : trained MultimodalFusion
    train_losses : list of per-epoch average training loss
    val_accs     : list of per-epoch validation accuracy
    """
    if criterion is None:
        criterion = nn.CrossEntropyLoss()

    # Build a completely fresh model (no pretrained weights loaded)
    model = MultimodalFusion(n_classes=num_classes, dropout=0.3, modality_dropout=0.0).to(device)

    # Differential learning rates prevent the large R3D-18 backbone from
    # dominating the gradient updates and destroying pretrained representations.
    optimizer = torch.optim.Adam([
        {"params": model.classifier.parameters(),      "lr": 1e-3},
        {"params": model.inertial_encoder.parameters(), "lr": 1e-4},
        {"params": model.video_encoder.parameters(),   "lr": 1e-5},
    ], weight_decay=WEIGHT_DECAY)

    train_losses: List[float] = []
    val_accs: List[float] = []

    for epoch in range(epochs):
        loss = train_epoch(model, train_loader, optimizer, criterion, device, model_type="fusion")
        res  = evaluate(model, val_loader, criterion, device, model_type="fusion")
        train_losses.append(loss)
        val_accs.append(res["accuracy"])
        if verbose and (epoch + 1) % 5 == 0:
            print(f"  Epoch {epoch+1}/{epochs}  loss={loss:.4f}  val_acc={res['accuracy']:.3f}")

    return model, train_losses, val_accs


# ── Metrics extraction ────────────────────────────────────────────────────────

def extract_macro_f1(report_str: str) -> float:
    """Parse a sklearn classification_report string and return macro-avg F1.

    Parameters
    ----------
    report_str : str — output of ``sklearn.metrics.classification_report``

    Returns
    -------
    macro_f1 : float in [0, 1]
    """
    # The macro-avg line looks like:
    #   "   macro avg       0.xx      0.xx      0.xx      NNN"
    match = re.search(r"macro avg\s+[\d.]+\s+[\d.]+\s+([\d.]+)", report_str)
    if match:
        return float(match.group(1))
    return float("nan")


def get_minority_recall(report_str: str, class_name: str) -> float:
    """Extract the recall for a specific class from a classification report.

    Parameters
    ----------
    report_str : str — sklearn classification_report output
    class_name : str — exact class name as it appears in the report

    Returns
    -------
    recall : float in [0, 1], or NaN if not found
    """
    # Each class line: "  ClassName   precision  recall  f1  support"
    escaped = re.escape(class_name)
    match = re.search(rf"{escaped}\s+[\d.]+\s+([\d.]+)", report_str)
    if match:
        return float(match.group(1))
    return float("nan")


def print_metrics(
    eval_result: dict,
    title: str = "",
    minority_class_name: str = "Walk",
) -> None:
    """Print a concise metrics summary highlighting minority-class performance.

    Parameters
    ----------
    eval_result : dict returned by ``src.utils.training.evaluate``
    title : str label for this experiment
    minority_class_name : str name of the minority class to highlight
    """
    acc      = eval_result["accuracy"]
    macro_f1 = extract_macro_f1(eval_result["report"])
    recall   = get_minority_recall(eval_result["report"], minority_class_name)

    print(f"{'─'*50}")
    print(f"  {title}")
    print(f"{'─'*50}")
    print(f"  Overall Accuracy : {acc:.3f}")
    print(f"  Macro-avg F1     : {macro_f1:.3f}")
    print(f"  {minority_class_name} Recall    : {recall:.3f}  ← minority class")
    print()


# ── Visualisation ─────────────────────────────────────────────────────────────

def plot_normalized_confusion(
    cm: np.ndarray,
    class_names: List[str],
    title: str = "Confusion Matrix",
) -> plt.Figure:
    """Plot a row-normalised confusion matrix as a percentage heatmap.

    Row normalisation means each row sums to 100 %, so the diagonal entry for
    class c is the **recall** of that class — immediately showing which classes
    are misclassified most often.

    Parameters
    ----------
    cm : (n, n) integer confusion matrix from sklearn
    class_names : list of class-name strings (length n)
    title : str

    Returns
    -------
    fig : matplotlib Figure
    """
    # Row-normalise: divide each row by its sum → values in [0, 1]
    row_sums = cm.sum(axis=1, keepdims=True).astype(float)
    cm_norm = np.divide(cm, row_sums, where=row_sums != 0)

    n = len(class_names)
    size = max(7, n * 0.7)
    fontsize = 8 if n <= 10 else 6

    fig, ax = plt.subplots(figsize=(size, size - 0.5))
    sns.heatmap(
        cm_norm,
        annot=True,
        fmt=".0%",          # display as percentage
        cmap="Blues",
        vmin=0.0, vmax=1.0,
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
        annot_kws={"size": fontsize},
        linewidths=0.3,
    )
    ax.set_xlabel("Predicted", fontsize=fontsize + 2)
    ax.set_ylabel("True", fontsize=fontsize + 2)
    ax.set_title(title, fontsize=fontsize + 3, pad=10)
    ax.tick_params(axis="x", labelsize=fontsize, rotation=45)
    ax.tick_params(axis="y", labelsize=fontsize, rotation=0)
    plt.tight_layout()
    return fig


def plot_training_curves_imbalance(
    train_losses: List[float],
    val_accs: List[float],
    title: str = "",
) -> plt.Figure:
    """Plot training loss and validation accuracy side-by-side for one experiment."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 3.5))

    ax1.plot(train_losses, color="steelblue")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Train Loss")
    ax1.set_title(f"{title} — Loss")

    ax2.plot(val_accs, color="seagreen")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Val Accuracy")
    ax2.set_title(f"{title} — Val Accuracy")
    ax2.set_ylim(0, 1.05)

    plt.tight_layout()
    return fig
