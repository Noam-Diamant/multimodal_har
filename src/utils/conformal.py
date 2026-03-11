"""Conformal Prediction utilities for the UTD-MHAD multimodal HAR system.

Implements split conformal prediction (Venn–Abers / RAPS variant) following:

    Angelopoulos, A. N., & Bates, S. (2021).
    "A gentle introduction to conformal prediction and distribution-free
    uncertainty quantification." arXiv:2107.07511.

Split conformal prediction proceeds in three steps:

1.  Compute a *non-conformity score* for each calibration sample:
        s_i = 1 - f̂(x_i)[y_i]
    where f̂(x_i)[y_i] is the softmax probability assigned to the true class.

2.  Choose a miscoverage level α (e.g. 0.05 for 95 % coverage) and compute
    the corrected quantile of the calibration scores:
        q̂ = Quantile(s_1, …, s_n ; ⌈(n+1)(1−α)⌉/n)

3.  At test time, the *prediction set* is all classes whose softmax probability
    is at least 1 − q̂:
        C(x_test) = { y : f̂(x_test)[y] ≥ 1 − q̂ }

    The marginal coverage guarantee is P(y_true ∈ C(x_test)) ≥ 1 − α.
"""

from __future__ import annotations

import math
from typing import List, Tuple

import numpy as np


# ── Core calibration ──────────────────────────────────────────────────────────

def compute_nonconformity_scores(
    softmax_probs: np.ndarray,
    labels: np.ndarray,
) -> np.ndarray:
    """Compute the non-conformity score for each calibration sample.

    Parameters
    ----------
    softmax_probs : (n, num_classes) array of softmax probabilities.
    labels : (n,) integer array of true class indices.

    Returns
    -------
    scores : (n,) array of non-conformity scores in [0, 1].
        Higher score = less conforming = model is less confident in the
        true class.
    """
    n = len(labels)
    # Index into each row by the true class index to get the probability
    # assigned to the ground-truth label
    true_class_probs = softmax_probs[np.arange(n), labels]
    return 1.0 - true_class_probs


def calibrate(
    softmax_probs: np.ndarray,
    labels: np.ndarray,
    alpha: float = 0.05,
) -> float:
    """Calibrate and return the conformal quantile q̂.

    Parameters
    ----------
    softmax_probs : (n, num_classes) calibration-set softmax probabilities.
    labels : (n,) true class indices for the calibration set.
    alpha : float
        Desired miscoverage level.  Coverage guarantee is 1 − α.
        Default 0.05 → 95 % marginal coverage.

    Returns
    -------
    q_hat : float
        The score threshold.  At test time, include class y in the prediction
        set if softmax_probs[y] ≥ 1 − q_hat.
    """
    scores = compute_nonconformity_scores(softmax_probs, labels)
    n = len(scores)

    # Corrected quantile level: ceil((n+1)(1-α)) / n
    # This finite-sample correction guarantees marginal coverage.
    quantile_level = math.ceil((n + 1) * (1.0 - alpha)) / n
    # Clip to [0, 1] in case n is very small
    quantile_level = min(quantile_level, 1.0)

    q_hat = float(np.quantile(scores, quantile_level))
    return q_hat


# ── Prediction set construction ───────────────────────────────────────────────

def predict_set(
    softmax_probs: np.ndarray,
    q_hat: float,
) -> List[List[int]]:
    """Construct conformal prediction sets for a batch of test samples.

    A class y is included when its softmax probability is at least 1 − q̂,
    which is equivalent to its non-conformity score being at most q̂.

    Parameters
    ----------
    softmax_probs : (m, num_classes) array of softmax probabilities for
        m test samples.
    q_hat : float
        Calibrated quantile from ``calibrate``.

    Returns
    -------
    prediction_sets : list of m lists, each containing the indices of
        included classes.  An empty list means the sample is likely OOD
        (no class meets the threshold), which is a useful OOD signal
        (see Part 7).
    """
    threshold = 1.0 - q_hat
    prediction_sets = []
    for probs in softmax_probs:
        # Include every class whose probability exceeds the threshold
        included = [int(k) for k, p in enumerate(probs) if p >= threshold]
        prediction_sets.append(included)
    return prediction_sets


# ── Convenience: set sizes and coverage metrics ────────────────────────────────

def set_sizes(prediction_sets: List[List[int]]) -> np.ndarray:
    """Return an array of prediction-set sizes (one entry per sample)."""
    return np.array([len(s) for s in prediction_sets])


def empirical_coverage(
    prediction_sets: List[List[int]],
    labels: np.ndarray,
) -> float:
    """Fraction of test samples whose true label is inside the prediction set.

    Should be ≥ 1 − α when the calibration and test distributions match.

    Parameters
    ----------
    prediction_sets : output of ``predict_set``.
    labels : (m,) true class indices for the test samples.

    Returns
    -------
    coverage : float in [0, 1].
    """
    covered = sum(
        1 for s, y in zip(prediction_sets, labels) if int(y) in s
    )
    return covered / max(len(labels), 1)


def ood_flags(prediction_sets: List[List[int]]) -> np.ndarray:
    """Return a boolean array: True where the prediction set is empty (OOD signal)."""
    return np.array([len(s) == 0 for s in prediction_sets])
