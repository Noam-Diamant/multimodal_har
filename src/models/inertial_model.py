"""1D-CNN for inertial (accelerometer + gyroscope) classification.

The model processes 6-channel time-series of fixed length and produces
either class logits or a 128-d feature vector (for late fusion).
"""

import torch
import torch.nn as nn

from src.utils.config import INERTIAL_CHANNELS, INERTIAL_LEN, NUM_CLASSES, INERTIAL_FEAT_DIM


class InertialCNN(nn.Module):
    """Three-block 1D-CNN for 6-axis inertial time-series.

    Architecture
    ------------
    Block 1: Conv1d(6 → 64, k=5) → BN → ReLU → MaxPool(2)
    Block 2: Conv1d(64 → 128, k=5) → BN → ReLU → MaxPool(2)
    Block 3: Conv1d(128 → 128, k=3) → BN → ReLU → GlobalAvgPool(1)
    Head:    Dropout → Linear(128 → n_classes)

    Total ≈ 94 K parameters — fast to train even on CPU.

    Parameters
    ----------
    n_classes : int
        Number of output classes.
    dropout : float
        Dropout probability before the final linear layer.
    return_features : bool
        If True, ``forward`` returns the 128-d feature vector instead of logits.
        Used when this model is plugged into the fusion model.
    """

    def __init__(
        self,
        n_classes: int = NUM_CLASSES,
        dropout: float = 0.3,
        return_features: bool = False,
    ):
        super().__init__()
        self.return_features = return_features

        # Three convolutional blocks with increasing channel depth.
        # Each block: Conv1d → BatchNorm → ReLU → pooling.
        self.features = nn.Sequential(
            # Block 1: (B, 6, 150) → (B, 64, 75)
            nn.Conv1d(INERTIAL_CHANNELS, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),

            # Block 2: (B, 64, 75) → (B, 128, 37)
            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),

            # Block 3: (B, 128, 37) → (B, 128, 1) via global average pooling
            nn.Conv1d(128, INERTIAL_FEAT_DIM, kernel_size=3, padding=1),
            nn.BatchNorm1d(INERTIAL_FEAT_DIM),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1),  # collapse the time dimension
        )

        # Classification head maps the 128-d feature to class logits
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(INERTIAL_FEAT_DIM, n_classes),
        )

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Return the 128-d feature vector (before the classifier).

        Used by the fusion model to get intermediate representations.
        """
        x = self.features(x)          # (B, 128, 1)
        return x.squeeze(-1)          # (B, 128)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: inertial tensor of shape (B, 6, T).

        Returns:
            (B, n_classes) logits  — or (B, 128) features if return_features=True.
        """
        feat = self.extract_features(x)
        if self.return_features:
            return feat
        return self.classifier(feat)
