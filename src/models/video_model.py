"""Pretrained R3D-18 wrapper for video action classification.

R3D-18 is a 3-D ResNet with 18 layers, pretrained on Kinetics-400.
We freeze the early layers (stem, layer1, layer2) and fine-tune the
later layers (layer3, layer4) plus a new classification head.
"""

import torch
import torch.nn as nn
from torchvision.models.video import r3d_18, R3D_18_Weights

from src.utils.config import NUM_CLASSES, VIDEO_FEAT_DIM


class VideoClassifier(nn.Module):
    """R3D-18 pretrained on Kinetics-400, fine-tuned for UTD-MHAD.

    Freezing strategy
    -----------------
    * **Frozen**: stem, layer1, layer2 — these learn generic spatiotemporal
      features that transfer well, so we keep them fixed.
    * **Trainable**: layer3, layer4, classifier — these are fine-tuned on
      the target dataset.

    Parameters
    ----------
    n_classes : int
        Number of output classes.
    dropout : float
        Dropout probability before the final linear layer.
    return_features : bool
        If True, ``forward`` returns the 512-d feature vector instead of logits.
    """

    def __init__(
        self,
        n_classes: int = NUM_CLASSES,
        dropout: float = 0.3,
        return_features: bool = False,
    ):
        super().__init__()
        self.return_features = return_features

        # Load the pretrained R3D-18 backbone from torchvision
        backbone = r3d_18(weights=R3D_18_Weights.DEFAULT)

        # Freeze the early layers to retain generic Kinetics-400 features
        for name, param in backbone.named_parameters():
            if any(name.startswith(prefix) for prefix in ("stem", "layer1", "layer2")):
                param.requires_grad = False

        # Decompose the backbone so we can intercept features before the original FC
        self.stem = backbone.stem       # initial 3-D conv + BN + ReLU
        self.layer1 = backbone.layer1   # frozen
        self.layer2 = backbone.layer2   # frozen
        self.layer3 = backbone.layer3   # fine-tuned
        self.layer4 = backbone.layer4   # fine-tuned
        self.avgpool = backbone.avgpool # spatio-temporal global avg-pool → (B, 512, 1, 1, 1)

        # New classification head replacing the original 400-class FC
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(VIDEO_FEAT_DIM, n_classes),
        )

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Return the 512-d feature vector (before the classifier).

        Used by the fusion model to get intermediate representations.
        """
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        return x.flatten(1)        # (B, 512)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: video tensor of shape (B, 3, T, H, W).

        Returns:
            (B, n_classes) logits  — or (B, 512) features if return_features=True.
        """
        feat = self.extract_features(x)
        if self.return_features:
            return feat
        return self.classifier(feat)
