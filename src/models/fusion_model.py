"""Late-fusion multimodal model combining video and inertial streams.

Late fusion means each modality has its own encoder that produces a fixed-size
feature vector; these vectors are concatenated and fed to a shared MLP
classifier.  This design makes it straightforward to handle missing modalities
(just zero the corresponding feature vector) and to reuse pretrained unimodal
encoders.
"""

from typing import Optional

import torch
import torch.nn as nn

from src.utils.config import NUM_CLASSES, VIDEO_FEAT_DIM, INERTIAL_FEAT_DIM
from src.models.inertial_model import InertialCNN
from src.models.video_model import VideoClassifier


class MultimodalFusion(nn.Module):
    """Late fusion: independent encoders → concatenated features → joint classifier.

    Parameters
    ----------
    n_classes : int
        Number of output classes.
    dropout : float
        Dropout probability in the joint classifier MLP.
    modality_dropout : float
        During training, probability of zeroing out one entire modality's
        feature vector (chosen uniformly between video and inertial).
        Set to 0 to disable.  Part 4B uses 0.3.
    video_encoder : VideoClassifier or None
        If provided, reuses the supplied (pretrained) encoder.
        Otherwise creates a fresh one.
    inertial_encoder : InertialCNN or None
        Same as above for the inertial stream.
    """

    def __init__(
        self,
        n_classes: int = NUM_CLASSES,
        dropout: float = 0.3,
        modality_dropout: float = 0.0,
        video_encoder: Optional[VideoClassifier] = None,
        inertial_encoder: Optional[InertialCNN] = None,
    ):
        super().__init__()
        self.modality_dropout = modality_dropout

        # Use supplied encoders or create fresh ones.
        # Either way, set them to return feature vectors, not logits.
        self.video_encoder = video_encoder or VideoClassifier(
            n_classes=n_classes, dropout=0.0, return_features=True,
        )
        self.video_encoder.return_features = True

        self.inertial_encoder = inertial_encoder or InertialCNN(
            n_classes=n_classes, dropout=0.0, return_features=True,
        )
        self.inertial_encoder.return_features = True

        # Joint classifier operating on the concatenated feature vector.
        # LayerNorm stabilises training when the two feature vectors have
        # different scales.
        fused_dim = VIDEO_FEAT_DIM + INERTIAL_FEAT_DIM  # 512 + 128 = 640
        self.classifier = nn.Sequential(
            nn.LayerNorm(fused_dim),
            nn.Dropout(dropout),
            nn.Linear(fused_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, n_classes),
        )

    def forward(
        self,
        video: torch.Tensor,
        inertial: torch.Tensor,
        mask_video: bool = False,
        mask_inertial: bool = False,
    ) -> torch.Tensor:
        """Forward pass through both encoders, fusion, and classifier.

        Parameters
        ----------
        video : (B, 3, T, H, W)
        inertial : (B, 6, L)
        mask_video : force-zero the video features (Part 4A eval — sensor only)
        mask_inertial : force-zero the inertial features (Part 4A eval — video only)
        """
        # Extract features from each encoder
        v_feat = self.video_encoder.extract_features(video)      # (B, 512)
        i_feat = self.inertial_encoder.extract_features(inertial) # (B, 128)

        # Modality dropout during training (Part 4B):
        # With probability modality_dropout, zero out one randomly-chosen
        # modality.  This forces the classifier to learn from partial input.
        if self.training and self.modality_dropout > 0:
            if torch.rand(1).item() < self.modality_dropout:
                # Pick which modality to drop (50/50)
                if torch.rand(1).item() < 0.5:
                    v_feat = torch.zeros_like(v_feat)
                else:
                    i_feat = torch.zeros_like(i_feat)

        # Explicit masking for missing-modality evaluation (Part 4A)
        if mask_video:
            v_feat = torch.zeros_like(v_feat)
        if mask_inertial:
            i_feat = torch.zeros_like(i_feat)

        # Concatenate and classify
        fused = torch.cat([v_feat, i_feat], dim=1)  # (B, 640)
        return self.classifier(fused)

    @torch.no_grad()
    def extract_fusion_vector(
        self,
        video: torch.Tensor,
        inertial: torch.Tensor,
    ) -> torch.Tensor:
        """Return the 640-d concatenated feature vector without passing through the classifier.

        This exposes the bottleneck representation used by Parts 6 & 7 (Conformal
        Prediction and VAE-based OOD detection).  Modality dropout is never
        applied here since this is always used in evaluation mode.

        Parameters
        ----------
        video : (B, 3, T, H, W)
        inertial : (B, 6, L)

        Returns
        -------
        fused : (B, 640) tensor on the same device as the model
        """
        self.eval()
        v_feat = self.video_encoder.extract_features(video)       # (B, 512)
        i_feat = self.inertial_encoder.extract_features(inertial) # (B, 128)
        return torch.cat([v_feat, i_feat], dim=1)                 # (B, 640)
