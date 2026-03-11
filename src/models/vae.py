"""Variational Autoencoder (VAE) for OOD detection on fusion feature vectors.

The VAE is trained on the 640-d late-fusion bottleneck vectors extracted from
in-distribution (ID) training samples.  At inference time, a high reconstruction
error signals that the input is out-of-distribution.

Architecture
------------
Encoder : Linear(input_dim, 256) → ReLU → Linear(256, latent_dim × 2)
            └─ first latent_dim dims → μ
            └─ last  latent_dim dims → log σ²

Decoder : Linear(latent_dim, 256) → ReLU → Linear(256, input_dim)

Training objective (ELBO):
    L = E[||x - x̂||²] + β · KL( q(z|x) ∥ p(z) )

where KL( N(μ,σ²) ∥ N(0,I) ) = −½ Σ(1 + log σ² − μ² − σ²).

"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.utils.config import VIDEO_FEAT_DIM, INERTIAL_FEAT_DIM

# Default input dimension matches the MultimodalFusion bottleneck
FUSION_DIM = VIDEO_FEAT_DIM + INERTIAL_FEAT_DIM  # 512 + 128 = 640


class FusionVAE(nn.Module):
    """Variational Autoencoder operating on 640-d late-fusion feature vectors.

    Parameters
    ----------
    input_dim : int
        Dimensionality of the input feature vector.  Should match
        ``VIDEO_FEAT_DIM + INERTIAL_FEAT_DIM`` (default 640).
    latent_dim : int
        Dimensionality of the latent Gaussian space.  Default 64.
    hidden_dim : int
        Width of the hidden layer shared by encoder and decoder.  Default 256.
    beta : float
        Weight applied to the KL term in the ELBO (β-VAE style).
        β = 1 is the standard VAE; smaller values make reconstruction
        dominate, which is often better for anomaly detection.  Default 1.0.
    """

    def __init__(
        self,
        input_dim: int = FUSION_DIM,
        latent_dim: int = 64,
        hidden_dim: int = 256,
        beta: float = 1.0,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.beta = beta

        # ── Encoder ─────────────────────────────────────────────────────────
        # Projects the input to a hidden representation, then outputs μ and
        # log σ² as a single vector of size 2 × latent_dim.
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, latent_dim * 2),  # μ ∥ log σ²
        )

        # ── Decoder ─────────────────────────────────────────────────────────
        # Reconstructs the input from a latent sample z ~ q(z|x).
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, input_dim),
        )

    # ── Encoder / decoder helpers ─────────────────────────────────────────────

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode input to Gaussian parameters (μ, log σ²).

        Parameters
        ----------
        x : (B, input_dim)

        Returns
        -------
        mu : (B, latent_dim)
        log_var : (B, latent_dim)
        """
        h = self.encoder(x)
        # Split the output into mean and log-variance halves
        mu, log_var = h.chunk(2, dim=-1)
        return mu, log_var

    def reparameterise(
        self, mu: torch.Tensor, log_var: torch.Tensor
    ) -> torch.Tensor:
        """Sample z ~ N(μ, σ²) using the reparameterisation trick.

        During evaluation, returns the mean directly (deterministic).
        """
        if self.training:
            # σ = exp(0.5 · log σ²);  z = μ + σ · ε,  ε ~ N(0, I)
            std = torch.exp(0.5 * log_var)
            eps = torch.randn_like(std)
            return mu + std * eps
        else:
            # Deterministic decode from the mean at test time
            return mu

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode a latent vector back to input space.

        Parameters
        ----------
        z : (B, latent_dim)

        Returns
        -------
        x_hat : (B, input_dim)  reconstructed input
        """
        return self.decoder(z)

    # ── Full forward pass ─────────────────────────────────────────────────────

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Encode → reparameterise → decode.

        Returns
        -------
        x_hat : (B, input_dim) reconstruction
        mu    : (B, latent_dim) posterior mean
        log_var : (B, latent_dim) posterior log-variance
        """
        mu, log_var = self.encode(x)
        z = self.reparameterise(mu, log_var)
        x_hat = self.decode(z)
        return x_hat, mu, log_var

    # ── Loss ─────────────────────────────────────────────────────────────────

    def loss(
        self,
        x: torch.Tensor,
        x_hat: torch.Tensor,
        mu: torch.Tensor,
        log_var: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute the ELBO loss (negative Evidence Lower BOund).

        Parameters
        ----------
        x     : (B, input_dim) original input
        x_hat : (B, input_dim) reconstruction
        mu    : (B, latent_dim) posterior mean
        log_var : (B, latent_dim) posterior log-variance

        Returns
        -------
        total_loss : scalar — ELBO = recon_loss + β × kl_loss
        recon_loss : scalar — mean squared error per sample, summed over dims
        kl_loss    : scalar — KL divergence per sample, summed over dims
        """
        # Reconstruction loss: mean over batch, sum over feature dimensions
        recon_loss = F.mse_loss(x_hat, x, reduction="sum") / x.size(0)

        # Analytical KL divergence KL(N(μ,σ²) ‖ N(0,I))
        # = −½ Σ_j (1 + log σ²_j − μ²_j − exp(log σ²_j))
        kl_loss = -0.5 * torch.sum(
            1 + log_var - mu.pow(2) - log_var.exp(), dim=1
        ).mean()

        total_loss = recon_loss + self.beta * kl_loss
        return total_loss, recon_loss, kl_loss

    # ── OOD scoring ───────────────────────────────────────────────────────────

    @torch.no_grad()
    def reconstruction_error(self, x: torch.Tensor) -> torch.Tensor:
        """Compute per-sample mean squared reconstruction error.

        Higher error → sample is more likely to be OOD.

        Parameters
        ----------
        x : (B, input_dim)

        Returns
        -------
        errors : (B,) per-sample MSE
        """
        self.eval()
        x_hat, _, _ = self.forward(x)
        # Per-sample MSE (mean over feature dimensions)
        return F.mse_loss(x_hat, x, reduction="none").mean(dim=-1)
