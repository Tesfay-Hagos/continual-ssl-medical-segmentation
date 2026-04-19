"""
SparK: Sparse Masked Modeling for Convolutional Networks (ICLR 2023).
arXiv: 2301.03580  |  145 citations

This module wraps the SparK masking + reconstruction logic so it can be
applied to a standard U-Net encoder.  We do NOT re-implement SparK from
scratch; instead, we clone the official repo and import from it.

Setup (run once):
    git clone https://github.com/keyu-tian/SparK.git external/SparK

Reference: Tian et al., "Designing BERT for Convolutional Networks:
Sparse and Hierarchical Masked Modeling", ICLR 2023.
"""

import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# ── Patch-level random masking (used independently of the SparK repo) ─────────

def random_masking_3d(x: torch.Tensor,
                      mask_ratio: float = 0.75,
                      patch_size: int = 16) -> tuple:
    """
    Apply random patch masking to a 3D volume tensor.

    Args:
        x:           (B, C, D, H, W) input volume
        mask_ratio:  fraction of patches to mask
        patch_size:  spatial size of each cubic patch

    Returns:
        x_masked:    x with masked patches zeroed out
        mask:        binary mask (1 = kept, 0 = masked), shape (B, nD, nH, nW)
    """
    B, C, D, H, W = x.shape
    nD, nH, nW = D // patch_size, H // patch_size, W // patch_size
    num_patches = nD * nH * nW
    num_keep = max(1, int(num_patches * (1 - mask_ratio)))

    # Random shuffle to select kept patches
    noise  = torch.rand(B, num_patches, device=x.device)
    ids_shuffle = torch.argsort(noise, dim=1)
    mask_flat = torch.ones(B, num_patches, device=x.device)
    mask_flat.scatter_(1, ids_shuffle[:, num_keep:], 0)  # 0 = masked
    mask = mask_flat.view(B, nD, nH, nW)

    # Expand mask to full volume resolution
    mask_expanded = mask.unsqueeze(1)  # (B, 1, nD, nH, nW)
    mask_expanded = mask_expanded.repeat(1, 1, patch_size, patch_size, patch_size)
    mask_expanded = mask_expanded.view(B, 1, D, H, W).float()

    x_masked = x * mask_expanded
    return x_masked, mask


class SparKPretrainer(nn.Module):
    """
    Lightweight SparK-style pretraining wrapper for a U-Net encoder.

    Pipeline:
      1. Randomly mask input patches (default 75%)
      2. Run the sparse masked input through the encoder
      3. A lightweight pixel-shuffle decoder reconstructs the original patches
      4. Loss = MSE on masked patches only

    This gives the encoder strong patch-level representation learning
    without requiring the full SparK infrastructure.
    """

    def __init__(self,
                 encoder: nn.Module,
                 encoder_out_channels: int = 512,
                 patch_size: int = 16,
                 mask_ratio: float = 0.75,
                 spatial_dims: int = 3):
        super().__init__()
        self.encoder       = encoder
        self.patch_size    = patch_size
        self.mask_ratio    = mask_ratio
        self.spatial_dims  = spatial_dims

        # Lightweight decoder: 1×1 conv + pixel shuffle to restore resolution
        upsample_factor = patch_size ** spatial_dims
        self.decoder = nn.Sequential(
            nn.Conv3d(encoder_out_channels, 256, kernel_size=1),
            nn.GELU(),
            nn.Conv3d(256, upsample_factor, kernel_size=1),
        )
        self._ps = patch_size

    def forward(self, x: torch.Tensor):
        """Returns (loss, reconstruction, mask)."""
        x_masked, mask = random_masking_3d(x, self.mask_ratio, self.patch_size)

        # Encode the masked input
        enc_out = self.encoder(x_masked)
        # If encoder returns list of features, take the deepest
        if isinstance(enc_out, (list, tuple)):
            enc_out = enc_out[-1]

        # Decode to patch predictions
        pred_patches = self.decoder(enc_out)  # (B, patch_size^3, nD, nH, nW)

        # Reshape predictions to full volume
        B, _, nD, nH, nW = pred_patches.shape
        ps = self.patch_size
        pred = pred_patches.view(B, ps, ps, ps, nD, nH, nW)
        pred = pred.permute(0, 4, 1, 5, 2, 6, 3).contiguous()
        pred = pred.view(B, 1, nD * ps, nH * ps, nW * ps)

        # MSE loss only on masked patches
        mask_expanded = (1 - mask).unsqueeze(1).float()
        mask_up = F.interpolate(mask_expanded,
                                size=x.shape[2:],
                                mode="nearest")
        loss = F.mse_loss(pred * mask_up, x * mask_up)
        return loss, pred, mask
