"""
SparK: Sparse Masked Modeling for Convolutional Networks (ICLR 2023).
arXiv: 2301.03580

Wraps SparK-style masking + reconstruction for a MONAI U-Net encoder.

Pipeline:
  1. Randomly mask 75% of 3D patches
  2. Run masked input through the U-Net encoder path ONLY (no decoder)
  3. Lightweight decoder reconstructs original patches from 512-ch bottleneck
  4. MSE loss on masked patches trains the encoder representations
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Patch masking ─────────────────────────────────────────────────────────────

def random_masking_3d(x: torch.Tensor,
                      mask_ratio: float = 0.75,
                      patch_size: int = 16) -> tuple:
    """
    Randomly zero out cubic patches of a 3D volume.

    Returns:
        x_masked : input with masked patches set to zero
        mask     : binary (B, nD, nH, nW), 1 = kept, 0 = masked
    """
    B, C, D, H, W = x.shape
    nD, nH, nW    = D // patch_size, H // patch_size, W // patch_size
    num_patches   = nD * nH * nW
    num_keep      = max(1, int(num_patches * (1 - mask_ratio)))

    noise       = torch.rand(B, num_patches, device=x.device)
    ids_shuffle = torch.argsort(noise, dim=1)
    mask_flat   = torch.ones(B, num_patches, device=x.device)
    mask_flat.scatter_(1, ids_shuffle[:, num_keep:], 0)
    mask        = mask_flat.view(B, nD, nH, nW)

    mask_exp    = mask.unsqueeze(1).repeat(1, 1, patch_size, patch_size, patch_size)
    mask_exp    = mask_exp.view(B, 1, D, H, W).float()
    return x * mask_exp, mask


# ── Encoder-path traversal ────────────────────────────────────────────────────

def _encode_down_path(unet_model: nn.Sequential,
                      x: torch.Tensor) -> torch.Tensor:
    """
    Traverse a MONAI UNet model, applying ONLY the down (encoder) blocks.

    MONAI UNet is structured as nested SkipConnections:
        Sequential(
            ResidualUnit(in→c0),          <- top encoder
            SkipConnection(Sequential(
                ResidualUnit(c0→c1),       <- encoder level 2
                SkipConnection(Sequential(
                    ...
                    ResidualUnit(cN-1→cN), <- bottleneck
                    UpSample(...)          <- decoder (NOT applied)
                )),
                UpSample(...)
            )),
            UpSample(...)
        )

    We recurse into each SkipConnection.submodule and stop before UpSample,
    returning the bottleneck feature map (B, cN, D', H', W').

    At the deepest level MONAI's SkipConnection.submodule is a bare ResidualUnit
    (not a Sequential), so we handle that as the bottleneck base case.
    """
    # Bottleneck base case: deepest submodule is a bare module, not Sequential
    if not isinstance(unet_model, nn.Sequential):
        return unet_model(x)

    out = unet_model[0](x)          # apply top encoder block

    if len(unet_model) > 1:
        next_block = unet_model[1]
        if hasattr(next_block, "submodule"):  # it's a SkipConnection → go deeper
            out = _encode_down_path(next_block.submodule, out)
        # else: next block is UpSample → we are already at the bottleneck

    return out


# ── SparK pretrainer ──────────────────────────────────────────────────────────

class SparKPretrainer(nn.Module):
    """
    SparK-style pretraining wrapper for a MONAI U-Net.

    encoder must be a bare MONAI UNet (build_unet() output).
    Only the encoder (down-path) runs during pretraining; the UNet decoder
    is completely bypassed. After pretraining, encoder.state_dict() is saved
    and loaded into UNetWithEncoder for downstream fine-tuning.
    """

    def __init__(self,
                 encoder: nn.Module,
                 encoder_out_channels: int = 512,
                 patch_size: int = 16,
                 mask_ratio: float = 0.75):
        super().__init__()
        self.encoder            = encoder
        self.patch_size         = patch_size
        self.mask_ratio         = mask_ratio

        upsample_factor = patch_size ** 3          # e.g. 16^3 = 4096
        self.decoder = nn.Sequential(
            nn.Conv3d(encoder_out_channels, 256, kernel_size=1),
            nn.GELU(),
            nn.Conv3d(256, upsample_factor, kernel_size=1),
        )

    def forward(self, x: torch.Tensor):
        """Returns (loss, reconstruction, mask)."""
        x_masked, mask = random_masking_3d(x, self.mask_ratio, self.patch_size)

        # Run ONLY the encoder down-path — never touches the UNet decoder
        enc_out = _encode_down_path(self.encoder.model, x_masked)

        # Lightweight decoder predicts each patch independently
        pred_patches = self.decoder(enc_out)        # (B, ps^3, nD, nH, nW)

        B, _, nD, nH, nW = pred_patches.shape
        ps   = self.patch_size
        pred = pred_patches.view(B, ps, ps, ps, nD, nH, nW)
        pred = pred.permute(0, 4, 1, 5, 2, 6, 3).contiguous()
        pred = pred.view(B, 1, nD * ps, nH * ps, nW * ps)

        # MSE only on masked-out patches
        mask_up = F.interpolate(
            (1 - mask).unsqueeze(1).float(), size=x.shape[2:], mode="nearest")
        loss = F.mse_loss(pred * mask_up, x * mask_up)
        return loss, pred, mask
