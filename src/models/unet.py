"""
U-Net encoder-decoder for volumetric medical image segmentation.

Uses MONAI's UNet which supports:
  - 2D / 3D / 2.5D
  - Residual connections (residual=True recommended)
  - Variable depth via channels tuple

The encoder (down-sampling path) is exposed separately so SparK
pretraining can operate on it before attaching the decoder head.
"""

import torch
import torch.nn as nn
from monai.networks.nets import UNet as MonaiUNet
from monai.networks.layers import Norm


def build_unet(
    spatial_dims: int = 3,
    in_channels: int = 1,
    out_channels: int = 2,
    channels: tuple = (32, 64, 128, 256, 512),
    strides: tuple = (2, 2, 2, 2),
    num_res_units: int = 2,
    dropout: float = 0.1,
) -> MonaiUNet:
    """Return a MONAI U-Net.  Default config fits in ~12 GB VRAM for 96^3 patches."""
    return MonaiUNet(
        spatial_dims=spatial_dims,
        in_channels=in_channels,
        out_channels=out_channels,
        channels=channels,
        strides=strides,
        num_res_units=num_res_units,
        norm=Norm.BATCH,
        dropout=dropout,
    )


class UNetWithEncoder(nn.Module):
    """
    Wrapper that exposes the encoder stem separately from the full U-Net,
    so SparK pretraining can pretrain just the encoder, then the full
    model (encoder + decoder) is used for fine-tuning.
    """

    def __init__(self, unet: MonaiUNet):
        super().__init__()
        self.unet = unet

    def encode(self, x: torch.Tensor) -> list:
        """Return list of encoder feature maps (skip connections)."""
        features = []
        out = x
        for down in self.unet.model:
            out = down(out)
            features.append(out)
            if len(features) == len(self.unet.model) - 1:
                break
        return features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.unet(x)

    def load_pretrained_encoder(self, ckpt_path: str, strict: bool = False):
        """Load SparK-pretrained encoder weights into the U-Net encoder layers."""
        state = torch.load(ckpt_path, map_location="cpu")
        encoder_state = {k: v for k, v in state.items() if "decoder" not in k}
        missing, unexpected = self.unet.load_state_dict(encoder_state, strict=strict)
        print(f"Loaded pretrained encoder: {len(encoder_state)} keys")
        if missing:
            print(f"  Missing  : {missing[:5]}")
        if unexpected:
            print(f"  Unexpected: {unexpected[:5]}")
