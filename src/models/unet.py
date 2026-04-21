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
    """Return a MONAI U-Net.  Default config fits in ~12 GB VRAM for 96^3 patches.

    InstanceNorm instead of BatchNorm: BN statistics are unstable at batch_size=2
    (standard for 3D medical volumes). nnU-Net (Isensee et al. 2021) uses InstanceNorm
    for the same reason.
    """
    return MonaiUNet(
        spatial_dims=spatial_dims,
        in_channels=in_channels,
        out_channels=out_channels,
        channels=channels,
        strides=strides,
        num_res_units=num_res_units,
        norm=Norm.INSTANCE,
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.unet(x)

    def parameter_groups(self, base_lr: float,
                         encoder_lr_scale: float = 0.1) -> list:
        """
        Differential LR: ConvTranspose (decoder up-path) gets base_lr;
        everything else (encoder Conv3d, norms, residuals) gets base_lr * scale.

        Only call when loading pretrained encoder — protects learned representations
        from being overwritten in early fine-tuning steps.
        """
        decoder_ids = {
            id(p)
            for m in self.unet.modules()
            if isinstance(m, (nn.ConvTranspose3d, nn.ConvTranspose2d))
            for p in m.parameters(recurse=False)
        }
        enc = [p for p in self.unet.parameters() if id(p) not in decoder_ids]
        dec = [p for p in self.unet.parameters() if id(p) in decoder_ids]
        return [
            {"params": enc, "lr": base_lr * encoder_lr_scale},
            {"params": dec, "lr": base_lr},
        ]

    def load_pretrained_encoder(self, ckpt_path: str, strict: bool = False):
        """Load SparK-pretrained encoder weights into the U-Net."""
        state = torch.load(ckpt_path, map_location="cpu")
        missing, unexpected = self.unet.load_state_dict(state, strict=strict)
        print(f"Loaded pretrained encoder: {len(state)} keys "
              f"| missing={len(missing)} | unexpected={len(unexpected)}")
        if missing:
            print(f"  Missing   : {missing[:3]}")
        if unexpected:
            print(f"  Unexpected: {unexpected[:3]}")
