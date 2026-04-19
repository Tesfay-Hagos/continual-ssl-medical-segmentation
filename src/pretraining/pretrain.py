"""
SparK pretraining loop for the U-Net encoder.

Run:
    python scripts/pretrain.py --config configs/pretraining.yaml
"""

import os
import time
import argparse
import yaml

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from monai.data import CacheDataset
from monai.transforms import Compose, LoadImaged, EnsureChannelFirstd, \
    Spacingd, Orientationd, ScaleIntensityRanged, NormalizeIntensityd, \
    RandSpatialCropd, RandFlipd, ToTensord

from models.unet import build_unet
from pretraining.spark import SparKPretrainer
from data.datasets import get_unlabelled_files


def get_ssl_transforms(patch_size: int = 96) -> Compose:
    return Compose([
        LoadImaged(keys=["image"]),
        EnsureChannelFirstd(keys=["image"]),
        Spacingd(keys=["image"], pixdim=(1.5, 1.5, 2.0), mode="bilinear"),
        Orientationd(keys=["image"], axcodes="RAS"),
        NormalizeIntensityd(keys=["image"], nonzero=True),
        RandSpatialCropd(keys=["image"],
                         roi_size=(patch_size, patch_size, patch_size),
                         random_size=False),
        RandFlipd(keys=["image"], prob=0.5, spatial_axis=0),
        ToTensord(keys=["image"]),
    ])


def pretrain(cfg: dict):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Data ──
    files = get_unlabelled_files(cfg["data_root"])
    ds = CacheDataset(files, transform=get_ssl_transforms(cfg["patch_size"]),
                      cache_rate=cfg.get("cache_rate", 0.05))
    loader = DataLoader(ds, batch_size=cfg["batch_size"], shuffle=True,
                        num_workers=cfg["num_workers"], pin_memory=True)
    print(f"Pretraining on {len(files)} volumes, {len(loader)} batches/epoch")

    # ── Model ──
    encoder = build_unet(
        in_channels=1, out_channels=2,
        channels=tuple(cfg["channels"]),
        strides=tuple(cfg["strides"]),
    )
    model = SparKPretrainer(
        encoder=encoder,
        encoder_out_channels=cfg["channels"][-1],
        patch_size=cfg["spark_patch_size"],
        mask_ratio=cfg["mask_ratio"],
    ).to(device)

    optimizer = optim.AdamW(model.parameters(),
                            lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg["epochs"], eta_min=1e-6)

    os.makedirs(cfg["output_dir"], exist_ok=True)
    best_loss = float("inf")

    for epoch in range(1, cfg["epochs"] + 1):
        model.train()
        epoch_loss = 0.0
        t0 = time.time()

        for batch in loader:
            imgs = batch["image"].to(device)
            optimizer.zero_grad()
            loss, _, _ = model(imgs)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item()

        scheduler.step()
        avg = epoch_loss / len(loader)
        elapsed = time.time() - t0
        print(f"Epoch {epoch:>4}/{cfg['epochs']} | loss={avg:.4f} | {elapsed:.0f}s")

        if avg < best_loss:
            best_loss = avg
            ckpt = os.path.join(cfg["output_dir"], "pretrained_encoder_best.pth")
            torch.save(model.encoder.state_dict(), ckpt)

        if epoch % cfg.get("save_every", 10) == 0:
            ckpt = os.path.join(cfg["output_dir"], f"pretrained_encoder_ep{epoch}.pth")
            torch.save(model.encoder.state_dict(), ckpt)

    print(f"Pretraining done. Best loss: {best_loss:.4f}")
    print(f"Encoder saved to {cfg['output_dir']}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/pretraining.yaml")
    args = ap.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    pretrain(cfg)
