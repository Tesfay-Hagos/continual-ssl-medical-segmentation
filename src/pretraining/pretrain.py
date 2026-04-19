"""
SparK pretraining loop for the U-Net encoder.

Run:
    python scripts/pretrain.py --config configs/pretraining.yaml
"""

import argparse
import time
from pathlib import Path

import torch
import torch.optim as optim
import yaml
from monai.data import CacheDataset
from monai.transforms import (Compose, EnsureChannelFirstd, LoadImaged,
                               NormalizeIntensityd, Orientationd,
                               RandFlipd, RandSpatialCropd, Spacingd, ToTensord)
from torch.utils.data import DataLoader

from data.datasets import get_unlabelled_files
from models.unet import build_unet
from pretraining.spark import SparKPretrainer

try:
    import wandb
    _WANDB = True
except ImportError:
    _WANDB = False


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


def _build_model(cfg: dict, device: torch.device):
    encoder = build_unet(in_channels=1, out_channels=2,
                         channels=tuple(cfg["channels"]),
                         strides=tuple(cfg["strides"]))
    return SparKPretrainer(encoder=encoder,
                           encoder_out_channels=cfg["channels"][-1],
                           patch_size=cfg["spark_patch_size"],
                           mask_ratio=cfg["mask_ratio"]).to(device)


def _resume(out_dir: Path, model, optimizer, scheduler):
    path = out_dir / "latest.pth"
    if not path.exists():
        return 0, float("inf"), float("inf")
    state = torch.load(path, map_location="cpu")
    model.load_state_dict(state["model"])
    optimizer.load_state_dict(state["optimizer"])
    scheduler.load_state_dict(state["scheduler"])
    return state["epoch"], state["best_loss"], state.get("best_loss_ema", float("inf"))


def _save_state(out_dir: Path, epoch: int, model, optimizer, scheduler,
                best_loss: float, best_loss_ema: float):
    torch.save({"epoch": epoch, "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "best_loss": best_loss,
                "best_loss_ema": best_loss_ema}, out_dir / "latest.pth")


def _run_epoch(model, loader, optimizer, device) -> float:
    model.train()
    total = 0.0
    for batch in loader:
        optimizer.zero_grad()
        loss, _, _ = model(batch["image"].to(device))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total += loss.item()
    return total / len(loader)


def _update_checkpoints(out_dir: Path, epoch: int, avg: float,
                         best_loss: float, save_every: int,
                         model, n_epochs: int) -> float:
    if (epoch + 1) % save_every == 0 or epoch == n_epochs - 1:
        torch.save(model.encoder.state_dict(), out_dir / f"epoch-{epoch+1}.pth")
    if avg < best_loss:
        torch.save(model.encoder.state_dict(), out_dir / "best.pth")
        return avg
    return best_loss


def _early_stop_step(avg: float, best_ema: float,
                     trigger: int, patience: int):
    if avg < best_ema:
        return avg, 0, False
    trigger += 1
    stop = trigger >= patience
    return best_ema, trigger, stop


# ── Main entry ────────────────────────────────────────────────────────────────

def pretrain(cfg: dict):
    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Device: {device}")

    if _WANDB and cfg.get("use_wandb", True):
        wandb.init(project=cfg.get("wandb_project", "cssl-medical"),
                   name=cfg.get("wandb_run", "spark-pretrain"),
                   config={k: v for k, v in cfg.items() if k != "task_roots"},
                   reinit=True)

    files  = get_unlabelled_files(cfg["task_roots"])
    ds     = CacheDataset(files, transform=get_ssl_transforms(cfg["patch_size"]),
                          cache_rate=cfg.get("cache_rate", 0.05))
    loader = DataLoader(ds, batch_size=cfg["batch_size"], shuffle=True,
                        num_workers=cfg["num_workers"], pin_memory=True)
    print(f"Pretraining on {len(files)} volumes, {len(loader)} batches/epoch")

    model     = _build_model(cfg, device)
    optimizer = optim.AdamW(model.parameters(),
                            lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg["epochs"], eta_min=1e-6)

    start_epoch, best_loss, best_ema = _resume(out_dir, model, optimizer, scheduler)
    if start_epoch:
        print(f"Resumed from epoch {start_epoch}, best_loss={best_loss:.5f}")

    save_every = cfg.get("save_every", 10)
    patience   = cfg.get("patience", 15)
    trigger    = 0
    n_epochs   = cfg["epochs"]

    for epoch in range(start_epoch, n_epochs):
        t0  = time.time()
        avg = _run_epoch(model, loader, optimizer, device)
        scheduler.step()

        print(f"Epoch {epoch+1:>4}/{n_epochs} | loss={avg:.5f} | "
              f"best={best_loss:.5f} | {time.time()-t0:.0f}s")

        if _WANDB and cfg.get("use_wandb", True):
            wandb.log({"pretrain/loss": avg,
                       "pretrain/lr":   scheduler.get_last_lr()[0],
                       "epoch":         epoch + 1})

        best_loss = _update_checkpoints(out_dir, epoch, avg, best_loss,
                                         save_every, model, n_epochs)
        _save_state(out_dir, epoch + 1, model, optimizer, scheduler, best_loss, best_ema)

        best_ema, trigger, stop = _early_stop_step(avg, best_ema, trigger, patience)
        if stop:
            print(f"Loss did not improve for {patience} epochs. Early stopping.")
            break

    print(f"Pretraining done. Best loss: {best_loss:.5f}\nEncoder saved to {out_dir}")
    if _WANDB and cfg.get("use_wandb", True):
        wandb.finish()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/pretraining.yaml")
    args = ap.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    pretrain(cfg)
