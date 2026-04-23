"""
Continual learning training script.
Supports EWC, LwF, and Experience Replay strategies.

Run:
    python scripts/train_continual.py --config configs/ewc.yaml
    python scripts/train_continual.py --config configs/lwf.yaml
    python scripts/train_continual.py --config configs/replay.yaml
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from monai.losses import DiceCELoss

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from monai.inferers import sliding_window_inference

from data.datasets import get_loaders
from evaluation.metrics import (SegmentationEvaluator, print_cl_metrics,
                                 backward_transfer, forgetting_measure,
                                 average_accuracy)
from models.unet import build_unet, UNetWithEncoder
from utils.storage import save_checkpoint, restore_checkpoint

try:
    import wandb
    _WANDB = True
except ImportError:
    _WANDB = False


_LATEST_PTH = "latest.pth"


# ── Helpers ───────────────────────────────────────────────────────────────────

def _log(cfg: dict, log_dict: dict):
    if _WANDB and cfg.get("use_wandb", True):
        wandb.log(log_dict)


def _make_scheduler(optimizer, n_epochs: int, warmup_epochs: int):
    """Linear warmup then cosine annealing."""
    warmup = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_epochs)
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, n_epochs - warmup_epochs), eta_min=1e-6)
    return torch.optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=[warmup, cosine], milestones=[warmup_epochs])


def load_model(cfg: dict, device: torch.device) -> UNetWithEncoder:
    unet  = build_unet(in_channels=1, out_channels=2,
                       channels=tuple(cfg["channels"]),
                       strides=tuple(cfg["strides"]))
    model = UNetWithEncoder(unet).to(device)
    ckpt  = cfg.get("pretrained_ckpt")
    if cfg.get("use_pretrained") and ckpt and os.path.exists(ckpt):
        model.load_pretrained_encoder(ckpt)
    else:
        print("Training from random initialization (no pretrained encoder)")
    return model


def _setup_strategy(cfg: dict, model, device):
    strategy = cfg.get("strategy", "none")
    cl_reg, replay_buf = None, None
    if strategy == "ewc":
        from continual.ewc import EWC
        cl_reg = EWC(model, lambda_=cfg.get("ewc_lambda", 1000.0))
    elif strategy == "lwf":
        from continual.lwf import LwF
        cl_reg = LwF(alpha=cfg.get("lwf_alpha", 1.0),
                     temperature=cfg.get("lwf_temperature", 2.0))
    elif strategy == "replay":
        from continual.replay import ReplayBuffer
        replay_buf = ReplayBuffer(capacity=cfg.get("buffer_capacity", 200),
                                  device=device)
    return cl_reg, replay_buf


def _resume(ckpt_dir: Path, model, optimizer, scheduler, scaler,
            artifact_name: str, project: str,
            gdrive_folder: str, gdrive_creds: str):
    restore_checkpoint(_LATEST_PTH, ckpt_dir, artifact_name,
                       project, gdrive_folder, gdrive_creds)
    path = ckpt_dir / _LATEST_PTH
    if not path.exists():
        return 0, 0.0, float("inf"), None
    state = torch.load(path, map_location="cpu")
    model.load_state_dict(state["model"])
    optimizer.load_state_dict(state["optimizer"])
    scheduler.load_state_dict(state["scheduler"])
    if "scaler" in state:
        scaler.load_state_dict(state["scaler"])
    return (state["epoch"], state.get("best_val_dsc", 0.0),
            state.get("best_val_loss", float("inf")), state.get("wandb_run_id"))


def _save(ckpt_dir: Path, epoch: int, model, optimizer, scheduler, scaler,
          best_val_dsc: float, best_val_loss: float,
          artifact_name: str, gdrive_folder: str, gdrive_creds: str):
    run_id = wandb.run.id if (_WANDB and wandb.run is not None) else None
    path = ckpt_dir / _LATEST_PTH
    torch.save({"epoch": epoch, "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "scaler":    scaler.state_dict(),
                "best_val_dsc": best_val_dsc,
                "best_val_loss": best_val_loss,
                "wandb_run_id": run_id}, path)
    save_checkpoint(path, artifact_name, gdrive_folder, gdrive_creds)


def train_one_epoch(model, loader, optimizer, scaler, criterion,
                    cl_reg, replay_buf, cfg: dict, device) -> float:
    model.train()
    total_loss = 0.0
    strategy   = cfg.get("strategy", "none")

    for batch in loader:
        imgs   = batch["image"].to(device)
        labels = batch["label"].long().to(device)
        optimizer.zero_grad()

        with torch.amp.autocast(device_type=device.type):
            loss = criterion(model(imgs), labels)

            if strategy == "ewc" and cl_reg is not None:
                loss = loss + cl_reg.penalty(model)
            elif strategy == "lwf" and cl_reg is not None:
                loss = loss + cl_reg.distillation_loss(model, imgs)

            if replay_buf is not None and len(replay_buf) > 0:
                r_imgs, r_lbl = replay_buf.sample(cfg.get("replay_batch_size", 2))
                if r_imgs is not None:
                    r_loss = criterion(model(r_imgs.to(device)),
                                       r_lbl.long().to(device))
                    loss = loss + r_loss

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item()

    return total_loss / max(len(loader), 1)


_ROI = (96, 96, 96)   # must match training patch size

@torch.no_grad()
def evaluate(model, loader, device) -> dict:
    """
    Sliding-window inference over full volumes.
    Calling model() directly on a 512x512x400 CT would OOM and give wrong
    InstanceNorm statistics. sliding_window_inference tiles the volume into
    overlapping 96^3 windows matching the training distribution, then blends.
    """
    model.eval()
    ev = SegmentationEvaluator(num_classes=2)
    for batch in loader:
        img  = batch["image"].to(device)
        pred = sliding_window_inference(
            img, _ROI, sw_batch_size=2, predictor=model, overlap=0.25)
        ev.update(pred, batch["label"].to(device))
    return ev.aggregate()


def _train_task(model, task_name: str, t: int, cfg: dict, criterion,
                cl_reg, replay_buf, device, out_dir: Path, val_loaders: dict):
    ckpt_dir = out_dir / task_name
    ckpt_dir.mkdir(exist_ok=True)

    train_loader, val_loader = get_loaders(
        cfg["task_roots"], task_name,
        batch_size=cfg["batch_size"],
        num_workers=cfg["num_workers"],
        cache_rate=cfg.get("cache_rate", 0.1),
    )
    val_loaders[task_name] = val_loader

    n_epochs      = cfg["epochs_per_task"]
    warmup_epochs = cfg.get("warmup_epochs", 2)
    base_lr       = cfg["lr"]

    # Differential LR: pretrained encoder gets 10× lower LR to protect
    # learned representations; decoder adapts at full LR.
    if cfg.get("use_pretrained"):
        param_groups = model.parameter_groups(
            base_lr, encoder_lr_scale=cfg.get("encoder_lr_scale", 0.1))
    else:
        param_groups = model.parameters()

    optimizer = torch.optim.AdamW(param_groups,
                                  lr=base_lr, weight_decay=cfg["weight_decay"])
    scheduler = _make_scheduler(optimizer, n_epochs, warmup_epochs)
    scaler    = torch.amp.GradScaler(device.type, enabled=(device.type == "cuda"))

    artifact_name = f"cl-{cfg.get('strategy','none')}-{task_name}"
    gdrive_folder = cfg.get("gdrive_folder_id", "")
    gdrive_creds  = cfg.get("gdrive_credentials", "")

    start_epoch, best_val_dsc, best_val_loss, _ = _resume(
        ckpt_dir, model, optimizer, scheduler, scaler,
        artifact_name, cfg.get("wandb_project", "cssl-medical"),
        gdrive_folder, gdrive_creds)
    if start_epoch:
        print(f"  Resumed {task_name} from epoch {start_epoch}, "
              f"best_dsc={best_val_dsc:.4f}")

    save_every = cfg.get("save_every_n_epochs", 10)
    patience   = cfg.get("patience", 10)
    trigger    = 0

    for epoch in range(start_epoch, n_epochs):
        train_loss  = train_one_epoch(model, train_loader, optimizer, scaler,
                                      criterion, cl_reg, replay_buf, cfg, device)
        scheduler.step()
        val_metrics = evaluate(model, val_loader, device)
        val_dsc     = val_metrics["dice"]
        val_hd95    = val_metrics["hd95"]

        print(f"  Epoch {epoch+1:>3}/{n_epochs} | loss={train_loss:.5f} | "
              f"DSC={val_dsc:.4f} | HD95={val_hd95:.1f} | best={best_val_dsc:.4f}")

        _log(cfg, {f"{task_name}/train_loss": train_loss,
                   f"{task_name}/val_dsc":    val_dsc,
                   f"{task_name}/val_hd95":   val_hd95,
                   "global_epoch":            t * n_epochs + epoch + 1})

        if (epoch + 1) % save_every == 0 or epoch == n_epochs - 1:
            torch.save(model.state_dict(), ckpt_dir / f"epoch-{epoch+1}.pth")

        if val_dsc >= best_val_dsc:
            torch.save(model.state_dict(), ckpt_dir / "best.pth")
            best_val_dsc = val_dsc

        _save(ckpt_dir, epoch + 1, model, optimizer, scheduler, scaler,
              best_val_dsc, best_val_loss,
              artifact_name, gdrive_folder, gdrive_creds)

        if val_dsc > best_val_loss:
            best_val_loss = val_dsc
            trigger = 0
        else:
            trigger += 1
            if trigger == patience:
                print(f"  Val DSC did not improve for {patience} epochs. "
                      f"Early stopping.")
                break

    return train_loader, val_loader


def _post_task(strategy: str, cl_reg, replay_buf, model,
               val_loader, train_loader, t: int, cfg: dict, device):
    if strategy == "ewc" and cl_reg is not None:
        cl_reg.register_task(model, val_loader, device,
                              num_batches=cfg.get("fisher_batches", 100))
    elif strategy == "lwf" and cl_reg is not None:
        cl_reg.register_task(model)
    elif strategy == "replay" and replay_buf is not None:
        replay_buf.populate_from_loader(train_loader, task_id=t)


def _eval_all_tasks(model, tasks: list, t: int, val_loaders: dict,
                    R: np.ndarray, cfg: dict, device):
    for i, past_task in enumerate(tasks[:t + 1]):
        m = evaluate(model, val_loaders[past_task], device)
        R[t, i] = m["dice"]
        print(f"  Eval {past_task:<10}: DSC={m['dice']:.4f}  HD95={m['hd95']:.1f}")
        _log(cfg, {f"eval/{past_task}_dsc_after_task{t+1}":  m["dice"],
                   f"eval/{past_task}_hd95_after_task{t+1}": m["hd95"]})


# ── Main entry ────────────────────────────────────────────────────────────────

def run(cfg: dict):
    device   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    strategy = cfg.get("strategy", "none")
    run_name = cfg.get("wandb_run",
                       f"{strategy}_{'ssl' if cfg.get('use_pretrained') else 'no_ssl'}")

    print(f"\nStrategy: {strategy.upper()}  |  Device: {device}")
    print(f"Pretrained: {cfg.get('use_pretrained', False)}  |  "
          f"Tasks: {cfg['task_order']}\n")

    out_dir = Path(cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    if _WANDB and cfg.get("use_wandb", True):
        run_id_file = out_dir / "wandb_run_id.json"
        saved_run_id = None
        if run_id_file.exists():
            saved_run_id = json.load(open(run_id_file)).get("run_id")
        init_kwargs = dict(
            project=cfg.get("wandb_project", "cssl-medical"),
            config={k: v for k, v in cfg.items() if k != "task_roots"},
        )
        if saved_run_id:
            init_kwargs["id"]     = saved_run_id
            init_kwargs["resume"] = "must"
        else:
            init_kwargs["name"] = run_name
        wandb.init(**init_kwargs)
        run_id_file.write_text(json.dumps({"run_id": wandb.run.id}))

    tasks     = cfg["task_order"]
    R         = np.zeros((len(tasks), len(tasks)))
    model     = load_model(cfg, device)
    criterion = DiceCELoss(to_onehot_y=True, softmax=True)
    cl_reg, replay_buf = _setup_strategy(cfg, model, device)
    val_loaders = {}

    for t, task_name in enumerate(tasks):
        print(f"\n{'='*60}\nTask {t+1}/{len(tasks)}: {task_name.upper()}\n{'='*60}")
        train_loader, val_loader = _train_task(
            model, task_name, t, cfg, criterion, cl_reg, replay_buf,
            device, out_dir, val_loaders)
        _post_task(strategy, cl_reg, replay_buf, model,
                   val_loader, train_loader, t, cfg, device)
        _eval_all_tasks(model, tasks, t, val_loaders, R, cfg, device)

    print_cl_metrics(R, tasks)

    with open(out_dir / "cl_results.json", "w") as f:
        json.dump({"strategy": strategy,
                   "use_pretrained": cfg.get("use_pretrained", False),
                   "task_order": tasks, "R_matrix": R.tolist()}, f, indent=2)
    print(f"\nResults saved to {out_dir / 'cl_results.json'}")

    if _WANDB and cfg.get("use_wandb", True):
        T = len(tasks)
        wandb.log({"summary/AA":  average_accuracy(R),
                   "summary/BWT": backward_transfer(R) if T > 1 else float("nan"),
                   "summary/FM":  forgetting_measure(R) if T > 1 else float("nan")})
        wandb.finish()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--strategy", default=None,
                    choices=["ewc", "lwf", "replay", "none"])
    ap.add_argument("--no_pretrained", action="store_true")
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    if args.strategy:
        cfg["strategy"] = args.strategy
    if args.no_pretrained:
        cfg["use_pretrained"] = False
    if "strategy" not in cfg:
        cfg["strategy"] = os.path.splitext(os.path.basename(args.config))[0]

    run(cfg)
