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

import numpy as np
import torch
import torch.nn as nn
import yaml
from monai.losses import DiceCELoss

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.datasets import get_loaders, TASK_ORDER
from evaluation.metrics import SegmentationEvaluator, print_cl_metrics
from models.unet import build_unet, UNetWithEncoder


def load_model(cfg: dict, device: torch.device) -> UNetWithEncoder:
    unet  = build_unet(in_channels=1, out_channels=2,
                       channels=tuple(cfg["channels"]),
                       strides=tuple(cfg["strides"]))
    model = UNetWithEncoder(unet).to(device)

    ckpt = cfg.get("pretrained_ckpt")
    if cfg.get("use_pretrained") and ckpt and os.path.exists(ckpt):
        model.load_pretrained_encoder(ckpt)
    else:
        print("Training from random initialization (no pretrained encoder)")
    return model


def train_one_epoch(model, loader, optimizer, criterion,
                    cl_regularizer, replay_buffer, cfg, device):
    model.train()
    total_loss = 0.0

    for batch in loader:
        imgs   = batch["image"].to(device)
        labels = batch["label"].long().squeeze(1).to(device)

        optimizer.zero_grad()
        pred  = model(imgs)
        loss  = criterion(pred, labels)

        # CL regularization
        strategy = cfg.get("strategy", "none")
        if strategy == "ewc" and cl_regularizer is not None:
            loss = loss + cl_regularizer.penalty(model)
        elif strategy == "lwf" and cl_regularizer is not None:
            loss = loss + cl_regularizer.distillation_loss(model, imgs)

        # Replay
        if replay_buffer is not None and len(replay_buffer) > 0:
            r_imgs, r_labels = replay_buffer.sample(cfg.get("replay_batch_size", 2))
            if r_imgs is not None:
                r_imgs   = r_imgs.to(device)
                r_labels = r_labels.long().squeeze(1).to(device)
                loss = loss + criterion(model(r_imgs), r_labels)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()

    return total_loss / max(len(loader), 1)


@torch.no_grad()
def evaluate(model, loader, device) -> dict:
    model.eval()
    evaluator = SegmentationEvaluator(num_classes=2)
    for batch in loader:
        imgs   = batch["image"].to(device)
        labels = batch["label"].to(device)
        pred   = model(imgs)
        evaluator.update(pred, labels)
    return evaluator.aggregate()


def run(cfg: dict):
    device   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    strategy = cfg.get("strategy", "none")
    print(f"\nStrategy: {strategy.upper()}  |  Device: {device}")
    print(f"Pretrained: {cfg.get('use_pretrained', False)}  |  "
          f"Tasks: {cfg['task_order']}\n")

    os.makedirs(cfg["output_dir"], exist_ok=True)
    tasks     = cfg["task_order"]
    T         = len(tasks)
    R         = np.zeros((T, T))   # R[t][i] = DSC on task i after training t

    model     = load_model(cfg, device)
    criterion = DiceCELoss(to_onehot_y=True, softmax=True)

    # Strategy objects
    cl_reg      = None
    replay_buf  = None

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

    # Keep all val loaders to re-evaluate on previous tasks
    val_loaders = {}

    for t, task_name in enumerate(tasks):
        print(f"\n{'='*60}")
        print(f"Task {t+1}/{T}: {task_name.upper()}")
        print(f"{'='*60}")

        train_loader, val_loader = get_loaders(
            cfg["data_root"], task_name,
            batch_size=cfg["batch_size"],
            num_workers=cfg["num_workers"],
            cache_rate=cfg.get("cache_rate", 0.1),
        )
        val_loaders[task_name] = val_loader

        optimizer = torch.optim.AdamW(model.parameters(),
                                      lr=cfg["lr"],
                                      weight_decay=cfg["weight_decay"])
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=cfg["epochs_per_task"], eta_min=1e-6)

        best_dice = 0.0
        for epoch in range(1, cfg["epochs_per_task"] + 1):
            cfg_copy = {**cfg, "strategy": strategy}
            train_loss = train_one_epoch(model, train_loader, optimizer,
                                         criterion, cl_reg, replay_buf,
                                         cfg_copy, device)
            scheduler.step()

            if epoch % 10 == 0 or epoch == cfg["epochs_per_task"]:
                val_metrics = evaluate(model, val_loader, device)
                print(f"  Epoch {epoch:>3} | loss={train_loss:.4f} "
                      f"| DSC={val_metrics['dice']:.4f} "
                      f"| HD95={val_metrics['hd95']:.1f}")
                if val_metrics["dice"] > best_dice:
                    best_dice = val_metrics["dice"]
                    ckpt = os.path.join(cfg["output_dir"],
                                        f"best_{task_name}.pth")
                    torch.save(model.state_dict(), ckpt)

        # ── Post-task: register CL state ──
        if strategy == "ewc" and cl_reg is not None:
            cl_reg.register_task(model, val_loader, device,
                                  num_batches=cfg.get("fisher_batches", 100))
        elif strategy == "lwf" and cl_reg is not None:
            cl_reg.register_task(model)
        elif strategy == "replay" and replay_buf is not None:
            replay_buf.populate_from_loader(train_loader, task_id=t)

        # ── Evaluate on ALL tasks seen so far ──
        for i, past_task in enumerate(tasks[:t + 1]):
            metrics = evaluate(model, val_loaders[past_task], device)
            R[t, i] = metrics["dice"]
            print(f"  Eval {past_task:<10}: DSC={metrics['dice']:.4f}  "
                  f"HD95={metrics['hd95']:.1f}")

    # ── Final report ──
    print_cl_metrics(R, tasks)

    # Save results
    result_path = os.path.join(cfg["output_dir"], "cl_results.json")
    with open(result_path, "w") as f:
        json.dump({"strategy": strategy,
                   "use_pretrained": cfg.get("use_pretrained", False),
                   "task_order": tasks,
                   "R_matrix": R.tolist()}, f, indent=2)
    print(f"\nResults saved to {result_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--data_root", default=None)
    ap.add_argument("--strategy", default=None,
                    choices=["ewc", "lwf", "replay", "none"])
    ap.add_argument("--no_pretrained", action="store_true")
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    # CLI overrides
    if args.data_root:
        cfg["data_root"] = args.data_root
    if args.strategy:
        cfg["strategy"] = args.strategy
    if args.no_pretrained:
        cfg["use_pretrained"] = False

    # Infer strategy from config filename if not set
    if "strategy" not in cfg:
        cfg["strategy"] = os.path.splitext(os.path.basename(args.config))[0]

    run(cfg)
