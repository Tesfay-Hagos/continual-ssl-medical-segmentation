# %% [markdown]
# # SSL + Knowledge Distillation for Data-Efficient Medical Image Segmentation
#
# **Task:** Heart segmentation (Task02 — only 1 labeled training volume)
#
# **Method:**
# 1. SparK masked pretraining on unlabeled volumes (SSL)
# 2. Fine-tune pretrained encoder → Teacher model
# 3. Train Student with DiceCE + KD loss (soft predictions from Teacher)
#
# **Research Question:**
# With only 1 labeled volume, does SSL pretraining + Knowledge Distillation
# outperform random initialization?
#
# **Experiments:**
# | Run          | Init      | Loss            |
# |--------------|-----------|-----------------|
# | Baseline     | Random    | DiceCE          |
# | SSL only     | SparK     | DiceCE          |
# | SSL + KD     | SparK     | DiceCE + KL     |
#
# ---
# ### Before running on Kaggle
# 1. Add dataset input: `vivekprajapati2048/medical-segmentation-decathlon-heart`
# 2. Enable GPU (P100 or T4)
# 3. Add Kaggle secret: `WANDB_API_KEY`
# 4. Run all cells top-to-bottom

# %% [markdown]
# ## 0 — Environment setup

# %%
import os
import sys
import subprocess

ON_KAGGLE = os.path.exists("/kaggle/working")

REPO_URL = "https://github.com/Tesfay-Hagos/continual-ssl-medical-segmentation.git"
REPO_DIR = "/kaggle/working/project" if ON_KAGGLE else os.path.abspath(
               os.path.join(os.path.dirname(__file__), "..", ".."))
OUT_DIR  = "/kaggle/working/checkpoints" if ON_KAGGLE else "/tmp/ssl_kd_ckpts"

print(f"ON_KAGGLE : {ON_KAGGLE}")
print(f"REPO_DIR  : {REPO_DIR}")
print(f"OUT_DIR   : {OUT_DIR}")

# %%
if ON_KAGGLE:
    if not os.path.exists(REPO_DIR):
        result = subprocess.run(["git", "clone", REPO_URL, REPO_DIR],
                                capture_output=True, text=True)
    else:
        result = subprocess.run(["git", "-C", REPO_DIR, "pull"],
                                capture_output=True, text=True)
    print(result.stdout or result.stderr)
    subprocess.run(["find", REPO_DIR, "-type", "d", "-name", "__pycache__",
                    "-exec", "rm", "-rf", "{}", "+"], capture_output=True)
else:
    print(f"Using repo at: {REPO_DIR}")

# %%
if ON_KAGGLE:
    subprocess.run(
        [sys.executable, "-m", "pip", "install",
         "monai[all]", "nibabel", "scipy", "scikit-image", "pyyaml",
         "wandb", "--quiet"],
        check=True
    )
    print("Dependencies installed.")
else:
    print("Skipping pip install (local run).")

# %%
import importlib

SRC_DIR = os.path.join(REPO_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)
importlib.invalidate_caches()

import torch
import numpy as np

gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU only"
print(f"PyTorch  : {torch.__version__}")
print(f"Device   : {gpu_name}")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.makedirs(OUT_DIR, exist_ok=True)

# %%
import wandb

WANDB_PROJECT = "ssl-kd-heart"
try:
    if ON_KAGGLE:
        from kaggle_secrets import UserSecretsClient
        _key = UserSecretsClient().get_secret("SSL_KD_WANDB")
        wandb.login(key=_key)
    else:
        wandb.login()
    USE_WANDB = True
    print(f"WandB logged in. Project: {WANDB_PROJECT}")
except Exception as e:
    USE_WANDB = False
    print(f"WandB login failed ({e}) — running without logging.")

# %% [markdown]
# ## 1 — Dataset setup

# %%
from data.datasets import kaggle_task_roots, build_task_roots, verify_datasets

TASK_ROOTS = (kaggle_task_roots() if ON_KAGGLE
              else build_task_roots(os.environ.get("DATA_ROOT", "/data/decathlon")))

if not verify_datasets(TASK_ROOTS):
    raise RuntimeError(
        "Heart dataset missing. "
        "Add vivekprajapati2048/medical-segmentation-decathlon-heart as Kaggle input."
    )
print("Dataset OK.")
print(f"  Heart root: {TASK_ROOTS.get('heart', 'NOT FOUND')}")

# %% [markdown]
# ## 2 — SparK Pretraining (or restore from WandB)
#
# Pretrains the U-Net encoder on unlabeled volumes using sparse masked image modeling.
# If a checkpoint already exists it is restored automatically.

# %%
import json
import yaml
from pathlib import Path
from utils.storage import restore_checkpoint, save_checkpoint

PRETRAIN_DIR  = Path(os.path.join(OUT_DIR, "pretrain"))
PRETRAIN_CKPT = PRETRAIN_DIR / "best.pth"
PRETRAIN_DONE = PRETRAIN_DIR / "pretrain_done.json"
PRETRAIN_DIR.mkdir(parents=True, exist_ok=True)

# Try to restore completion marker + encoder from WandB
restore_checkpoint("pretrain_done.json", PRETRAIN_DIR,
                   "pretrain-done", "cssl-medical", "", "")
restore_checkpoint("best.pth", PRETRAIN_DIR,
                   "pretrain-encoder", "cssl-medical", "", "")

# Fallback: extract encoder weights from full training checkpoint
if not PRETRAIN_CKPT.exists():
    restore_checkpoint("latest.pth", PRETRAIN_DIR,
                       "pretrain-checkpoint", "cssl-medical", "", "")
    _latest = PRETRAIN_DIR / "latest.pth"
    if _latest.exists():
        _full = torch.load(_latest, map_location="cpu")["model"]
        _enc  = {k[len("encoder."):]: v
                 for k, v in _full.items() if k.startswith("encoder.")}
        torch.save(_enc, PRETRAIN_CKPT)
        print("Extracted encoder weights from latest.pth → best.pth")

if PRETRAIN_DONE.exists():
    _info = json.loads(PRETRAIN_DONE.read_text())
    print(f"Pretraining already complete "
          f"({_info['epochs_completed']} epochs, best_loss={_info['best_loss']:.5f})")
    print("Delete pretrain_done.json to re-run pretraining.")
else:
    from pretraining.pretrain import pretrain
    cfg_path = os.path.join(SRC_DIR, "configs", "pretraining.yaml")
    with open(cfg_path) as f:
        pretrain_cfg = yaml.safe_load(f)

    pretrain_cfg.update({
        "task_roots":    TASK_ROOTS,
        "output_dir":    str(PRETRAIN_DIR),
        "epochs":        100,
        "batch_size":    2,
        "num_workers":   2 if ON_KAGGLE else 0,
        "use_wandb":     USE_WANDB,
        "wandb_project": "cssl-medical",
        "wandb_run":     "spark-pretrain",
    })
    pretrain(pretrain_cfg)
    print(f"Pretraining complete. Encoder: {PRETRAIN_CKPT}")

# %% [markdown]
# ## 3 — Shared config and helpers

# %%
# Architecture — same as pretraining so encoder weights load cleanly
MODEL_CFG = {
    "channels": [32, 64, 128, 256, 512],
    "strides":  [2, 2, 2, 2],
}

TRAIN_CFG = {
    "task":            "heart",
    "task_roots":      TASK_ROOTS,
    "batch_size":      2,
    "num_workers":     2 if ON_KAGGLE else 0,
    "cache_rate":      1.0,   # heart has only 1 vol — cache it all
    "pin_memory":      False,
    "epochs":          30,
    "warmup_epochs":   3,
    "lr":              1.0e-4,
    "weight_decay":    1.0e-5,
    "patience":        15,
    # KD
    "kd_alpha":        1.0,   # weight of KD loss relative to DiceCE
    "kd_temperature":  2.0,
}

# %%
from models.unet import build_unet, UNetWithEncoder
from monai.losses import DiceCELoss
from monai.inferers import sliding_window_inference
from data.datasets import get_loaders


def build_model(pretrained: bool) -> UNetWithEncoder:
    unet  = build_unet(in_channels=1, out_channels=2,
                       channels=tuple(MODEL_CFG["channels"]),
                       strides=tuple(MODEL_CFG["strides"]))
    model = UNetWithEncoder(unet).to(DEVICE)
    if pretrained and PRETRAIN_CKPT.exists():
        model.load_pretrained_encoder(str(PRETRAIN_CKPT))
        print("  Loaded SparK pretrained encoder.")
    else:
        print("  Random initialization.")
    return model


def make_scheduler(optimizer, epochs, warmup):
    warmup_sched = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup)
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, epochs - warmup), eta_min=1e-6)
    return torch.optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=[warmup_sched, cosine], milestones=[warmup])


@torch.no_grad()
def evaluate(model, loader) -> dict:
    from evaluation.metrics import SegmentationEvaluator
    model.eval()
    ev = SegmentationEvaluator(num_classes=2)
    for batch in loader:
        img  = batch["image"].to(DEVICE)
        pred = sliding_window_inference(
            img, (96, 96, 96), sw_batch_size=2, predictor=model, overlap=0.25)
        ev.update(pred.cpu(), batch["label"].cpu())
    return ev.aggregate()


def log_wandb(metrics: dict):
    if USE_WANDB and wandb.run is not None:
        wandb.log(metrics)

# %% [markdown]
# ## 4 — Fine-tuning loop (shared by all three experiments)

# %%
import copy
import torch.nn.functional as F


def finetune(run_name: str, model, train_loader, val_loader,
             teacher=None) -> dict:
    """
    Fine-tune model on heart.

    Args:
        run_name:     WandB run name and checkpoint subfolder
        model:        UNetWithEncoder to train
        train_loader: training DataLoader
        val_loader:   validation DataLoader
        teacher:      frozen teacher model for KD loss (None = no KD)

    Returns dict with best DSC, HD95 and checkpoint path.
    """
    ckpt_dir = Path(OUT_DIR) / run_name
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    criterion = DiceCELoss(to_onehot_y=True, softmax=True)
    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=TRAIN_CFG["lr"],
                                  weight_decay=TRAIN_CFG["weight_decay"])
    scheduler  = make_scheduler(optimizer, TRAIN_CFG["epochs"],
                                 TRAIN_CFG["warmup_epochs"])
    scaler     = torch.amp.GradScaler(DEVICE.type,
                                       enabled=(DEVICE.type == "cuda"))

    best_dsc   = 0.0
    best_hd95  = float("inf")
    trigger    = 0
    patience   = TRAIN_CFG["patience"]
    alpha      = TRAIN_CFG["kd_alpha"]
    T_kd       = TRAIN_CFG["kd_temperature"]
    start_epoch = 0

    # Resume from latest.pth if interrupted
    resume_ckpt = ckpt_dir / "latest.pth"
    if resume_ckpt.exists():
        state = torch.load(resume_ckpt, map_location=DEVICE)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        start_epoch = state["epoch"] + 1
        best_dsc    = state["best_dsc"]
        best_hd95   = state["best_hd95"]
        trigger     = state["trigger"]
        print(f"  Resumed {run_name} from epoch {start_epoch}, best_dsc={best_dsc:.4f}")

    if teacher is not None:
        teacher.eval()
        for p in teacher.parameters():
            p.requires_grad = False
        print(f"  KD enabled (alpha={alpha}, T={T_kd})")

    for epoch in range(start_epoch, TRAIN_CFG["epochs"]):
        model.train()
        epoch_loss = 0.0
        n_batches  = 0

        for batch in train_loader:
            imgs   = batch["image"].to(DEVICE)
            labels = batch["label"].to(DEVICE)
            if labels.dim() == 4:
                labels = labels.unsqueeze(1)
            labels = labels.long()

            optimizer.zero_grad()
            with torch.amp.autocast(device_type=DEVICE.type):
                preds = model(imgs)
                loss  = criterion(preds, labels)

                if teacher is not None:
                    with torch.no_grad():
                        t_soft = F.softmax(teacher(imgs) / T_kd, dim=1)
                    s_log = F.log_softmax(preds / T_kd, dim=1)
                    kd_loss = F.kl_div(s_log, t_soft, reduction="mean") * (T_kd ** 2)
                    loss = loss + alpha * kd_loss

            if not torch.isfinite(loss):
                optimizer.zero_grad()
                continue

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            epoch_loss += loss.item()
            n_batches  += 1

        scheduler.step()
        avg_loss = epoch_loss / max(n_batches, 1)
        metrics  = evaluate(model, val_loader)
        dsc      = metrics["dice"]
        hd95     = metrics["hd95"]
        lr_now   = scheduler.get_last_lr()[0]

        print(f"  [{run_name}] Epoch {epoch+1:>3}/{TRAIN_CFG['epochs']} | "
              f"loss={avg_loss:.4f} | DSC={dsc:.4f} | HD95={hd95:.1f} | "
              f"best={best_dsc:.4f} | lr={lr_now:.2e}")

        log_wandb({f"{run_name}/loss": avg_loss,
                   f"{run_name}/dsc":  dsc,
                   f"{run_name}/hd95": hd95,
                   f"{run_name}/epoch": epoch + 1})

        if dsc >= best_dsc:
            best_dsc  = dsc
            best_hd95 = hd95
            torch.save(model.state_dict(), ckpt_dir / "best.pth")
            trigger = 0
        else:
            trigger += 1
            if trigger >= patience:
                print(f"  Early stopping at epoch {epoch+1}.")
                break

        # Save resume checkpoint every epoch
        torch.save({"model":     model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "epoch":     epoch,
                    "best_dsc":  best_dsc,
                    "best_hd95": best_hd95,
                    "trigger":   trigger}, resume_ckpt)

    log_wandb({f"{run_name}/best_dsc": best_dsc,
               f"{run_name}/best_hd95": best_hd95})
    print(f"  [{run_name}] Done. Best DSC={best_dsc:.4f}  HD95={best_hd95:.1f}")
    return {"run": run_name, "best_dsc": best_dsc, "best_hd95": best_hd95,
            "ckpt": str(ckpt_dir / "best.pth")}

# %% [markdown]
# ## 5 — Run experiments

# %%
# Build data loaders once — shared across all three runs
train_loader, val_loader = get_loaders(
    TASK_ROOTS, "heart",
    batch_size=TRAIN_CFG["batch_size"],
    num_workers=TRAIN_CFG["num_workers"],
    cache_rate=TRAIN_CFG["cache_rate"],
    pin_memory=TRAIN_CFG["pin_memory"])

print(f"Heart loader — train batches: {len(train_loader)}, "
      f"val batches: {len(val_loader)}")

# Restore results from previous run if the cell is re-executed after a crash
_results_path = Path(OUT_DIR) / "ssl_kd_results.json"
results = {}
if _results_path.exists():
    results = json.loads(_results_path.read_text())
    print(f"Restored previous results: {list(results.keys())}")

def _save_results():
    _results_path.write_text(json.dumps(results, indent=2))

# %%
# --- Experiment A: Baseline (random init, DiceCE only) ---
if "baseline" in results:
    print("Experiment A already done — skipping.")
else:
    if USE_WANDB:
        wandb.init(project=WANDB_PROJECT, name="baseline", reinit=True,
                   config={**MODEL_CFG, **TRAIN_CFG, "use_pretrained": False, "kd": False})
    try:
        print("\n=== Experiment A: Baseline (random init) ===")
        model_baseline = build_model(pretrained=False)
        results["baseline"] = finetune("baseline", model_baseline,
                                       train_loader, val_loader, teacher=None)
        _save_results()
    finally:
        if USE_WANDB:
            wandb.finish()

# %%
# --- Experiment B: SSL only (SparK pretrained, DiceCE only) ---
if "ssl_only" in results:
    print("Experiment B already done — skipping.")
else:
    if USE_WANDB:
        wandb.init(project=WANDB_PROJECT, name="ssl_only", reinit=True,
                   config={**MODEL_CFG, **TRAIN_CFG, "use_pretrained": True, "kd": False})
    try:
        print("\n=== Experiment B: SSL only (SparK pretrained, no KD) ===")
        model_ssl = build_model(pretrained=True)
        results["ssl_only"] = finetune("ssl_only", model_ssl,
                                       train_loader, val_loader, teacher=None)
        _save_results()
    finally:
        if USE_WANDB:
            wandb.finish()

# %%
# --- Experiment C: SSL + KD ---
# Teacher = SSL-only model (Experiment B). Student = fresh pretrained encoder.
if "ssl_kd" in results:
    print("Experiment C already done — skipping.")
else:
    if "ssl_only" not in results:
        raise RuntimeError(
            "Experiment B (ssl_only) must complete before running SSL+KD. "
            "Re-run the ssl_only cell first."
        )
    if USE_WANDB:
        wandb.init(project=WANDB_PROJECT, name="ssl_kd", reinit=True,
                   config={**MODEL_CFG, **TRAIN_CFG, "use_pretrained": True, "kd": True})
    try:
        print("\n=== Experiment C: SSL + KD ===")
        teacher = build_model(pretrained=True)
        teacher.load_state_dict(
            torch.load(results["ssl_only"]["ckpt"], map_location=DEVICE))
        teacher.eval()

        model_kd = build_model(pretrained=True)
        results["ssl_kd"] = finetune("ssl_kd", model_kd,
                                     train_loader, val_loader, teacher=teacher)
        _save_results()
    finally:
        if USE_WANDB:
            wandb.finish()

# %% [markdown]
# ## 6 — Results

# %%
print("\n" + "=" * 55)
print(f"  Heart segmentation — Task02 (1 labeled training volume)")
print("=" * 55)
print(f"  {'Method':<22} {'DSC':>8} {'HD95':>10}")
print("-" * 55)

run_labels = {
    "baseline": "Baseline (random init)",
    "ssl_only": "SSL only (SparK)",
    "ssl_kd":   "SSL + KD",
}
for key, label in run_labels.items():
    r = results.get(key, {})
    dsc  = r.get("best_dsc",  float("nan"))
    hd95 = r.get("best_hd95", float("nan"))
    print(f"  {label:<22} {dsc:>8.4f} {hd95:>10.2f}")

print("=" * 55)

# Compute gains
if "baseline" in results and "ssl_only" in results:
    gain_ssl = results["ssl_only"]["best_dsc"] - results["baseline"]["best_dsc"]
    print(f"\n  SSL gain over baseline :  {gain_ssl:+.4f} DSC")
if "ssl_only" in results and "ssl_kd" in results:
    gain_kd = results["ssl_kd"]["best_dsc"] - results["ssl_only"]["best_dsc"]
    print(f"  KD  gain over SSL-only :  {gain_kd:+.4f} DSC")

# %%
# Save results JSON
results_path = Path(OUT_DIR) / "ssl_kd_results.json"
results_path.write_text(json.dumps(results, indent=2))
print(f"\nResults saved to {results_path}")

if USE_WANDB:
    save_checkpoint(results_path, "ssl-kd-results", "", "")
    print("Results uploaded to WandB artifact: ssl-kd-results")
