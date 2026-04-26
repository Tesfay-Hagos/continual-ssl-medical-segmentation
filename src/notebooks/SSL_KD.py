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
# Reproducibility seed — set before any model/data construction
import random
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark     = False

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
    "cache_rate":      1.0,
    "pin_memory":      False,
    "epochs":          300,
    "warmup_epochs":   10,
    "lr":              1.0e-4,
    "weight_decay":    1.0e-5,
    "patience":        50,
    # KD
    "kd_alpha":        1.0,   # weight of KD loss relative to DiceCE
    "kd_temperature":  2.0,
}

# %%
from models.unet import build_unet, UNetWithEncoder
from monai.losses import DiceCELoss
from monai.inferers import sliding_window_inference
from monai.data import CacheDataset
from torch.utils.data import DataLoader
from data.datasets import get_loaders, get_file_list, get_transforms
from sklearn.model_selection import KFold


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


@torch.inference_mode()
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


def get_all_label_loaders(task_roots, task_name, batch_size, num_workers):
    """Load all labeled training volumes — used for the supervised upper bound."""
    train_files, val_files = get_file_list(task_roots, task_name)
    train_ds = CacheDataset(train_files,
                            transform=get_transforms(task_name, train=True),
                            cache_rate=1.0)
    val_ds   = CacheDataset(val_files,
                            transform=get_transforms(task_name, train=False),
                            cache_rate=1.0)
    return (DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                       num_workers=num_workers, pin_memory=False),
            DataLoader(val_ds,   batch_size=1,          shuffle=False,
                       num_workers=num_workers, pin_memory=False))

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
    t_kd       = TRAIN_CFG["kd_temperature"]
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
        print(f"  KD enabled (alpha={alpha}, T={t_kd})")

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
                with torch.inference_mode():
                    t_soft = F.softmax(teacher(imgs).float() / t_kd, dim=1)
                s_log    = F.log_softmax(preds.float() / t_kd, dim=1)
                # Per-voxel scaling: same normalization as DiceCELoss
                n_voxels = imgs.shape[2] * imgs.shape[3] * imgs.shape[4]  # D*H*W
                kd_loss  = F.kl_div(s_log, t_soft, reduction="sum") / (imgs.shape[0] * n_voxels)
                kd_loss  = kd_loss * (t_kd ** 2)
                loss     = loss + alpha * kd_loss

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
# ## 5 — Cross-validation setup and experiments
#
# 3-fold CV: each fold uses 13 train / 7 val volumes.
# Reports mean ± std across folds for statistical validity.

# %%
from sklearn.model_selection import KFold

# Get all heart files for CV splitting
train_files, val_files = get_file_list(TASK_ROOTS, "heart")
all_files = train_files + val_files  # 20 volumes total

# 3-fold CV setup
N_FOLDS = 3
kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
fold_splits = list(kf.split(all_files))

print(f"3-fold CV on {len(all_files)} heart volumes:")
for fold, (train_idx, val_idx) in enumerate(fold_splits):
    print(f"  Fold {fold+1}: {len(train_idx)} train, {len(val_idx)} val")

# Restore CV results from previous run
_cv_results_path = Path(OUT_DIR) / "ssl_kd_cv_results.json"
cv_results = {}
if _cv_results_path.exists():
    cv_results = json.loads(_cv_results_path.read_text())
    print(f"Restored CV results: {list(cv_results.keys())}")

def _save_cv_results():
    _cv_results_path.write_text(json.dumps(cv_results, indent=2))

def get_fold_loaders(fold_idx, use_all_train=False):
    """Get train/val loaders for a specific fold.
    
    Args:
        fold_idx: which fold (0, 1, 2)
        use_all_train: if True, use all training files (for upper bound)
                      if False, use only 1 file (for few-shot experiments)
    """
    train_idx, val_idx = fold_splits[fold_idx]
    
    fold_train_files = [all_files[i] for i in train_idx]
    fold_val_files   = [all_files[i] for i in val_idx]
    
    if not use_all_train:
        # Few-shot: use only 1 training volume
        fold_train_files = fold_train_files[:1]
    
    train_ds = CacheDataset(fold_train_files,
                            transform=get_transforms("heart", train=True),
                            cache_rate=1.0)
    val_ds   = CacheDataset(fold_val_files,
                            transform=get_transforms("heart", train=False),
                            cache_rate=1.0)
    
    return (DataLoader(train_ds, batch_size=TRAIN_CFG["batch_size"], shuffle=True,
                       num_workers=TRAIN_CFG["num_workers"], pin_memory=False),
            DataLoader(val_ds,   batch_size=1, shuffle=False,
                       num_workers=TRAIN_CFG["num_workers"], pin_memory=False))

# %%
# Run all experiments across 3 folds
for fold in range(N_FOLDS):
    print(f"\n{'='*60}")
    print(f"FOLD {fold+1}/{N_FOLDS}")
    print(f"{'='*60}")
    
    fold_key = f"fold_{fold}"
    if fold_key not in cv_results:
        cv_results[fold_key] = {}
    
    # Get fold-specific loaders
    train_loader, val_loader = get_fold_loaders(fold, use_all_train=False)
    ub_train_loader, ub_val_loader = get_fold_loaders(fold, use_all_train=True)
    
    print(f"Fold {fold+1} loaders: train={len(train_loader)} batches, val={len(val_loader)} batches")
    print(f"Upper bound: train={len(ub_train_loader)} batches")
    
    # --- Experiment A: Baseline ---
    if "baseline" not in cv_results[fold_key]:
        if USE_WANDB:
            wandb.init(project=WANDB_PROJECT, name=f"baseline_fold{fold+1}", reinit=True,
                       config={**MODEL_CFG, **TRAIN_CFG, "fold": fold+1, "use_pretrained": False, "kd": False})
        try:
            print(f"\n=== Fold {fold+1} — Baseline (random init) ===")
            model_baseline = build_model(pretrained=False)
            cv_results[fold_key]["baseline"] = finetune(f"baseline_fold{fold+1}", model_baseline,
                                                        train_loader, val_loader, teacher=None)
            _save_cv_results()
        finally:
            if USE_WANDB:
                wandb.finish()
    else:
        print(f"Fold {fold+1} baseline already done — skipping.")
    
    # --- Experiment B: SSL only ---
    if "ssl_only" not in cv_results[fold_key]:
        if USE_WANDB:
            wandb.init(project=WANDB_PROJECT, name=f"ssl_only_fold{fold+1}", reinit=True,
                       config={**MODEL_CFG, **TRAIN_CFG, "fold": fold+1, "use_pretrained": True, "kd": False})
        try:
            print(f"\n=== Fold {fold+1} — SSL only (SparK pretrained) ===")
            model_ssl = build_model(pretrained=True)
            cv_results[fold_key]["ssl_only"] = finetune(f"ssl_only_fold{fold+1}", model_ssl,
                                                        train_loader, val_loader, teacher=None)
            _save_cv_results()
        finally:
            if USE_WANDB:
                wandb.finish()
    else:
        print(f"Fold {fold+1} SSL only already done — skipping.")
    
    # --- Experiment C: SSL + KD ---
    if "ssl_kd" not in cv_results[fold_key]:
        if "ssl_only" not in cv_results[fold_key]:
            raise RuntimeError(f"Fold {fold+1} SSL-only must complete before SSL+KD.")
        
        if USE_WANDB:
            wandb.init(project=WANDB_PROJECT, name=f"ssl_kd_fold{fold+1}", reinit=True,
                       config={**MODEL_CFG, **TRAIN_CFG, "fold": fold+1, "use_pretrained": True, "kd": True})
        try:
            print(f"\n=== Fold {fold+1} — SSL + KD ===")
            teacher = build_model(pretrained=True)
            teacher.load_state_dict(
                torch.load(cv_results[fold_key]["ssl_only"]["ckpt"], map_location=DEVICE))
            teacher.eval()
            
            model_kd = build_model(pretrained=True)
            cv_results[fold_key]["ssl_kd"] = finetune(f"ssl_kd_fold{fold+1}", model_kd,
                                                      train_loader, val_loader, teacher=teacher)
            _save_cv_results()
        finally:
            if USE_WANDB:
                wandb.finish()
    else:
        print(f"Fold {fold+1} SSL+KD already done — skipping.")
    
    # --- Experiment D: Upper bound ---
    if "upper_bound" not in cv_results[fold_key]:
        if USE_WANDB:
            wandb.init(project=WANDB_PROJECT, name=f"upper_bound_fold{fold+1}", reinit=True,
                       config={**MODEL_CFG, **TRAIN_CFG, "fold": fold+1, "use_pretrained": False, "kd": False, "all_labels": True})
        try:
            print(f"\n=== Fold {fold+1} — Supervised upper bound ===")
            model_ub = build_model(pretrained=False)
            cv_results[fold_key]["upper_bound"] = finetune(f"upper_bound_fold{fold+1}", model_ub,
                                                           ub_train_loader, ub_val_loader, teacher=None)
            _save_cv_results()
        finally:
            if USE_WANDB:
                wandb.finish()
    else:
        print(f"Fold {fold+1} upper bound already done — skipping.")

# %% [markdown]
# ## 6 — Cross-validation results

# %%
import numpy as np

# Aggregate results across folds
def compute_cv_stats(cv_results, metric_key):
    """Compute mean ± std for a metric across all folds."""
    values = []
    for fold in range(N_FOLDS):
        fold_key = f"fold_{fold}"
        if fold_key in cv_results:
            for method in ["baseline", "ssl_only", "ssl_kd", "upper_bound"]:
                if method in cv_results[fold_key]:
                    val = cv_results[fold_key][method].get(metric_key, float("nan"))
                    if not np.isnan(val):
                        values.append((method, val))
    
    # Group by method
    method_values = {}
    for method, val in values:
        if method not in method_values:
            method_values[method] = []
        method_values[method].append(val)
    
    # Compute stats
    stats = {}
    for method, vals in method_values.items():
        if len(vals) > 0:
            stats[method] = {
                "mean": np.mean(vals),
                "std":  np.std(vals, ddof=1) if len(vals) > 1 else 0.0,
                "n":    len(vals)
            }
    return stats

dsc_stats  = compute_cv_stats(cv_results, "best_dsc")
hd95_stats = compute_cv_stats(cv_results, "best_hd95")

print("\n" + "=" * 70)
print("  Heart segmentation — Task02 (3-fold cross-validation)")
print("=" * 70)
print(f"  {'Method':<30} {'DSC (mean ± std)':<20} {'HD95 (mean ± std)':<20}")
print("-" * 70)

method_labels = {
    "upper_bound": "Supervised UB (all labels)",
    "baseline":    "Baseline (random, 1 label)",
    "ssl_only":    "SSL only (SparK, 1 label)",
    "ssl_kd":      "SSL + KD (SparK, 1 label)",
}

for method, label in method_labels.items():
    dsc_info  = dsc_stats.get(method, {})
    hd95_info = hd95_stats.get(method, {})
    
    if dsc_info:
        dsc_str = f"{dsc_info['mean']:.3f} ± {dsc_info['std']:.3f}"
    else:
        dsc_str = "N/A"
    
    if hd95_info:
        hd95_str = f"{hd95_info['mean']:.1f} ± {hd95_info['std']:.1f}"
    else:
        hd95_str = "N/A"
    
    print(f"  {label:<30} {dsc_str:<20} {hd95_str:<20}")

print("=" * 70)
print("  NOTE: nnU-Net reports 0.923 DSC on the official Decathlon")
print("  test server (different split) — cited as context only.")
print("=" * 70)

# Compute statistical comparisons
if "baseline" in dsc_stats and "ssl_only" in dsc_stats:
    baseline_mean = dsc_stats["baseline"]["mean"]
    ssl_mean      = dsc_stats["ssl_only"]["mean"]
    gain_ssl      = ssl_mean - baseline_mean
    print(f"\n  SSL gain over baseline   :  {gain_ssl:+.3f} DSC")

if "ssl_only" in dsc_stats and "ssl_kd" in dsc_stats:
    ssl_mean = dsc_stats["ssl_only"]["mean"]
    kd_mean  = dsc_stats["ssl_kd"]["mean"]
    gain_kd  = kd_mean - ssl_mean
    print(f"  KD  gain over SSL-only   :  {gain_kd:+.3f} DSC")

if "upper_bound" in dsc_stats and "ssl_kd" in dsc_stats and "baseline" in dsc_stats:
    ub_mean       = dsc_stats["upper_bound"]["mean"]
    kd_mean       = dsc_stats["ssl_kd"]["mean"]
    baseline_mean = dsc_stats["baseline"]["mean"]
    
    gap_to_ub = ub_mean - kd_mean
    if ub_mean > baseline_mean:
        pct_closed = (kd_mean - baseline_mean) / (ub_mean - baseline_mean) * 100
        print(f"  Gap to upper bound       :  {gap_to_ub:+.3f} DSC")
        print(f"  SSL+KD closes {pct_closed:.1f}% of gap to supervised UB")

# %%
# Save CV results
cv_results_path = Path(OUT_DIR) / "ssl_kd_cv_results.json"
cv_results_path.write_text(json.dumps(cv_results, indent=2))
print(f"\nCV results saved to {cv_results_path}")

# Also save summary stats for the paper
summary = {
    "dsc_stats":  dsc_stats,
    "hd95_stats": hd95_stats,
    "n_folds":    N_FOLDS,
    "seed":       SEED
}
summary_path = Path(OUT_DIR) / "ssl_kd_summary.json"
summary_path.write_text(json.dumps(summary, indent=2))
print(f"Summary stats saved to {summary_path}")

if USE_WANDB:
    save_checkpoint(cv_results_path, "ssl-kd-cv-results", "", "")
    save_checkpoint(summary_path, "ssl-kd-summary", "", "")
    print("Results uploaded to WandB artifacts")
