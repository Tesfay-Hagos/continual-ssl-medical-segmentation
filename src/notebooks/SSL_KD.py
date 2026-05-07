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
# 3. Add Kaggle secret: `SSL_KD_WANDB`
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
         "wandb", "scikit-learn", "pandas", "matplotlib", "seaborn", "--quiet"],
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
from utils.storage import set_wandb_entity

WANDB_PROJECT = "ssl-kd-heart"
try:
    if ON_KAGGLE:
        from kaggle_secrets import UserSecretsClient
        _key = UserSecretsClient().get_secret("SSL_KD_WANDB")
        wandb.login(key=_key)
    else:
        wandb.login()
    USE_WANDB = True
    _wandb_entity = wandb.Api().default_entity or ""
    set_wandb_entity(_wandb_entity)
    print(f"WandB logged in. Project: {WANDB_PROJECT}  Entity: {_wandb_entity}")
except Exception as e:
    USE_WANDB = False
    print(f"WandB login failed ({e}) — running without logging.")

# %% [markdown]
# ## 1 — Heart Dataset Setup (SSL+KD Experiment)

# %%
from data.datasets import kaggle_task_roots, build_task_roots, verify_datasets

TASK_ROOTS = (kaggle_task_roots() if ON_KAGGLE
              else build_task_roots(os.environ.get("DATA_ROOT", "/data/decathlon")))

# Only verify heart dataset for SSL+KD experiment
heart_root = TASK_ROOTS.get('heart')
if not heart_root or not os.path.exists(heart_root):
    raise RuntimeError(
        "Heart dataset missing. "
        "Add vivekprajapati2048/medical-segmentation-decathlon-heart as Kaggle input."
    )
print("Heart dataset verified ✅")
print(f"  Heart root: {heart_root}")

# %% [markdown]
# ## 1.5 — Exploratory Data Analysis
#
# Before training we explore all 20 Heart MRI volumes to understand the raw data
# and justify every key preprocessing and training decision.
#
# **Four questions we answer:**
# 1. What do the raw images and segmentation masks look like?
# 2. Why does MRI need per-volume intensity normalisation?
# 3. How severe is the class imbalance (heart vs background)?
# 4. What are the native voxel spacings and volume sizes?

# %%
import nibabel as nib
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from data.datasets import resolve_task_dir, glob_nii

_eda_fig_dir = Path(OUT_DIR) / "figures"
_eda_fig_dir.mkdir(parents=True, exist_ok=True)

try:
    _heart_task_dir = resolve_task_dir(heart_root, "heart")
    _img_paths = glob_nii(_heart_task_dir / "imagesTr")
    _lbl_paths = glob_nii(_heart_task_dir / "labelsTr")

    # Match labels to images by filename stem (handles .nii and .nii.gz)
    _lbl_map = {p.name.split(".")[0]: p for p in _lbl_paths}
    _img_lbl_pairs = [
        (ip, _lbl_map[ip.name.split(".")[0]])
        for ip in _img_paths
        if ip.name.split(".")[0] in _lbl_map
    ]
    print(f"Heart EDA: {len(_img_paths)} images, {len(_lbl_paths)} labels, "
          f"{len(_img_lbl_pairs)} matched pairs")
    _EDA_OK = True
except Exception as _e:
    print(f"EDA skipped — data not accessible: {_e}")
    _EDA_OK = False

# %% [markdown]
# ### 1.5.1 — Sample slices with heart mask overlay
#
# We display axial, sagittal, and coronal mid-slices for three randomly selected
# volumes. The heart mask (red, α = 0.45) shows the segmentation target.
#
# **Key observation:** MRI intensity varies dramatically across subjects — the same
# tissue appears bright in one volume and dim in another. This is the fundamental
# reason `NormalizeIntensityd(nonzero=True)` is applied: it z-scores each volume
# over its non-zero (foreground) voxels, making intensities comparable across scans.

# %%
if _EDA_OK:
    _rng_eda = np.random.default_rng(0)
    _sample_idx = _rng_eda.choice(len(_img_lbl_pairs),
                                   size=min(3, len(_img_lbl_pairs)), replace=False)

    fig, axes = plt.subplots(len(_sample_idx), 3,
                              figsize=(13, 4.2 * len(_sample_idx)))
    fig.suptitle(
        "Heart MRI — Axial / Sagittal / Coronal mid-slices\n"
        "(raw intensity, heart mask in red)",
        fontsize=13, y=1.01
    )

    for row, idx in enumerate(_sample_idx):
        ip, lp = _img_lbl_pairs[idx]
        vol  = nib.load(ip).get_fdata().astype(np.float32)
        mask = (nib.load(lp).get_fdata() > 0).astype(np.float32)

        cz, cy, cx = [s // 2 for s in vol.shape[:3]]
        views = [
            (vol[:, :, cz],  mask[:, :, cz],  "Axial"),
            (vol[:, cy, :],  mask[:, cy, :],  "Sagittal"),
            (vol[cx, :, :],  mask[cx, :, :],  "Coronal"),
        ]
        for col, (img_sl, msk_sl, title) in enumerate(views):
            ax = axes[row, col] if len(_sample_idx) > 1 else axes[col]
            fg_vals = img_sl[img_sl > 0]
            vmin, vmax = (np.percentile(fg_vals, [1, 99])
                          if fg_vals.size > 0 else (0.0, 1.0))
            ax.imshow(img_sl.T, cmap="gray", origin="lower", vmin=vmin, vmax=vmax)
            # Red mask overlay
            overlay = np.zeros((*img_sl.shape, 4))
            overlay[..., 0] = msk_sl
            overlay[..., 3] = msk_sl * 0.45
            ax.imshow(overlay.transpose(1, 0, 2), origin="lower")
            ax.set_title(f"Vol {idx} — {title}", fontsize=10)
            ax.axis("off")

    plt.tight_layout()
    fig.savefig(_eda_fig_dir / "eda_sample_slices.png", dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Saved → {_eda_fig_dir / 'eda_sample_slices.png'}")

# %% [markdown]
# ### 1.5.2 — Intensity distribution and effect of normalisation
#
# The scatter plot (left) shows that each volume has a different mean and standard
# deviation of foreground intensities — there is no fixed MRI scale analogous to
# Hounsfield Units in CT. The overlaid histograms (right) show the foreground
# intensities **after** `NormalizeIntensityd`: all volumes now centre near 0 with
# unit variance, making the network's input distribution stable across patients.

# %%
if _EDA_OK:
    raw_means, raw_stds = [], []

    for ip, _ in _img_lbl_pairs:
        vol = nib.load(ip).get_fdata().astype(np.float32)
        fg  = vol[vol > 0]
        if fg.size == 0:
            continue
        raw_means.append(float(fg.mean()))
        raw_stds.append(float(fg.std()))

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    # Left — per-volume mean vs std scatter
    ax = axes[0]
    ax.scatter(raw_means, raw_stds, alpha=0.75, edgecolors="black", s=65,
               color="#5588cc")
    for i, (m, s) in enumerate(zip(raw_means, raw_stds)):
        ax.annotate(str(i), (m, s), fontsize=7, alpha=0.55,
                    xytext=(3, 3), textcoords="offset points")
    ax.set_xlabel("Foreground mean intensity (raw)")
    ax.set_ylabel("Foreground std intensity (raw)")
    ax.set_title("Per-volume intensity statistics\n(before normalisation)")
    ax.grid(alpha=0.3)

    # Right — overlaid normalised histograms for first 10 volumes
    ax = axes[1]
    for ip, _ in list(_img_lbl_pairs)[:10]:
        vol = nib.load(ip).get_fdata().astype(np.float32)
        fg  = vol[vol > 0]
        if fg.size == 0:
            continue
        norm = (fg - fg.mean()) / (fg.std() + 1e-8)
        ax.hist(norm, bins=60, alpha=0.35, density=True)
    ax.axvline(0, color="black", linestyle="--", linewidth=1, alpha=0.5,
               label="μ = 0 after norm.")
    ax.set_xlabel("Normalised intensity  (z-score, nonzero voxels)")
    ax.set_ylabel("Density")
    ax.set_title("Foreground intensity after NormalizeIntensityd\n"
                 "(first 10 volumes, overlaid)")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    fig.savefig(_eda_fig_dir / "eda_intensity.png", dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Raw intensity range across volumes:")
    print(f"  mean ∈ [{min(raw_means):.0f}, {max(raw_means):.0f}]  "
          f"std  ∈ [{min(raw_stds):.0f}, {max(raw_stds):.0f}]")
    print(f"Saved → {_eda_fig_dir / 'eda_intensity.png'}")

# %% [markdown]
# ### 1.5.3 — Class balance: heart vs background
#
# The heart (left atrium) occupies roughly **5–10 % of the total MRI volume**.
# This severe imbalance means that a random 96³ patch drawn uniformly has a high
# probability of containing zero heart voxels, causing the network to collapse to
# predicting all-background.
#
# `RandCropByPosNegLabeld(pos=1, neg=1, num_samples=8)` directly counters this:
# half the patches are guaranteed to be centred on a foreground voxel, giving the
# model balanced exposure to both classes during training.

# %%
if _EDA_OK:
    fg_fractions = []
    for _, lp in _img_lbl_pairs:
        mask = nib.load(lp).get_fdata() > 0
        fg_fractions.append(float(mask.mean()))

    fg_mean = np.mean(fg_fractions)
    fg_std  = np.std(fg_fractions)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    # Left — histogram of foreground fractions
    ax = axes[0]
    ax.hist(fg_fractions, bins=10, color="#7f7fff", edgecolor="black", alpha=0.8)
    ax.axvline(fg_mean, color="red", linestyle="--",
               label=f"Mean = {fg_mean*100:.1f}%")
    ax.set_xlabel("Foreground fraction  (heart / total voxels)")
    ax.set_ylabel("Number of volumes")
    ax.set_title("Class balance per volume")
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)

    # Right — average pie chart
    ax = axes[1]
    ax.pie(
        [1 - fg_mean, fg_mean],
        labels=["Background", f"Heart\n({fg_mean*100:.1f}%)"],
        colors=["#dddddd", "#7f7fff"],
        autopct="%1.1f%%",
        startangle=140,
        wedgeprops={"edgecolor": "black", "linewidth": 0.8},
    )
    ax.set_title(f"Mean class split  (n = {len(fg_fractions)} volumes)")

    plt.tight_layout()
    fig.savefig(_eda_fig_dir / "eda_class_balance.png", dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Foreground fraction — mean: {fg_mean*100:.2f}%  std: {fg_std*100:.2f}%")
    print(f"  min: {min(fg_fractions)*100:.2f}%   max: {max(fg_fractions)*100:.2f}%")
    print(f"Saved → {_eda_fig_dir / 'eda_class_balance.png'}")

# %% [markdown]
# ### 1.5.4 — Native voxel spacings and volume dimensions
#
# Each MRI scanner and protocol produces a different voxel size and field of view.
# The scatter plot (left) shows that in-plane and through-plane spacings are close
# to 1.25 mm for most volumes but vary enough to affect learned spatial features.
# All volumes are therefore resampled to **isotropic 1.25 × 1.25 × 1.25 mm** with
# `Spacingd`, making metric distances consistent across subjects.
#
# The colour-coded scatter (right) shows that H × W extent is fairly consistent
# (~320 px), while depth D (colour) varies more widely — motivating `SpatialPadd`
# to guarantee the minimum 96³ crop size.

# %%
if _EDA_OK:
    spacings, shapes = [], []
    for ip, _ in _img_lbl_pairs:
        nib_img = nib.load(ip)
        sp = tuple(float(x) for x in nib_img.header.get_zooms()[:3])
        sh = nib_img.shape[:3]
        spacings.append(sp)
        shapes.append(sh)

    spacings = np.array(spacings, dtype=float)
    shapes   = np.array(shapes,   dtype=int)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

    # Left — in-plane vs through-plane spacing
    ax = axes[0]
    ax.scatter(spacings[:, 0], spacings[:, 2], alpha=0.75,
               edgecolors="black", s=65, color="#55aa88")
    ax.axhline(1.25, color="red", linestyle="--", linewidth=1.2,
               label="Target 1.25 mm")
    ax.axvline(1.25, color="red", linestyle="--", linewidth=1.2)
    ax.set_xlabel("In-plane spacing (mm)")
    ax.set_ylabel("Through-plane spacing (mm)")
    ax.set_title("Native voxel spacing per volume\n(red dashed = resampling target)")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    # Right — H × W scatter, colour = depth D
    ax = axes[1]
    sc = ax.scatter(shapes[:, 0], shapes[:, 1], c=shapes[:, 2],
                    cmap="viridis", s=75, edgecolors="black", alpha=0.85)
    plt.colorbar(sc, ax=ax, label="Depth  D  (slices)")
    ax.set_xlabel("Height  H  (voxels)")
    ax.set_ylabel("Width   W  (voxels)")
    ax.set_title("Native volume dimensions\n(colour = slice depth)")
    ax.grid(alpha=0.3)

    plt.tight_layout()
    fig.savefig(_eda_fig_dir / "eda_spacings.png", dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Native spacing  — mean {spacings.mean(axis=0).round(3)} mm  "
          f"std {spacings.std(axis=0).round(3)} mm")
    print(f"Native shape    — mean {shapes.mean(axis=0).astype(int)}  "
          f"std {shapes.std(axis=0).round(1)}")
    print(f"Saved → {_eda_fig_dir / 'eda_spacings.png'}")

# %% [markdown]
# ### 1.5.5 — EDA Summary
#
# | Property | Finding | Design consequence |
# |---|---|---|
# | Volumes | 20 total (13 train / 7 val per fold) | 3-fold CV to make each volume count |
# | Modality | MRI — left atrium | No HU scale → `NormalizeIntensityd` |
# | Intensity range | Varies widely per subject | z-score normalisation per foreground |
# | Foreground fraction | ~5–10 % | `pos=1, neg=1` balanced crop sampling |
# | Native spacing | ≈1.25 × 1.25 × … mm, variable | Resample to isotropic 1.25 mm |
# | Volume size | ~320 × 320 × 130 voxels (native) | `SpatialPadd` ensures ≥ 96³ crops |
# | Few-shot setup | Only **1** labeled volume per fold | SSL + KD compensate for label scarcity |

# %%
if _EDA_OK:
    print("\n" + "=" * 60)
    print("  EDA Summary — Heart MRI (Task02_Heart)")
    print("=" * 60)
    print(f"  Total volumes        : {len(_img_lbl_pairs)}")
    print(f"  Foreground (mean±std): {fg_mean*100:.2f}% ± {fg_std*100:.2f}%")
    print(f"  Native spacing (mm)  : {spacings.mean(axis=0).round(2)}")
    print(f"  Native shape (HWD)   : {shapes.mean(axis=0).astype(int)}")
    print(f"  Resampled to         : (1.25, 1.25, 1.25) mm isotropic")
    print(f"  EDA figures saved to : {_eda_fig_dir}")
    print("=" * 60)

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

_BEST_CKPT      = "best.pth"
_LATEST_CKPT    = "latest.pth"
_CV_RESULTS_FILE = "ssl_kd_cv_results.json"

PRETRAIN_DIR  = Path(os.path.join(OUT_DIR, "pretrain"))
PRETRAIN_CKPT = PRETRAIN_DIR / _BEST_CKPT
PRETRAIN_DONE = PRETRAIN_DIR / "pretrain_done.json"
PRETRAIN_DIR.mkdir(parents=True, exist_ok=True)

# Try to restore completion marker + encoder from WandB
restore_checkpoint("pretrain_done.json", PRETRAIN_DIR,
                   "pretrain-done", "cssl-medical", "", "")
restore_checkpoint(_BEST_CKPT, PRETRAIN_DIR,
                   "pretrain-encoder", "cssl-medical", "", "")

# Fallback: extract encoder weights from full training checkpoint
if not PRETRAIN_CKPT.exists():
    restore_checkpoint(_LATEST_CKPT, PRETRAIN_DIR,
                       "pretrain-checkpoint", "cssl-medical", "", "")
    _latest = PRETRAIN_DIR / _LATEST_CKPT
    if _latest.exists():
        _full = torch.load(_latest, map_location="cpu")["model"]
        _enc  = {k[len("encoder."):]: v
                 for k, v in _full.items() if k.startswith("encoder.")}
        torch.save(_enc, PRETRAIN_CKPT)
        print(f"Extracted encoder weights from {_LATEST_CKPT} → {_BEST_CKPT}")

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
    "epochs":          50,    # Reduced from 300 for one-day run
    "warmup_epochs":   5,     # Reduced from 10
    "lr":              2.0e-4, # Slightly higher LR for faster convergence
    "weight_decay":    1.0e-5,
    "patience":        20,    # Reduced from 50
    # KD
    "kd_alpha":        1.0,   # weight of KD loss relative to DiceCE
    "kd_temperature":  2.0,
}

# %%
from models.unet import build_unet, UNetWithEncoder
from monai.losses import DiceCELoss
from monai.inferers import sliding_window_inference
from monai.data import CacheDataset, DataLoader
from monai.transforms import (Compose, LoadImaged, EnsureChannelFirstd,
                               Spacingd, Orientationd, NormalizeIntensityd,
                               SpatialPadd, RandCropByPosNegLabeld,
                               RandFlipd, RandGaussianNoised,
                               RandScaleIntensityd, ToTensord)
from data.datasets import get_loaders, get_file_list, get_transforms
from sklearn.model_selection import KFold

NUM_CROPS = 8  # patches extracted per training volume per epoch

def get_multicrop_train_transform(num_crops: int = NUM_CROPS) -> Compose:
    """Heart MRI train transform that extracts num_crops patches per volume.
    RandCropByPosNegLabeld guarantees ~50% of crops land on heart voxels,
    preventing the model from predicting all-background due to class imbalance.
    """
    return Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        Spacingd(keys=["image", "label"],
                 pixdim=(1.25, 1.25, 1.25), mode=("bilinear", "nearest")),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        NormalizeIntensityd(keys=["image"], nonzero=True),
        SpatialPadd(keys=["image", "label"], spatial_size=(96, 96, 96)),
        RandCropByPosNegLabeld(
            keys=["image", "label"],
            label_key="label",
            spatial_size=(96, 96, 96),
            pos=1, neg=1,          # equal chance of heart-centered vs background crop
            num_samples=num_crops, # extract num_crops patches per volume
            image_key="image",
        ),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
        RandGaussianNoised(keys=["image"], prob=0.2, std=0.05),
        RandScaleIntensityd(keys=["image"], factors=0.1, prob=0.5),
        ToTensord(keys=["image", "label"]),
    ])


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
        # Handle multiple samples from RandCropByPosNegLabeld
        if isinstance(batch, list):
            batch = batch[0]  # Take first sample if multiple generated
            
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


def _finetune_step(model, batch, criterion, teacher, optimizer, scaler,
                   alpha, t_kd, debug=False):
    """Run one gradient step; return loss value or None if non-finite."""
    if isinstance(batch, list):
        batch = batch[0]
    imgs   = batch["image"].to(DEVICE)
    labels = batch["label"].to(DEVICE)
    if labels.dim() == 4:
        labels = labels.unsqueeze(1)
    labels = labels.long()

    optimizer.zero_grad()
    with torch.amp.autocast(device_type=DEVICE.type):
        preds = model(imgs)
        loss  = criterion(preds, labels)
        if debug:
            print(f"  Debug - Image shape: {imgs.shape}, range: [{imgs.min():.3f}, {imgs.max():.3f}]")
            print(f"  Debug - Label shape: {labels.shape}, unique values: {torch.unique(labels, dim=None)}")
            print(f"  Debug - Pred shape: {preds.shape}, range: [{preds.min():.3f}, {preds.max():.3f}]")
            print(f"  Debug - Label coverage: {(labels > 0).float().mean():.4f}")
            print(f"  Debug - Loss value: {loss.item():.6f}")

    if teacher is not None:
        with torch.no_grad():
            t_soft = F.softmax(teacher(imgs).float() / t_kd, dim=1)
        s_log    = F.log_softmax(preds.float() / t_kd, dim=1)
        n_voxels = imgs.shape[2] * imgs.shape[3] * imgs.shape[4]
        kd_loss  = F.kl_div(s_log, t_soft.detach(), reduction="sum") / (imgs.shape[0] * n_voxels)
        loss     = loss + alpha * kd_loss * (t_kd ** 2)

    if not torch.isfinite(loss):
        optimizer.zero_grad()
        return None

    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    scaler.step(optimizer)
    scaler.update()
    return loss.item()


def finetune(run_name: str, model, train_loader, val_loader,
             teacher=None) -> dict:
    """Fine-tune model on heart. Returns dict with best DSC, HD95, checkpoint path."""
    ckpt_dir = Path(OUT_DIR) / run_name
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    criterion   = DiceCELoss(to_onehot_y=True, softmax=True)
    optimizer   = torch.optim.AdamW(model.parameters(),
                                    lr=TRAIN_CFG["lr"],
                                    weight_decay=TRAIN_CFG["weight_decay"])
    scheduler   = make_scheduler(optimizer, TRAIN_CFG["epochs"], TRAIN_CFG["warmup_epochs"])
    scaler      = torch.amp.GradScaler(DEVICE.type, enabled=(DEVICE.type == "cuda"))
    patience    = TRAIN_CFG["patience"]
    alpha       = TRAIN_CFG["kd_alpha"]
    t_kd        = TRAIN_CFG["kd_temperature"]
    best_dsc    = 0.0
    best_hd95   = float("inf")
    trigger     = 0
    start_epoch = 0

    resume_ckpt = ckpt_dir / _LATEST_CKPT
    if resume_ckpt.exists():
        state       = torch.load(resume_ckpt, map_location=DEVICE)
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
        epoch_loss, n_batches = 0.0, 0
        for batch in train_loader:
            step_loss = _finetune_step(model, batch, criterion, teacher, optimizer, scaler,
                                       alpha, t_kd, debug=(epoch == 0 and n_batches == 0))
            if step_loss is not None:
                epoch_loss += step_loss
                n_batches  += 1

        scheduler.step()
        avg_loss = epoch_loss / max(n_batches, 1)
        metrics  = evaluate(model, val_loader)
        dsc, hd95 = metrics["dice"], metrics["hd95"]

        print(f"  [{run_name}] Epoch {epoch+1:>3}/{TRAIN_CFG['epochs']} | "
              f"loss={avg_loss:.4f} | DSC={dsc:.4f} | HD95={hd95:.1f} | "
              f"best={best_dsc:.4f} | lr={scheduler.get_last_lr()[0]:.2e}")
        log_wandb({f"{run_name}/loss": avg_loss, f"{run_name}/dsc": dsc,
                   f"{run_name}/hd95": hd95, f"{run_name}/epoch": epoch + 1})

        if dsc >= best_dsc:
            best_dsc, best_hd95 = dsc, hd95
            best_ckpt_path = ckpt_dir / _BEST_CKPT
            torch.save(model.state_dict(), best_ckpt_path)
            # Upload best checkpoint to WandB so teacher can be loaded after restart
            save_checkpoint(best_ckpt_path, f"{run_name}-best", "", "")
            trigger = 0
        else:
            trigger += 1
            if trigger >= patience:
                print(f"  Early stopping at epoch {epoch+1}.")
                break

        torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(), "epoch": epoch,
                    "best_dsc": best_dsc, "best_hd95": best_hd95,
                    "trigger": trigger}, resume_ckpt)

    log_wandb({f"{run_name}/best_dsc": best_dsc, f"{run_name}/best_hd95": best_hd95})
    print(f"  [{run_name}] Done. Best DSC={best_dsc:.4f}  HD95={best_hd95:.1f}")
    return {"run": run_name, "best_dsc": best_dsc, "best_hd95": best_hd95,
            "ckpt": str(ckpt_dir / _BEST_CKPT)}

# %% [markdown]
# ## 5 — Cross-validation setup and experiments
#
# 3-fold CV: each fold uses 13 train / 7 val volumes.
# Reports mean ± std across folds for statistical validity.

# %%
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

_cv_results_path = Path(OUT_DIR) / _CV_RESULTS_FILE
cv_results = {}

# Primary restore: query WandB API directly for each expected run.
# This is more reliable than downloading a JSON artifact because it works
# regardless of WandB's internal storage format (parquet, etc.).
_CV_METHODS = ["baseline", "ssl_only", "ssl_kd", "upper_bound"]

from utils.cv_restore import restore_cv_from_wandb as _restore_cv_from_wandb
if USE_WANDB:
    _restore_cv_from_wandb(cv_results, WANDB_PROJECT, N_FOLDS, _BEST_CKPT, OUT_DIR)

# Also try JSON artifact as secondary fallback (covers runs logged before this change)
if not cv_results:
    restore_checkpoint(_CV_RESULTS_FILE, Path(OUT_DIR),
                       "ssl-kd-cv-results", WANDB_PROJECT, "", "")
    if _cv_results_path.exists():
        cv_results.update(json.loads(_cv_results_path.read_text()))

if cv_results:
    print(f"CV results loaded — folds: {sorted(cv_results)}")
    for fk, fv in sorted(cv_results.items()):
        print(f"  {fk}: {sorted(fv)}")
else:
    print("No previous CV results found — starting fresh.")


def _save_cv_results():
    _cv_results_path.write_text(json.dumps(cv_results, indent=2))
    if USE_WANDB and wandb.run is not None:
        save_checkpoint(_cv_results_path, "ssl-kd-cv-results", "", "")
        print(f"  💾 cv_results saved (folds={sorted(cv_results)})")

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
        # Few-shot: repeat the single volume NUM_CROPS times so DataLoader
        # sees NUM_CROPS independent samples, each getting a different random
        # crop. CacheDataset caches the 1 unique file once and serves it
        # NUM_CROPS times with different augmentations each epoch.
        fold_train_files = fold_train_files[:1] * NUM_CROPS
        train_transform  = get_transforms("heart", train=True)
    else:
        # Upper bound: all volumes, standard single-crop transform
        train_transform  = get_transforms("heart", train=True)

    train_ds = CacheDataset(fold_train_files,
                            transform=train_transform,
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
            ssl_ckpt_path = Path(cv_results[fold_key]["ssl_only"]["ckpt"])
            if not ssl_ckpt_path.exists():
                # Local checkpoint lost (session restart) — restore from WandB artifact
                run_name = cv_results[fold_key]["ssl_only"]["run"]
                restore_checkpoint(_BEST_CKPT, ssl_ckpt_path.parent,
                                   f"{run_name}-best", WANDB_PROJECT, "", "")
            teacher.load_state_dict(torch.load(ssl_ckpt_path, map_location=DEVICE))
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
    buckets = {"baseline": [], "ssl_only": [], "ssl_kd": [], "upper_bound": []}
    for fold in range(N_FOLDS):
        fold_data = cv_results.get(f"fold_{fold}", {})
        for method, vals in buckets.items():
            v = fold_data.get(method, {}).get(metric_key, float("nan"))
            if not np.isnan(v):
                vals.append(v)
    return {
        m: {"mean": np.mean(vs), "std": np.std(vs, ddof=1) if len(vs) > 1 else 0.0, "n": len(vs)}
        for m, vs in buckets.items() if vs
    }

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

# %% [markdown]
# ## 7 — Publication Figures

# %%
import matplotlib.pyplot as plt
import seaborn as sns

# Set publication style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")
plt.rcParams.update({
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 11,
    'figure.titlesize': 16
})

fig_dir = Path(OUT_DIR) / "figures"
fig_dir.mkdir(exist_ok=True)

# %%
# Figure 1: Method comparison bar plot with error bars
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

methods = ['Baseline\n(Random)', 'SSL Only\n(SparK)', 'SSL + KD\n(SparK)', 'Upper Bound\n(All Labels)']
method_keys = ['baseline', 'ssl_only', 'ssl_kd', 'upper_bound']
colors = ['#ff7f7f', '#7fbf7f', '#7f7fff', '#bf7fbf']

# DSC plot
dsc_means = [dsc_stats[k]['mean'] if k in dsc_stats else 0 for k in method_keys]
dsc_stds = [dsc_stats[k]['std'] if k in dsc_stats else 0 for k in method_keys]

bars1 = ax1.bar(methods, dsc_means, yerr=dsc_stds, capsize=5, color=colors, alpha=0.8, edgecolor='black')
ax1.set_ylabel('Dice Similarity Coefficient')
ax1.set_title('Heart Segmentation Performance')
ax1.set_ylim(0, 1.0)
ax1.grid(axis='y', alpha=0.3)

# Add value labels on bars
for bar, mean, std in zip(bars1, dsc_means, dsc_stds):
    if mean > 0:
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + std + 0.01,
                f'{mean:.3f}±{std:.3f}', ha='center', va='bottom', fontweight='bold')

# HD95 plot
hd95_means = [hd95_stats[k]['mean'] if k in hd95_stats else 0 for k in method_keys]
hd95_stds = [hd95_stats[k]['std'] if k in hd95_stats else 0 for k in method_keys]

bars2 = ax2.bar(methods, hd95_means, yerr=hd95_stds, capsize=5, color=colors, alpha=0.8, edgecolor='black')
ax2.set_ylabel('Hausdorff Distance 95% (mm)')
ax2.set_title('Boundary Accuracy')
ax2.grid(axis='y', alpha=0.3)

# Add value labels on bars
for bar, mean, std in zip(bars2, hd95_means, hd95_stds):
    if mean > 0:
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + std + 1,
                f'{mean:.1f}±{std:.1f}', ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
fig.savefig(fig_dir / 'method_comparison.png', dpi=300, bbox_inches='tight')
fig.savefig(fig_dir / 'method_comparison.pdf', bbox_inches='tight')
print(f"Figure 1 saved: {fig_dir / 'method_comparison.png'}")

# %%
# Figure 2: Individual fold results (box plot)
fig, ax = plt.subplots(1, 1, figsize=(10, 6))

# Collect all fold data
fold_data = []
for method_key, method_label in zip(method_keys, methods):
    for fold in range(N_FOLDS):
        fold_key = f"fold_{fold}"
        if fold_key in cv_results and method_key in cv_results[fold_key]:
            dsc_val = cv_results[fold_key][method_key].get('best_dsc', None)
            if dsc_val is not None:
                fold_data.append({
                    'Method': method_label,
                    'DSC': dsc_val,
                    'Fold': fold + 1
                })

if fold_data:
    import pandas as pd
    df = pd.DataFrame(fold_data)
    
    # Box plot with individual points
    sns.boxplot(data=df, x='Method', y='DSC', ax=ax, palette=colors)
    sns.stripplot(data=df, x='Method', y='DSC', ax=ax, color='black', alpha=0.7, size=8)
    
    ax.set_ylabel('Dice Similarity Coefficient')
    ax.set_title('Cross-Validation Results (Individual Folds)')
    ax.set_ylim(0, 1.0)
    ax.grid(axis='y', alpha=0.3)
    
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    fig.savefig(fig_dir / 'cv_boxplot.png', dpi=300, bbox_inches='tight')
    fig.savefig(fig_dir / 'cv_boxplot.pdf', bbox_inches='tight')
    print(f"Figure 2 saved: {fig_dir / 'cv_boxplot.png'}")

# %%
# Figure 3: Improvement analysis
fig, ax = plt.subplots(1, 1, figsize=(8, 6))

if 'baseline' in dsc_stats and 'ssl_only' in dsc_stats and 'ssl_kd' in dsc_stats:
    baseline_mean = dsc_stats['baseline']['mean']
    ssl_mean = dsc_stats['ssl_only']['mean']
    kd_mean = dsc_stats['ssl_kd']['mean']
    
    improvements = {
        'SSL Pretraining\nGain': ssl_mean - baseline_mean,
        'Knowledge Distillation\nGain': kd_mean - ssl_mean,
        'Total SSL+KD\nGain': kd_mean - baseline_mean
    }
    
    bars = ax.bar(improvements.keys(), improvements.values(), 
                  color=['#7fbf7f', '#7f7fff', '#bf7fbf'], alpha=0.8, edgecolor='black')
    
    ax.set_ylabel('DSC Improvement')
    ax.set_title('Contribution Analysis')
    ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for bar, val in zip(bars, improvements.values()):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
                f'{val:+.3f}', ha='center', va='bottom', fontweight='bold')
    
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    fig.savefig(fig_dir / 'improvement_analysis.png', dpi=300, bbox_inches='tight')
    fig.savefig(fig_dir / 'improvement_analysis.pdf', bbox_inches='tight')
    print(f"Figure 3 saved: {fig_dir / 'improvement_analysis.png'}")

# %%
# Create figure summary for paper
fig_summary = {
    "figures_generated": [
        "method_comparison.png - Main results bar chart with error bars",
        "cv_boxplot.png - Cross-validation individual fold results", 
        "improvement_analysis.png - SSL and KD contribution breakdown"
    ],
    "figure_directory": str(fig_dir),
    "formats": ["PNG (300 DPI)", "PDF (vector)"]
}

fig_summary_path = fig_dir / "figure_summary.json"
fig_summary_path.write_text(json.dumps(fig_summary, indent=2))
print(f"\nFigure summary: {fig_summary_path}")
print("\n📊 Publication figures generated:")
for fig_desc in fig_summary["figures_generated"]:
    print(f"  ✅ {fig_desc}")

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
    "seed":       SEED,
    "figures":    fig_summary
}
summary_path = Path(OUT_DIR) / "ssl_kd_summary.json"
summary_path.write_text(json.dumps(summary, indent=2))
print(f"Summary stats saved to {summary_path}")

if USE_WANDB:
    save_checkpoint(cv_results_path, "ssl-kd-cv-results", "", "")
    save_checkpoint(summary_path, "ssl-kd-summary", "", "")
    # Upload figures to WandB
    for fig_file in fig_dir.glob("*.png"):
        save_checkpoint(fig_file, f"figure-{fig_file.stem}", "", "")
    print("Results and figures uploaded to WandB artifacts")

# %% [markdown]
# ## 8 — Model Architecture Analysis & FlashTorch Visualizations
#
# Three complementary analyses:
# 1. **Numeric architecture analysis** — parameter counts and encoder/decoder split for
#    every model variant (Baseline / SSL-only / SSL+KD teacher / Improved-KD student).
# 2. **FlashTorch gradient saliency** — which heart pixels drive each model's prediction.
#    We adapt FlashTorch's `Backprop` to work with a 3D U-Net segmentation model.
# 3. **Encoder activation maps** — forward hooks compare feature representations at each
#    encoder stage between a randomly-initialised model and an SSL-pretrained one.

# %%
# Install flashtorch if not already available (self-contained — only needed for this section)
try:
    import flashtorch  # noqa: F401
except ImportError:
    subprocess.run([sys.executable, "-m", "pip", "install", "flashtorch", "--quiet"],
                   check=True)

from flashtorch.saliency import Backprop
from flashtorch.activmax import GradientAscent

# %% [markdown]
# ### 8.1 — Numeric Architecture Analysis
#
# Every model variant in this experiment uses the same U-Net backbone but differs in
# initialisation (random vs SparK pretrained) and loss (DiceCE vs DiceCE + KL).
# The improved-KD student uses a **2× narrower** encoder to study capacity vs accuracy.
#
# | Model | Channels | Init | Loss |
# |---|---|---|---|
# | Baseline | 32-64-128-256-512 | Random | DiceCE |
# | SSL-only | 32-64-128-256-512 | SparK | DiceCE |
# | SSL+KD teacher | 32-64-128-256-512 | SparK | DiceCE + KL |
# | Improved-KD student | 16-32-64-128-256 | Random | DiceCE + KL |

# %%
def _param_stats(model: nn.Module, name: str) -> dict:
    """Compute parameter count and encoder/decoder split for a UNetWithEncoder."""
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    size_mb   = total * 4 / (1024 ** 2)

    decoder_ids = {
        id(p)
        for m in model.unet.modules()
        if isinstance(m, (nn.ConvTranspose3d, nn.ConvTranspose2d))
        for p in m.parameters(recurse=False)
    }
    dec_params = sum(p.numel() for p in model.parameters() if id(p) in decoder_ids)
    enc_params = total - dec_params
    return {
        "name":      name,
        "total":     total,
        "trainable": trainable,
        "encoder":   enc_params,
        "decoder":   dec_params,
        "size_mb":   size_mb,
    }


# Build models on CPU (no checkpoints needed for structure analysis)
_m_base    = build_model(pretrained=False).cpu()
try:
    _m_student = _build_student().cpu()   # defined in improved_kd_experiment.py section
except NameError:
    from models.unet import build_unet
    _m_student = UNetWithEncoder(
        build_unet(in_channels=1, out_channels=2,
                   channels=(16, 32, 64, 128, 256), strides=(2, 2, 2, 2))
    ).cpu()

_arch_variants = [
    ("Baseline / SSL-only / SSL+KD (teacher)",  _m_base),
    ("Improved-KD student  (2× narrower)",       _m_student),
]
_arch_stats = [_param_stats(m, n) for n, m in _arch_variants]

print("\n" + "=" * 78)
print("  Model Architecture — Parameter Summary")
print("=" * 78)
print(f"  {'Model':<44} {'Total':>10}  {'Encoder':>10}  {'Decoder':>10}  {'MB':>6}")
print("-" * 78)
for s in _arch_stats:
    print(f"  {s['name']:<44} {s['total']:>10,}  {s['encoder']:>10,}  "
          f"{s['decoder']:>10,}  {s['size_mb']:>6.1f}")
print("=" * 78)

# Encoder / decoder ratio for each model
for s in _arch_stats:
    enc_pct = s['encoder'] / s['total'] * 100
    print(f"  {s['name']}")
    print(f"    Encoder {enc_pct:.1f}%  |  Decoder {100-enc_pct:.1f}%  "
          f"|  Trainable {s['trainable']/s['total']*100:.0f}%")

# %%
# Detailed layer table for the base U-Net (one shared backbone for methods 1–3)
print("\nLayer-by-layer breakdown — base model (channels 32-64-128-256-512):")
print(f"  {'Parameter name':<58} {'Shape':<22} {'#Params':>10}")
print("-" * 94)
_running = 0
for _pname, _p in _m_base.named_parameters():
    _n = _p.numel()
    _running += _n
    print(f"  {_pname:<58} {str(list(_p.shape)):<22} {_n:>10,}")
print("-" * 94)
print(f"  {'TOTAL':<58} {'':22} {_running:>10,}")

# %% [markdown]
# ### 8.2 — FlashTorch Gradient Saliency
#
# **How it works:** `Backprop.calculate_gradients` back-propagates the gradient of the
# predicted *foreground score* (class 1 = heart) through the entire network to the input.
# Large gradient magnitude at a pixel means that pixel has high influence on the predicted
# heart segmentation — effectively a sensitivity map.
#
# **3D adaptation:** FlashTorch expects a 2D input and a classification model returning
# `[B, num_classes]`. We use `_SegSaliencyAdapter` to:
# 1. Accept a 2D axial slice `[B, 1, H, W]`
# 2. Tile it to a `SLAB_D=32` deep volume (sufficient depth for 4 stride-2 layers)
# 3. Run the full 3D U-Net and collapse spatial output → `[B, 2]` class scores
# 4. Override FlashTorch's backward hook to extract the 2D mid-slice gradient
#
# **What to look for:** SSL-pretrained models typically show sharper, more anatomically
# focused gradients around the left-atrium boundary compared to random-init baselines.

# %%
class _SegSaliencyAdapter(nn.Module):
    """Makes the 3D U-Net segmentation model compatible with FlashTorch Backprop.

    FlashTorch expects: model(x) → [B, C]  (classification scores)
    This adapter:       receives [B, 1, H, W] (2D slice)
                        tiles to  [B, 1, SLAB_D, H, W] (pseudo-3D volume)
                        returns   [B, 2] (bg / fg spatial mean scores)
    """
    SLAB_D = 32  # must satisfy 32 >= 2^num_strides = 16

    def __init__(self, model_3d: nn.Module):
        super().__init__()
        self.model = model_3d

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        x3d    = x.unsqueeze(2).expand(B, C, self.SLAB_D, H, W).contiguous()
        logits = self.model(x3d)                     # [B, 2, SLAB_D, H, W]
        mid    = self.SLAB_D // 2
        return logits[:, :, mid, :, :].mean((-2, -1))  # [B, 2]


class _Backprop3D(Backprop):
    """FlashTorch Backprop subclass that handles 3D gradients from our adapter.

    The default hook captures grad_in[0] from the first sub-module. For our adapter
    that sub-module is UNetWithEncoder whose input is a 5-D tensor [B,1,D,H,W].
    We sum over the D dimension to project back to the 2D input plane.
    """
    def _register_hooks(self):
        def _hook_fn(module, grad_in, grad_out):
            if grad_in[0] is None:
                return
            g = grad_in[0]
            if g.ndim == 5:                          # [B, 1, SLAB_D, H, W]
                self.gradients = g.sum(dim=2)        # → [B, 1, H, W]
            else:
                self.gradients = g
        first_sub = list(self.model.modules())[1]
        first_sub.register_backward_hook(_hook_fn)


# %%
# Load best fold-0 checkpoints for each method (CPU)
_saliency_models = {}
for _mkey in ["baseline", "ssl_only", "ssl_kd"]:
    _ckpt = cv_results.get("fold_0", {}).get(_mkey, {}).get("ckpt", "")
    if _ckpt and Path(_ckpt).exists():
        _m = build_model(pretrained=False).cpu().eval()
        _m.load_state_dict(torch.load(_ckpt, map_location="cpu"))
        _saliency_models[_mkey] = _m
        print(f"  Loaded {_mkey} checkpoint for saliency: {_ckpt}")
    else:
        print(f"  {_mkey}: checkpoint not found — saliency will be skipped for this method.")

# %%
if _EDA_OK and _saliency_models:
    # Prepare a normalised 2D axial mid-slice from the first Heart MRI volume
    _ip, _lp = _img_lbl_pairs[0]
    _vol  = nib.load(_ip).get_fdata().astype(np.float32)
    _mask = (nib.load(_lp).get_fdata() > 0)
    _fg   = _vol[_vol > 0]
    _vnorm = (_vol - _fg.mean()) / (_fg.std() + 1e-8)

    _mid_z      = _vol.shape[2] // 2
    _slice_np   = _vnorm[:, :, _mid_z]                      # [H, W]
    _mask_sl    = _mask[:, :, _mid_z]                       # [H, W]
    _input_2d   = torch.tensor(_slice_np, dtype=torch.float32)[None, None]  # [1,1,H,W]

    _method_labels = {
        "baseline": "Baseline\n(random init)",
        "ssl_only": "SSL-only\n(SparK pretrained)",
        "ssl_kd":   "SSL + KD\n(SparK + teacher)",
    }

    _n_rows = len(_saliency_models)
    fig, axes = plt.subplots(_n_rows, 4, figsize=(16, 4.4 * _n_rows))
    fig.suptitle(
        "FlashTorch Gradient Saliency — Heart MRI Axial Slice\n"
        "Which pixels drive the predicted heart (foreground) score?",
        fontsize=13, y=1.01,
    )
    _col_titles = ["Input + mask", "Vanilla gradients", "Guided backprop", "Gradient × Input"]
    _row_axes0  = axes[0] if _n_rows > 1 else axes
    for ax, ct in zip(_row_axes0, _col_titles):
        ax.set_title(ct, fontsize=11, fontweight="bold")

    for _row, (_mkey, _model) in enumerate(_saliency_models.items()):
        _adapter  = _SegSaliencyAdapter(_model)
        _backprop = _Backprop3D(_adapter)
        _rax      = axes[_row] if _n_rows > 1 else axes

        # Vanilla gradients
        try:
            _vgrad = _backprop.calculate_gradients(
                _input_2d.clone(), target_class=1, guided=False)
            _vg_np = _vgrad[0, 0].numpy() if _vgrad is not None else np.zeros_like(_slice_np)
        except Exception as _ex:
            print(f"  Vanilla backprop failed for {_mkey}: {_ex}")
            _vg_np = np.zeros_like(_slice_np)

        # Guided backprop (clips negative gradients at activation layers)
        try:
            _ggrad = _backprop.calculate_gradients(
                _input_2d.clone(), target_class=1, guided=True)
            _gg_np = _ggrad[0, 0].numpy() if _ggrad is not None else np.zeros_like(_slice_np)
        except Exception as _ex:
            print(f"  Guided backprop failed for {_mkey}: {_ex}")
            _gg_np = np.zeros_like(_slice_np)

        _gxi_np = np.abs(_vg_np * _slice_np)  # gradient × input (input attribution)

        # --- Column 0: input + heart mask overlay ---
        _rax[0].imshow(_slice_np.T, cmap="gray", origin="lower")
        _ov = np.zeros((*_slice_np.shape, 4))
        _ov[..., 0] = _mask_sl
        _ov[..., 3] = _mask_sl * 0.45
        _rax[0].imshow(_ov.transpose(1, 0, 2), origin="lower")
        _rax[0].set_ylabel(_method_labels.get(_mkey, _mkey), fontsize=11, fontweight="bold")
        _rax[0].axis("off")

        # --- Column 1: vanilla gradients ---
        _im1 = _rax[1].imshow(_vg_np.T, cmap="hot", origin="lower")
        plt.colorbar(_im1, ax=_rax[1], fraction=0.046, pad=0.04)
        _rax[1].axis("off")

        # --- Column 2: guided backprop ---
        _im2 = _rax[2].imshow(_gg_np.T, cmap="hot", origin="lower")
        plt.colorbar(_im2, ax=_rax[2], fraction=0.046, pad=0.04)
        _rax[2].axis("off")

        # --- Column 3: gradient × input attribution ---
        _v3 = np.abs(_gxi_np).max() or 1.0
        _im3 = _rax[3].imshow(_gxi_np.T, cmap="RdBu_r", origin="lower",
                               vmin=-_v3, vmax=_v3)
        plt.colorbar(_im3, ax=_rax[3], fraction=0.046, pad=0.04)
        _rax[3].axis("off")

    plt.tight_layout()
    fig.savefig(fig_dir / "flashtorch_saliency.png", dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Saved → {fig_dir / 'flashtorch_saliency.png'}")

elif not _EDA_OK:
    print("Saliency skipped — Heart MRI data not accessible.")
else:
    print("Saliency skipped — no trained checkpoints found (run Section 5 first).")

# %% [markdown]
# ### 8.3 — FlashTorch Activation Maximization (Encoder Filter Patterns)
#
# `GradientAscent` from FlashTorch optimises a random input image to maximally activate
# a chosen convolutional filter, revealing the *visual pattern* that filter has learned
# to detect. We target the first encoder Conv3d layer of each method and display the
# top-8 filter patterns.
#
# **What to look for:** Randomly-initialised models produce noisy, texture-like patterns.
# SSL-pretrained models tend to show sharper, more structured patterns (edges, blobs,
# curvature), reflecting the anatomy learned during masked pretraining.

# %%
def _get_first_conv(model: nn.Module) -> nn.Conv3d:
    """Return the first Conv3d in the model."""
    for m in model.modules():
        if isinstance(m, nn.Conv3d):
            return m
    raise ValueError("No Conv3d found in model.")


class _Conv3dTo2DWrapper(nn.Module):
    """Runs a single Conv3d on a 2D input [B, C, H, W] by adding a fake depth dim."""

    def __init__(self, conv3d: nn.Conv3d):
        super().__init__()
        self.conv = conv3d

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, H, W] → [B, C, 1, H, W]
        return self.conv(x.unsqueeze(2)).squeeze(2)  # [B, out_ch, H, W]


if _saliency_models:
    _ga_fig, _ga_axes = plt.subplots(len(_saliency_models), 8,
                                      figsize=(18, 3.2 * len(_saliency_models)))
    _ga_fig.suptitle(
        "FlashTorch Activation Maximization — First Encoder Conv3d Filter Patterns\n"
        "Patterns that maximally activate each of the first 8 filters",
        fontsize=13, y=1.01,
    )

    for _row, (_mkey, _model) in enumerate(_saliency_models.items()):
        _rax = _ga_axes[_row] if len(_saliency_models) > 1 else _ga_axes
        try:
            _first_conv   = _get_first_conv(_model)
            _conv_wrapper = _Conv3dTo2DWrapper(_first_conv)
            _ga = GradientAscent(_conv_wrapper, img_size=64, lr=1.0, use_gpu=False)
            _patterns = _ga.optimize(
                _conv_wrapper, filter_idxs=list(range(8)), num_iter=25)

            for _col, _pat in enumerate(_patterns[:8]):
                _ax = _rax[_col]
                _img = _pat[0].permute(1, 2, 0).detach().cpu().numpy()
                # single-channel MRI: show first channel
                _ax.imshow(_img[..., 0] if _img.ndim == 3 else _img,
                           cmap="gray", interpolation="nearest")
                _ax.set_title(f"Filter {_col}", fontsize=8)
                _ax.axis("off")

            _rax[0].set_ylabel(_method_labels.get(_mkey, _mkey),
                               fontsize=10, fontweight="bold")
        except Exception as _ex:
            print(f"  GradientAscent failed for {_mkey}: {_ex}")
            for _ax in _rax:
                _ax.axis("off")

    plt.tight_layout()
    _ga_fig.savefig(fig_dir / "flashtorch_activmax.png", dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Saved → {fig_dir / 'flashtorch_activmax.png'}")

# %% [markdown]
# ### 8.4 — Encoder Stage Activation Maps
#
# We register forward hooks on the first `Conv3d` at each of the 5 encoder stages
# (channels 32→64→128→256→512) and compare the mean activation magnitude between
# a **Baseline** model and an **SSL-only** model after seeing the same Heart MRI volume.
#
# - **Baseline**: activations are mostly noise in early stages → limited spatial structure.
# - **SSL-only**: pretraining on unlabeled data shapes activations to detect anatomy-relevant
#   edges and textures, visible as stronger, more spatially coherent responses.

# %%
def _extract_stage_activations(model: nn.Module, vol_5d: torch.Tensor) -> dict:
    """Register hooks on the first Conv3d at each depth and run one forward pass."""
    acts   = {}
    hooks  = []
    _count = [0]

    for _name, _mod in model.named_modules():
        if isinstance(_mod, nn.Conv3d) and _count[0] < 5:
            _s = _count[0]

            def _hook(_mod, _inp, _out, s=_s):
                acts[s] = _out.detach().cpu()

            hooks.append(_mod.register_forward_hook(_hook))
            _count[0] += 1

    with torch.no_grad():
        try:
            model(vol_5d)
        except Exception:
            pass  # volume may be smaller than required; partial activations are fine

    for h in hooks:
        h.remove()
    return acts


if _EDA_OK and _saliency_models:
    # Pad/crop volume to 96³ for safe inference
    from monai.transforms import Compose, EnsureChannelFirstd, NormalizeIntensityd, SpatialPadd
    import torch.nn.functional as F

    _vol_t = torch.tensor(_vnorm, dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # [1,1,H,W,D]
    # Pad to at least 96 in every spatial dim
    _pad   = [max(0, 96 - s) for s in _vol_t.shape[2:]]
    _vol_t = F.pad(_vol_t, [0, _pad[2], 0, _pad[1], 0, _pad[0]])  # [1,1,H',W',D']

    _compare = {k: _saliency_models[k]
                for k in ["baseline", "ssl_only"] if k in _saliency_models}

    if _compare:
        _n_comp  = len(_compare)
        _fig_act, _ax_act = plt.subplots(_n_comp, 5,
                                          figsize=(16, 4.2 * _n_comp))
        _fig_act.suptitle(
            "Encoder Stage Activation Maps — Baseline vs SSL-only\n"
            "Mean abs activation per stage (axial mid-slice, 5 encoder levels)",
            fontsize=13, y=1.01,
        )

        for _row, (_mkey, _model) in enumerate(_compare.items()):
            _stage_acts = _extract_stage_activations(_model, _vol_t)
            _rax = _ax_act[_row] if _n_comp > 1 else _ax_act

            for _col in range(5):
                _ax = _rax[_col]
                if _col not in _stage_acts:
                    _ax.axis("off")
                    continue
                _a = _stage_acts[_col]             # [1, ch, D, H, W] or [1, ch, H, W, D] …
                _a = _a[0].abs().mean(0)           # [D, H, W] — mean over channels
                # Take axial mid-slice along last dim (D)
                _sl = _a[:, :, _a.shape[2] // 2].numpy()
                _im = _ax.imshow(_sl.T, cmap="viridis", origin="lower")
                plt.colorbar(_im, ax=_ax, fraction=0.046, pad=0.04)
                _ax.set_title(
                    f"Stage {_col}  ({_a.shape[0]}ch)\n"
                    f"{list(_a.shape[1:])} voxels",
                    fontsize=9,
                )
                _ax.axis("off")

            _rax[0].set_ylabel(_method_labels.get(_mkey, _mkey),
                               fontsize=11, fontweight="bold")

        plt.tight_layout()
        _fig_act.savefig(fig_dir / "encoder_activations.png", dpi=150, bbox_inches="tight")
        plt.show()
        print(f"Saved → {fig_dir / 'encoder_activations.png'}")

# %%
# Upload Section 8 figures to WandB
if USE_WANDB and wandb.run is not None:
    for _f8 in ["flashtorch_saliency.png", "flashtorch_activmax.png",
                "encoder_activations.png"]:
        _fp = fig_dir / _f8
        if _fp.exists():
            save_checkpoint(_fp, f"viz-{_fp.stem}", "", "")
    print("Section 8 figures uploaded to WandB.")

print("\nSection 8 complete.")
print(f"Figures saved to: {fig_dir}")
print("  flashtorch_saliency.png  — gradient saliency maps per method")
print("  flashtorch_activmax.png  — activation maximization filter patterns")
print("  encoder_activations.png  — encoder stage activation maps")
