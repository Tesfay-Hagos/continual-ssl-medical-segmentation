# %% [markdown]
# # Semi-Supervised Skin Lesion Classification
# ## SSL Pretraining (SimCLR) + Mean Teacher — HAM10000
#
# **Research question:**
# Can domain-specific SSL pretraining + Mean Teacher semi-supervised fine-tuning
# close the gap to full supervision when only 1–20% of images are labeled?
#
# **Method pipeline:**
# 1. SimCLR pretraining on ALL images (no labels used)
# 2. Fine-tune SSL encoder on labeled subset only → Baseline-SSL
# 3. Mean Teacher: SSL encoder + consistency loss on unlabeled images → Main method
#
# **Experiments (3-fold CV × 4 label fractions):**
# | Method | Init | Labeled data | Loss |
# |---|---|---|---|
# | Baseline | ImageNet | Labeled only | CrossEntropy |
# | SSL-only | SimCLR | Labeled only | CrossEntropy |
# | SSL + Mean Teacher | SimCLR | Labeled + Unlabeled | CE + Consistency |
# | Upper bound | ImageNet | All labeled | CrossEntropy |
#
# **Dataset:** HAM10000 — 10,015 RGB dermoscopy images, 7 classes
# Kaggle: `kmader/skin-cancer-mnist-ham10000`
#
# ---
# ### Before running on Kaggle
# 1. Add dataset: `kmader/skin-cancer-mnist-ham10000`
# 2. Enable GPU (T4 or P100)
# 3. Add Kaggle secret: `HAM_10000`
# 4. Run all cells top-to-bottom

# %% [markdown]
# ## 0 — Environment Setup

# %%
import os
import sys
import subprocess
import json
import copy
import random
import warnings
from pathlib import Path

# Suppress pandas FutureWarning for groupby.apply with grouping columns
warnings.filterwarnings("ignore", category=FutureWarning, module="pandas")

ON_KAGGLE = os.path.exists("/kaggle/working")

REPO_URL = "https://github.com/Tesfay-Hagos/continual-ssl-medical-segmentation.git"
REPO_DIR = "/kaggle/working/project" if ON_KAGGLE else os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", ".."))
OUT_DIR  = "/kaggle/working/checkpoints/ham10000" if ON_KAGGLE else "/tmp/ham10000_ckpts"

print(f"ON_KAGGLE : {ON_KAGGLE}")
print(f"REPO_DIR  : {REPO_DIR}")
print(f"OUT_DIR   : {OUT_DIR}")

# %%
if ON_KAGGLE:
    if not os.path.exists(REPO_DIR):
        subprocess.run(["git", "clone", REPO_URL, REPO_DIR],
                       capture_output=True, text=True)
    else:
        subprocess.run(["git", "-C", REPO_DIR, "pull"],
                       capture_output=True, text=True)
    subprocess.run(["find", REPO_DIR, "-type", "d", "-name", "__pycache__",
                    "-exec", "rm", "-rf", "{}", "+"], capture_output=True)

# %%
if ON_KAGGLE:
    subprocess.run(
        [sys.executable, "-m", "pip", "install",
         "wandb", "scikit-learn", "pandas", "matplotlib", "seaborn",
         "Pillow", "flashtorch", "torchinfo", "--quiet"],
        check=True
    )
    print("Dependencies installed.")

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

import torchvision
from torchvision import transforms
from torchvision.models import efficientnet_b3, EfficientNet_B3_Weights
from torchvision.utils import make_grid
from torchvision.models.feature_extraction import create_feature_extractor

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (balanced_accuracy_score, confusion_matrix,
                              roc_auc_score, classification_report)
from sklearn.manifold import TSNE

import wandb

SRC_DIR = os.path.join(REPO_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from utils.storage import save_checkpoint, restore_checkpoint, set_wandb_entity

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED   = 42

# Reduce CUDA memory fragmentation (helps on T4 / P100 with limited VRAM)
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark     = False

print(f"PyTorch  : {torch.__version__}")
print(f"Device   : {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")

# ── Version control ───────────────────────────────────────────────────────────
# Bump RUN_VERSION to start a completely fresh run:
#   - new checkpoint directory (old results are never overwritten)
#   - new WandB project (separate dashboard, separate artifacts)
#   - all artifact names are versioned automatically
# v1 = 20-epoch SSL, batch=32, mt_lambda=1.0  (exploratory run)
# v2 = 40-epoch SSL, batch=64, mt_lambda=0.3  (tuned run)
RUN_VERSION   = "v2"
OUT_DIR       = f"{OUT_DIR}/{RUN_VERSION}" if not OUT_DIR.endswith(RUN_VERSION) else OUT_DIR
WANDB_PROJECT = f"ham10000-ssl-{RUN_VERSION}"

os.makedirs(OUT_DIR, exist_ok=True)

# %%
try:
    if ON_KAGGLE:
        from kaggle_secrets import UserSecretsClient
        _key = UserSecretsClient().get_secret("HAM_10000")
        wandb.login(key=_key)
    else:
        wandb.login()
    USE_WANDB = True
    _entity = wandb.Api().default_entity or ""
    set_wandb_entity(_entity)
    print(f"WandB ready. Project: {WANDB_PROJECT}")
except Exception as _e:
    USE_WANDB = False
    print(f"WandB unavailable ({_e}) — running without logging.")

# %% [markdown]
# ## 1 — Dataset Setup

# %%
# HAM10000 paths — Kaggle mounts the dataset at one of several locations
# depending on how it was added (via UI slug vs. full owner/dataset path).
if ON_KAGGLE:
    _candidates = [
        "/kaggle/input/skin-cancer-mnist-ham10000",               # added by slug
        "/kaggle/input/datasets/kmader/skin-cancer-mnist-ham10000", # added via full path
    ]
    DATA_ROOT = next(
        (p for p in _candidates if os.path.isdir(p)), _candidates[0]
    )
    print(f"Dataset root: {DATA_ROOT}  (exists={os.path.isdir(DATA_ROOT)})")
else:
    DATA_ROOT = os.environ.get("HAM10000_ROOT", "/data/ham10000")

META_CSV  = os.path.join(DATA_ROOT, "HAM10000_metadata.csv")
IMG_DIRS  = [
    os.path.join(DATA_ROOT, "HAM10000_images_part_1"),
    os.path.join(DATA_ROOT, "HAM10000_images_part_2"),
]

# Verify paths
assert os.path.exists(META_CSV), f"Metadata not found: {META_CSV}"
_found_dirs = [d for d in IMG_DIRS if os.path.isdir(d)]
assert _found_dirs, f"No image directories found. Checked: {IMG_DIRS}"
print(f"Metadata  : {META_CSV}")
print(f"Image dirs: {_found_dirs}")

# %%
CLASS_NAMES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]
NUM_CLASSES = len(CLASS_NAMES)
LABEL_MAP   = {c: i for i, c in enumerate(CLASS_NAMES)}
IDX_TO_CLS  = {i: c for c, i in LABEL_MAP.items()}

metadata = pd.read_csv(META_CSV)
# Drop duplicate image entries (same image_id, keep first)
metadata = metadata.drop_duplicates(subset="image_id").reset_index(drop=True)
metadata["label"] = metadata["dx"].map(LABEL_MAP)
assert metadata["label"].notna().all(), "Unknown class found in metadata"

print(f"Total unique images : {len(metadata)}")

# ── Stratified 3,000-image subset ────────────────────────────────────────────
# Keeps class proportions identical to the full dataset.
# Used for all training; SSL pretraining also uses only these 3K images.
# This fits comfortably on a T4 (15 GB) and trains in ~2.5 h.
SUBSET_SIZE = 3_000
metadata = (metadata.groupby("label", group_keys=False, observed=True)
            .apply(lambda g: g.sample(
                max(1, round(len(g) / len(metadata) * SUBSET_SIZE)),
                random_state=SEED))
            .reset_index(drop=True))
print(f"Subset size         : {len(metadata)}  (stratified from full 10,015)")

print(f"Classes ({NUM_CLASSES})         : {CLASS_NAMES}")
print("\nClass distribution (subset):")
for cls, grp in metadata.groupby("dx", observed=True):
    n   = len(grp)
    pct = n / len(metadata) * 100
    print(f"  {cls:<8} {n:>5}  ({pct:.1f}%)")

# %% [markdown]
# ## 1.5 — Exploratory Data Analysis

# %%
FIG_DIR = Path(OUT_DIR) / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# Build image path cache once
_path_cache: dict = {}
for _d in _found_dirs:
    for _p in Path(_d).glob("*.jpg"):
        _path_cache[_p.stem] = str(_p)

missing = [row.image_id for _, row in metadata.iterrows()
           if row.image_id not in _path_cache]
if missing:
    print(f"WARNING: {len(missing)} images listed in CSV not found on disk.")
else:
    print(f"All {len(metadata)} images found on disk.")

# %%
# --- Figure 1: Class distribution ---
fig, axes = plt.subplots(1, 2, figsize=(13, 4))

counts = metadata["dx"].value_counts().reindex(CLASS_NAMES)
colors = sns.color_palette("husl", NUM_CLASSES)

axes[0].bar(CLASS_NAMES, counts.values, color=colors, edgecolor="black", alpha=0.85)
axes[0].set_xlabel("Skin lesion class")
axes[0].set_ylabel("Number of images")
axes[0].set_title("HAM10000 — Class Distribution")
for i, v in enumerate(counts.values):
    axes[0].text(i, v + 30, str(v), ha="center", va="bottom", fontsize=9)
axes[0].grid(axis="y", alpha=0.3)

axes[1].pie(counts.values, labels=CLASS_NAMES, colors=colors,
            autopct="%1.1f%%", startangle=140,
            wedgeprops={"edgecolor": "black", "linewidth": 0.6})
axes[1].set_title("Class Share")

plt.tight_layout()
fig.savefig(FIG_DIR / "eda_class_dist.png", dpi=150, bbox_inches="tight")
plt.show()
print(f"Saved → {FIG_DIR / 'eda_class_dist.png'}")

# %%
# --- Figure 2: Sample image grid per class (torchvision make_grid) ---
_to_tensor = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])

_samples_per_class = 5
_grid_tensors = []
_grid_labels  = []

for cls in CLASS_NAMES:
    rows = metadata[metadata["dx"] == cls].sample(
        min(_samples_per_class, (metadata["dx"] == cls).sum()),
        random_state=SEED
    )
    for _, row in rows.iterrows():
        path = _path_cache.get(row["image_id"])
        if path:
            img = Image.open(path).convert("RGB")
            _grid_tensors.append(_to_tensor(img))
            _grid_labels.append(cls)

_grid = make_grid(torch.stack(_grid_tensors), nrow=_samples_per_class,
                  padding=4, normalize=False)

fig, ax = plt.subplots(figsize=(14, NUM_CLASSES * 2.2))
ax.imshow(_grid.permute(1, 2, 0).numpy())
ax.axis("off")
ax.set_title("HAM10000 — Sample Images per Class (5 per row)\n"
             + " | ".join(CLASS_NAMES), fontsize=12)
plt.tight_layout()
fig.savefig(FIG_DIR / "eda_sample_grid.png", dpi=150, bbox_inches="tight")
plt.show()
print(f"Saved → {FIG_DIR / 'eda_sample_grid.png'}")

# %%
# --- Figure 3: Age and sex distribution ---
fig, axes = plt.subplots(1, 2, figsize=(13, 4))

meta_clean = metadata.dropna(subset=["age"])
axes[0].hist(meta_clean["age"], bins=20, color="#5588cc", edgecolor="black", alpha=0.8)
axes[0].set_xlabel("Patient age")
axes[0].set_ylabel("Count")
axes[0].set_title("Age distribution")
axes[0].grid(alpha=0.3)

sex_counts = metadata["sex"].value_counts()
axes[1].bar(sex_counts.index, sex_counts.values,
            color=["#5588cc", "#cc8855", "#55cc88"][:len(sex_counts)],
            edgecolor="black", alpha=0.85)
axes[1].set_xlabel("Sex")
axes[1].set_ylabel("Count")
axes[1].set_title("Sex distribution")
axes[1].grid(axis="y", alpha=0.3)

plt.tight_layout()
fig.savefig(FIG_DIR / "eda_demographics.png", dpi=150, bbox_inches="tight")
plt.show()
print(f"Saved → {FIG_DIR / 'eda_demographics.png'}")

# %%
print("\n" + "=" * 55)
print("  EDA Summary — HAM10000 (3K stratified subset)")
print("=" * 55)
print(f"  Subset images   : {len(metadata)}  (of 10,015 total)")
print(f"  Classes         : {NUM_CLASSES}")
print(f"  Majority class  : nv  ({counts['nv']}, {counts['nv']/len(metadata)*100:.1f}%)")
print(f"  Minority class  : df  ({counts['df']}, {counts['df']/len(metadata)*100:.1f}%)")
print(f"  Imbalance ratio : {counts['nv'] / counts['df']:.0f}:1  (nv / df)")
print(f"  Age range       : {meta_clean['age'].min():.0f}–{meta_clean['age'].max():.0f} yrs")
print("=" * 55)
print("  → Use balanced accuracy (not overall accuracy) as primary metric.")
print("  → Use WeightedRandomSampler in training to handle imbalance.")

# %% [markdown]
# ## 2 — Dataset Class and Transforms

# %%
# ImageNet normalisation constants
_MEAN = [0.485, 0.456, 0.406]
_STD  = [0.229, 0.224, 0.225]

# Standard supervised transforms
IMG_SIZE = 160   # reduced from 224 to cut activation memory by ~50 % on T4

TRAIN_TRANSFORM = transforms.Compose([
    transforms.RandomResizedCrop(IMG_SIZE, scale=(0.7, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize(_MEAN, _STD),
])

VAL_TRANSFORM = transforms.Compose([
    transforms.Resize(int(IMG_SIZE * 1.14)),   # 182 → centre-crop to 160
    transforms.CenterCrop(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(_MEAN, _STD),
])

# SimCLR augmentation (stronger — two independent views)
SSL_TRANSFORM = transforms.Compose([
    transforms.RandomResizedCrop(IMG_SIZE, scale=(0.2, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomApply(
        [transforms.ColorJitter(brightness=0.8, contrast=0.8,
                                saturation=0.8, hue=0.2)], p=0.8),
    transforms.RandomGrayscale(p=0.2),
    transforms.RandomApply([transforms.GaussianBlur(kernel_size=15, sigma=(0.1, 2.0))],
                           p=0.5),   # kernel scaled proportionally to IMG_SIZE
    transforms.ToTensor(),
    transforms.Normalize(_MEAN, _STD),
])


class HAM10000Dataset(Dataset):
    """PyTorch Dataset for HAM10000.

    Parameters
    ----------
    df        : DataFrame subset (rows = images to include)
    transform : torchvision transform applied to each PIL image
    ssl_mode  : if True, returns two augmented views for SimCLR (ignores transform)
    """

    def __init__(self, df: pd.DataFrame, transform=None, ssl_mode: bool = False):
        self.df        = df.reset_index(drop=True)
        self.transform = transform
        self.ssl_mode  = ssl_mode

    def __len__(self) -> int:
        return len(self.df)

    def _load_pil(self, idx: int) -> Image.Image:
        img_id = self.df.iloc[idx]["image_id"]
        path   = _path_cache.get(img_id)
        if path is None:
            raise FileNotFoundError(f"Image '{img_id}' not in path cache.")
        return Image.open(path).convert("RGB")

    def __getitem__(self, idx: int):
        img   = self._load_pil(idx)
        label = int(self.df.iloc[idx]["label"])

        if self.ssl_mode:
            # Two independent augmented views — label is not used during SSL
            return SSL_TRANSFORM(img), SSL_TRANSFORM(img)

        if self.transform is not None:
            img = self.transform(img)
        return img, label


def make_weighted_sampler(df: pd.DataFrame) -> WeightedRandomSampler:
    """Oversample minority classes so each batch is approximately balanced."""
    labels  = df["label"].values
    counts  = np.bincount(labels, minlength=NUM_CLASSES).astype(float)
    weights = 1.0 / np.maximum(counts, 1.0)
    sample_weights = torch.tensor([weights[l] for l in labels], dtype=torch.float)
    return WeightedRandomSampler(sample_weights, num_samples=len(sample_weights),
                                 replacement=True)

# %% [markdown]
# ## 3 — Model Definitions

# %%
def build_efficientnet(pretrained: bool = True) -> tuple:
    """Return (backbone, feature_dim) for EfficientNet-B3.

    backbone : nn.Sequential(features, avgpool, Flatten) — no classifier head
    feature_dim : 1536
    """
    weights = EfficientNet_B3_Weights.IMAGENET1K_V1 if pretrained else None
    base    = efficientnet_b3(weights=weights)
    feature_dim = base.classifier[1].in_features  # 1536

    backbone = nn.Sequential(
        base.features,
        base.avgpool,
        nn.Flatten(),
    )
    return backbone, feature_dim


class HAM10000Classifier(nn.Module):
    """EfficientNet-B3 backbone + classification head for HAM10000."""

    def __init__(self, backbone: nn.Module, feature_dim: int,
                 num_classes: int = NUM_CLASSES, dropout: float = 0.3):
        super().__init__()
        self.backbone   = backbone
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(feature_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.backbone(x))

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Return backbone features (before classifier) — used for t-SNE."""
        return self.backbone(x)


class SimCLRModel(nn.Module):
    """SimCLR: backbone + 2-layer MLP projection head."""

    def __init__(self, backbone: nn.Module, feature_dim: int,
                 proj_hidden: int = 512, proj_out: int = 128):
        super().__init__()
        self.backbone  = backbone
        self.projector = nn.Sequential(
            nn.Linear(feature_dim, proj_hidden),
            nn.BatchNorm1d(proj_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(proj_hidden, proj_out),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.backbone(x)
        return F.normalize(self.projector(h), dim=1)


def nt_xent_loss(z1: torch.Tensor, z2: torch.Tensor,
                 temperature: float = 0.5) -> torch.Tensor:
    """Normalised temperature-scaled cross-entropy loss (SimCLR).

    z1, z2 : [B, D] — L2-normalised embeddings of two views.
    """
    B  = z1.size(0)
    z  = torch.cat([z1, z2], dim=0)          # [2B, D]
    # Cosine similarity matrix divided by temperature
    sim = torch.mm(z, z.T) / temperature     # [2B, 2B]
    # Mask out diagonal (self-similarity)
    mask = torch.eye(2 * B, dtype=torch.bool, device=z.device)
    sim  = sim.masked_fill(mask, float("-inf"))
    # Positive pair for z1[i] is z2[i] → index B+i, and vice versa
    labels = torch.cat([
        torch.arange(B, 2 * B, device=z.device),
        torch.arange(0, B,     device=z.device),
    ])
    return F.cross_entropy(sim, labels)


# %%
# --- Print model parameter summary ---
try:
    from torchinfo import summary as torchinfo_summary
    _bb, _fd = build_efficientnet(pretrained=False)
    _clf     = HAM10000Classifier(_bb, _fd)
    print("\nHAM10000Classifier (EfficientNet-B3 + head):")
    torchinfo_summary(_clf, input_size=(1, 3, 224, 224), device="cpu",
                      col_names=["output_size", "num_params"], depth=3, verbose=1)
except ImportError:
    _bb, _fd = build_efficientnet(pretrained=False)
    _clf     = HAM10000Classifier(_bb, _fd)
    total    = sum(p.numel() for p in _clf.parameters())
    backbone = sum(p.numel() for p in _clf.backbone.parameters())
    head     = sum(p.numel() for p in _clf.classifier.parameters())
    print(f"\nEfficientNet-B3 classifier  — total params: {total:,}")
    print(f"  Backbone : {backbone:,}  ({backbone/total*100:.1f}%)")
    print(f"  Head     : {head:,}      ({head/total*100:.1f}%)")
finally:
    del _bb, _fd, _clf

# %% [markdown]
# ## 4 — SimCLR Pretraining

# %%
SSL_CFG = {
    "epochs":      40,    # 40 epochs gives ~37K steps on 3K subset — enough to plateau
    "batch_size":  64,    # 64 doubles negatives per step vs 32 (better contrastive signal)
    "lr":          3e-4,
    "weight_decay": 1e-6,
    "temperature": 0.5,
    "num_workers": 4 if ON_KAGGLE else 0,
}

PRETRAIN_DIR  = Path(OUT_DIR) / "pretrain"
PRETRAIN_CKPT = PRETRAIN_DIR / "simclr_encoder.pth"
PRETRAIN_DONE = PRETRAIN_DIR / "pretrain_done.json"
PRETRAIN_DIR.mkdir(parents=True, exist_ok=True)

# Restore from WandB if available
restore_checkpoint("pretrain_done.json", PRETRAIN_DIR,
                   "ham10000-simclr-done", WANDB_PROJECT, "", "")
restore_checkpoint("simclr_encoder.pth", PRETRAIN_DIR,
                   "ham10000-simclr-encoder", WANDB_PROJECT, "", "")

# %%
if PRETRAIN_DONE.exists():
    _info = json.loads(PRETRAIN_DONE.read_text())
    print(f"SimCLR pretraining already complete "
          f"({_info['epochs']} epochs, best_loss={_info['best_loss']:.4f})")
    print("Delete pretrain_done.json to re-run.")
else:
    print("Starting SimCLR pretraining on all HAM10000 images (no labels used)...")

    ssl_dataset = HAM10000Dataset(metadata, transform=None, ssl_mode=True)
    ssl_loader  = DataLoader(
        ssl_dataset,
        batch_size  = SSL_CFG["batch_size"],
        shuffle     = True,
        num_workers = SSL_CFG["num_workers"],
        pin_memory  = (DEVICE.type == "cuda"),
        drop_last   = True,   # NT-Xent needs full batches
    )

    _backbone, _feature_dim = build_efficientnet(pretrained=True)
    ssl_model = SimCLRModel(_backbone, _feature_dim).to(DEVICE)

    ssl_optimizer = torch.optim.AdamW(
        ssl_model.parameters(),
        lr=SSL_CFG["lr"],
        weight_decay=SSL_CFG["weight_decay"],
    )
    ssl_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        ssl_optimizer, T_max=SSL_CFG["epochs"], eta_min=1e-6
    )

    if USE_WANDB:
        wandb.init(project=WANDB_PROJECT, name="simclr-pretrain", reinit=True,
                   config=SSL_CFG)

    best_ssl_loss = float("inf")
    for epoch in range(SSL_CFG["epochs"]):
        ssl_model.train()
        epoch_loss, n_batches = 0.0, 0

        for view1, view2 in ssl_loader:
            view1, view2 = view1.to(DEVICE), view2.to(DEVICE)
            ssl_optimizer.zero_grad()
            z1 = ssl_model(view1)
            z2 = ssl_model(view2)
            loss = nt_xent_loss(z1, z2, SSL_CFG["temperature"])
            loss.backward()
            ssl_optimizer.step()
            epoch_loss += loss.item()
            n_batches  += 1

        ssl_scheduler.step()
        avg = epoch_loss / max(n_batches, 1)
        lr  = ssl_scheduler.get_last_lr()[0]
        print(f"  SimCLR epoch {epoch+1:>3}/{SSL_CFG['epochs']}  "
              f"loss={avg:.4f}  lr={lr:.2e}")

        if USE_WANDB and wandb.run is not None:
            wandb.log({"simclr/loss": avg, "simclr/lr": lr, "simclr/epoch": epoch + 1})

        if avg < best_ssl_loss:
            best_ssl_loss = avg
            # Save encoder weights only (no projection head)
            torch.save(ssl_model.backbone.state_dict(), PRETRAIN_CKPT)

    done_info = {"epochs": SSL_CFG["epochs"], "best_loss": best_ssl_loss}
    PRETRAIN_DONE.write_text(json.dumps(done_info, indent=2))
    save_checkpoint(PRETRAIN_CKPT,  "ham10000-simclr-encoder", "", "")
    save_checkpoint(PRETRAIN_DONE,  "ham10000-simclr-done",    "", "")
    print(f"SimCLR complete. Best loss: {best_ssl_loss:.4f}  Encoder: {PRETRAIN_CKPT}")

    if USE_WANDB and wandb.run is not None:
        wandb.finish()

# Free GPU memory before fine-tuning phase
if DEVICE.type == "cuda":
    torch.cuda.empty_cache()
    print(f"GPU memory after SSL: "
          f"{torch.cuda.memory_allocated()/1e9:.2f} GB allocated, "
          f"{torch.cuda.memory_reserved()/1e9:.2f} GB reserved")

# %% [markdown]
# ## 5 — Shared Training Helpers

# %%
FINETUNE_CFG = {
    "epochs":        25,    # extra 5 epochs — MT needs more time to benefit
    "warmup_epochs":  2,
    "batch_size":    16,    # reduced from 32 to fit T4 VRAM
    "lr":           2e-4,
    "weight_decay": 1e-4,
    "patience":     12,     # more patience — MT gains come late in training
    "num_workers":  4 if ON_KAGGLE else 0,
    # Mean Teacher — λ=0.3 avoids consistency loss overwhelming supervised signal
    # when the teacher is still noisy early in training
    "ema_alpha":    0.999,
    "mt_lambda":    0.3,    # reduced from 1.0 — supervised loss stays dominant
    "mt_rampup":    15,     # slower ramp (was 8) — teacher needs time to stabilise
}

LABEL_FRACTIONS = [0.05, 0.10, 0.20]   # dropped 1 % (too few samples at 3K subset)
N_FOLDS         = 3


def build_classifier(pretrained_backbone: bool = True,
                     load_simclr: bool = False) -> HAM10000Classifier:
    """Build an EfficientNet-B3 classifier.

    pretrained_backbone : use ImageNet weights
    load_simclr         : overwrite backbone with SimCLR encoder
    """
    backbone, feature_dim = build_efficientnet(pretrained=pretrained_backbone)
    model = HAM10000Classifier(backbone, feature_dim).to(DEVICE)

    if load_simclr and PRETRAIN_CKPT.exists():
        state   = torch.load(PRETRAIN_CKPT, map_location=DEVICE)
        missing, unexpected = model.backbone.load_state_dict(state, strict=False)
        n_loaded = len(state) - len(missing)
        print(f"  SimCLR encoder loaded: {n_loaded}/{len(state)} keys "
              f"(missing={len(missing)}, unexpected={len(unexpected)})")
    elif load_simclr:
        print("  WARNING: SimCLR checkpoint not found — using ImageNet init.")

    return model


def make_scheduler(optimizer, cfg: dict):
    warmup = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.1, end_factor=1.0,
        total_iters=cfg["warmup_epochs"]
    )
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, cfg["epochs"] - cfg["warmup_epochs"]), eta_min=1e-6
    )
    return torch.optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=[warmup, cosine],
        milestones=[cfg["warmup_epochs"]]
    )


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader) -> dict:
    """Evaluate a model. Returns balanced_accuracy, auc_macro, per-class probs."""
    model.eval()
    all_preds, all_probs, all_labels = [], [], []

    for imgs, labels in loader:
        imgs   = imgs.to(DEVICE)
        logits = model(imgs)
        probs  = F.softmax(logits, dim=1).cpu().numpy()
        preds  = logits.argmax(dim=1).cpu().numpy()
        all_probs.append(probs)
        all_preds.extend(preds)
        all_labels.extend(labels.numpy())

    all_probs  = np.concatenate(all_probs, axis=0)   # [N, C]
    all_preds  = np.array(all_preds)
    all_labels = np.array(all_labels)

    bal_acc = balanced_accuracy_score(all_labels, all_preds)
    try:
        auc = roc_auc_score(all_labels, all_probs, multi_class="ovr", average="macro")
    except ValueError:
        auc = float("nan")

    return {"balanced_acc": bal_acc, "auc_macro": auc,
            "preds": all_preds, "labels": all_labels, "probs": all_probs}


@torch.no_grad()
def update_ema(student: nn.Module, teacher: nn.Module, alpha: float):
    """Exponential moving average update: θ_t ← α·θ_t + (1−α)·θ_s."""
    for s_p, t_p in zip(student.parameters(), teacher.parameters()):
        t_p.data.mul_(alpha).add_(s_p.data, alpha=1.0 - alpha)


def mt_rampup_weight(epoch: int, rampup_epochs: int, max_lambda: float) -> float:
    """Sigmoid ramp-up for consistency loss weight (Laine & Aila 2017 style)."""
    if rampup_epochs == 0:
        return max_lambda
    p = max(0.0, min(1.0, epoch / rampup_epochs))
    return max_lambda * float(np.exp(-5.0 * (1.0 - p) ** 2))

# %% [markdown]
# ## 6 — Fine-tuning and Mean Teacher Training Loop

# %%
def finetune(run_name: str, model: HAM10000Classifier,
             train_loader: DataLoader, val_loader: DataLoader,
             teacher: HAM10000Classifier = None,
             unlabeled_loader: DataLoader = None) -> dict:
    """Unified fine-tuning loop.

    If teacher and unlabeled_loader are provided → Mean Teacher mode.
    Otherwise → standard supervised fine-tuning.
    """
    ckpt_dir = Path(OUT_DIR) / run_name
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    resume_path = ckpt_dir / "latest.pth"
    best_path   = ckpt_dir / "best.pth"

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=FINETUNE_CFG["lr"],
        weight_decay=FINETUNE_CFG["weight_decay"],
    )
    scheduler = make_scheduler(optimizer, FINETUNE_CFG)
    scaler    = torch.amp.GradScaler(DEVICE.type,
                                     enabled=(DEVICE.type == "cuda"))

    best_acc    = 0.0
    best_auc    = 0.0
    trigger     = 0
    start_epoch = 0
    use_mt      = (teacher is not None) and (unlabeled_loader is not None)

    if use_mt:
        # Initialise teacher as copy of student
        teacher = copy.deepcopy(model).to(DEVICE)
        teacher.eval()
        for p in teacher.parameters():
            p.requires_grad = False
        print(f"  Mean Teacher enabled (α={FINETUNE_CFG['ema_alpha']}, "
              f"λ_max={FINETUNE_CFG['mt_lambda']})")

    if resume_path.exists():
        state       = torch.load(resume_path, map_location=DEVICE)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        start_epoch = state["epoch"] + 1
        best_acc    = state["best_acc"]
        trigger     = state["trigger"]
        if use_mt and "teacher" in state:
            teacher.load_state_dict(state["teacher"])
        print(f"  Resumed {run_name} from epoch {start_epoch}, "
              f"best_acc={best_acc:.4f}")

    unlabeled_iter = iter(unlabeled_loader) if use_mt else None

    for epoch in range(start_epoch, FINETUNE_CFG["epochs"]):
        model.train()
        if use_mt:
            teacher.eval()

        ep_loss, ep_ce, ep_mt, n_steps = 0.0, 0.0, 0.0, 0
        mt_w = mt_rampup_weight(epoch, FINETUNE_CFG["mt_rampup"],
                                FINETUNE_CFG["mt_lambda"])

        for imgs, labels in train_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()

            with torch.amp.autocast(device_type=DEVICE.type):
                logits = model(imgs)
                ce_loss = criterion(logits, labels)
                loss    = ce_loss

                if use_mt:
                    # Fetch unlabeled batch (cycle iterator)
                    try:
                        u_imgs, _ = next(unlabeled_iter)
                    except StopIteration:
                        unlabeled_iter = iter(unlabeled_loader)
                        u_imgs, _ = next(unlabeled_iter)

                    u_imgs = u_imgs.to(DEVICE)
                    s_probs = F.softmax(model(u_imgs),   dim=1)
                    with torch.no_grad():
                        t_probs = F.softmax(teacher(u_imgs), dim=1).detach()

                    mt_loss = F.mse_loss(s_probs, t_probs)
                    loss    = ce_loss + mt_w * mt_loss
                    ep_mt  += mt_loss.item()

            if not torch.isfinite(loss):
                optimizer.zero_grad()
                continue

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

            if use_mt:
                update_ema(model, teacher, FINETUNE_CFG["ema_alpha"])

            ep_loss += loss.item()
            ep_ce   += ce_loss.item()
            n_steps += 1

        scheduler.step()
        metrics = evaluate(model, val_loader)
        acc, auc = metrics["balanced_acc"], metrics["auc_macro"]
        n = max(n_steps, 1)
        lr = scheduler.get_last_lr()[0]

        print(f"  [{run_name}] ep {epoch+1:>3}/{FINETUNE_CFG['epochs']}  "
              f"loss={ep_loss/n:.4f}  ce={ep_ce/n:.4f}  "
              + (f"mt={ep_mt/n:.4f}  " if use_mt else "")
              + f"bal_acc={acc:.4f}  auc={auc:.4f}  "
              f"best={best_acc:.4f}  lr={lr:.2e}")

        if USE_WANDB and wandb.run is not None:
            wandb.log({f"{run_name}/loss": ep_loss/n,
                       f"{run_name}/ce_loss": ep_ce/n,
                       f"{run_name}/mt_loss": ep_mt/n if use_mt else 0,
                       f"{run_name}/bal_acc": acc,
                       f"{run_name}/auc": auc,
                       f"{run_name}/epoch": epoch + 1})

        if acc > best_acc:
            best_acc, best_auc = acc, auc
            torch.save(model.state_dict(), best_path)
            save_checkpoint(best_path, f"{run_name}-best", "", "")
            trigger = 0
        else:
            trigger += 1
            if trigger >= FINETUNE_CFG["patience"]:
                print(f"  Early stopping at epoch {epoch+1}.")
                break

        state_dict = {
            "model": model.state_dict(), "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(), "epoch": epoch,
            "best_acc": best_acc, "trigger": trigger,
        }
        if use_mt:
            state_dict["teacher"] = teacher.state_dict()
        torch.save(state_dict, resume_path)

    if USE_WANDB and wandb.run is not None:
        wandb.log({f"{run_name}/best_bal_acc": best_acc,
                   f"{run_name}/best_auc": best_auc})
    print(f"  [{run_name}] Done.  best_bal_acc={best_acc:.4f}  best_auc={best_auc:.4f}")
    return {"run": run_name, "best_bal_acc": best_acc, "best_auc": best_auc,
            "ckpt": str(best_path)}

# %% [markdown]
# ## 7 — Cross-Validation Experiments
#
# 3-fold CV × 4 label fractions × 4 methods = 48 runs.
# Label fraction = fraction of the training fold used as labeled data.
# The remaining training images are unlabeled (used by Mean Teacher only).

# %%
skf   = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
folds = list(skf.split(metadata, metadata["label"]))

CV_RESULTS_FILE = Path(OUT_DIR) / "ham10000_cv_results.json"
cv_results: dict = {}

# Restore previous results
restore_checkpoint("ham10000_cv_results.json", Path(OUT_DIR),
                   "ham10000-cv-results", WANDB_PROJECT, "", "")
if CV_RESULTS_FILE.exists():
    cv_results = json.loads(CV_RESULTS_FILE.read_text())
    print(f"Restored cv_results: {sorted(cv_results.keys())}")


def save_cv():
    CV_RESULTS_FILE.write_text(json.dumps(cv_results, indent=2))
    if USE_WANDB and wandb.run is not None:
        save_checkpoint(CV_RESULTS_FILE, "ham10000-cv-results", "", "")


def get_loaders(fold_idx: int, label_frac: float):
    """Build labeled, unlabeled, and val DataLoaders for one fold + label fraction."""
    train_idx, val_idx = folds[fold_idx]
    train_df = metadata.iloc[train_idx].reset_index(drop=True)
    val_df   = metadata.iloc[val_idx].reset_index(drop=True)

    # Stratified labeled/unlabeled split within training fold
    _sampled = (train_df.groupby("label", group_keys=False, observed=True)
                .apply(lambda g: g.sample(
                    min(len(g), max(1, round(len(g) * label_frac))),
                    random_state=SEED)))
    # groupby.apply can produce MultiIndex in pandas 2.x — flatten it
    if isinstance(_sampled.index, pd.MultiIndex):
        _sampled = _sampled.droplevel(0)
    labeled_df   = _sampled.reset_index(drop=True)
    unlabeled_df = train_df.drop(_sampled.index, errors="ignore").reset_index(drop=True)

    labeled_ds = HAM10000Dataset(labeled_df, transform=TRAIN_TRANSFORM)
    val_ds     = HAM10000Dataset(val_df,     transform=VAL_TRANSFORM)

    sampler      = make_weighted_sampler(labeled_df)
    labeled_loader = DataLoader(labeled_ds, batch_size=FINETUNE_CFG["batch_size"],
                                sampler=sampler,
                                num_workers=FINETUNE_CFG["num_workers"],
                                pin_memory=(DEVICE.type == "cuda"))
    val_loader     = DataLoader(val_ds, batch_size=64, shuffle=False,
                                num_workers=FINETUNE_CFG["num_workers"])

    # unlabeled_loader is None when label_frac=1.0 (upper bound — no unlabeled data)
    if len(unlabeled_df) > 0:
        unlabeled_ds     = HAM10000Dataset(unlabeled_df, transform=TRAIN_TRANSFORM)
        unlabeled_loader = DataLoader(unlabeled_ds, batch_size=FINETUNE_CFG["batch_size"],
                                      shuffle=True,
                                      num_workers=FINETUNE_CFG["num_workers"],
                                      pin_memory=(DEVICE.type == "cuda"),
                                      drop_last=True)
    else:
        unlabeled_loader = None

    print(f"  Fold {fold_idx+1}  label_frac={label_frac:.0%}  "
          f"labeled={len(labeled_df)}  unlabeled={len(unlabeled_df)}  "
          f"val={len(val_df)}")
    return labeled_loader, unlabeled_loader, val_loader


# %%
# Main experiment loop
for fold in range(N_FOLDS):
    for label_frac in LABEL_FRACTIONS:
        frac_key  = f"{int(label_frac*100)}pct"
        fold_key  = f"fold_{fold}_{frac_key}"

        if fold_key not in cv_results:
            cv_results[fold_key] = {}

        labeled_loader, unlabeled_loader, val_loader = get_loaders(fold, label_frac)

        print(f"\n{'='*65}")
        print(f"FOLD {fold+1}/{N_FOLDS}  |  label_frac={label_frac:.0%}")
        print(f"{'='*65}")

        # ── Baseline ────────────────────────────────────────────────────
        if "baseline" not in cv_results[fold_key]:
            run_name = f"baseline_f{fold+1}_{frac_key}"
            if USE_WANDB:
                wandb.init(project=WANDB_PROJECT, name=run_name, reinit=True,
                           config={**FINETUNE_CFG, "fold": fold+1,
                                   "label_frac": label_frac, "method": "baseline"})
            try:
                print("\n--- Baseline (ImageNet init, labeled only) ---")
                model = build_classifier(pretrained_backbone=True, load_simclr=False)
                cv_results[fold_key]["baseline"] = finetune(
                    run_name, model, labeled_loader, val_loader)
                save_cv()
            finally:
                if USE_WANDB and wandb.run is not None:
                    wandb.finish()
        else:
            print(f"Fold {fold+1} {frac_key} baseline already done — skipping.")

        # ── SSL-only ─────────────────────────────────────────────────────
        if "ssl_only" not in cv_results[fold_key]:
            run_name = f"ssl_only_f{fold+1}_{frac_key}"
            if USE_WANDB:
                wandb.init(project=WANDB_PROJECT, name=run_name, reinit=True,
                           config={**FINETUNE_CFG, "fold": fold+1,
                                   "label_frac": label_frac, "method": "ssl_only"})
            try:
                print("\n--- SSL-only (SimCLR init, labeled only) ---")
                model = build_classifier(pretrained_backbone=True, load_simclr=True)
                cv_results[fold_key]["ssl_only"] = finetune(
                    run_name, model, labeled_loader, val_loader)
                save_cv()
            finally:
                if USE_WANDB and wandb.run is not None:
                    wandb.finish()
        else:
            print(f"Fold {fold+1} {frac_key} ssl_only already done — skipping.")

        # ── SSL + Mean Teacher ────────────────────────────────────────────
        if "ssl_mt" not in cv_results[fold_key]:
            run_name = f"ssl_mt_f{fold+1}_{frac_key}"
            if USE_WANDB:
                wandb.init(project=WANDB_PROJECT, name=run_name, reinit=True,
                           config={**FINETUNE_CFG, "fold": fold+1,
                                   "label_frac": label_frac, "method": "ssl_mt"})
            try:
                print("\n--- SSL + Mean Teacher (SimCLR init + unlabeled consistency) ---")
                model   = build_classifier(pretrained_backbone=True, load_simclr=True)
                teacher = build_classifier(pretrained_backbone=True, load_simclr=True)
                cv_results[fold_key]["ssl_mt"] = finetune(
                    run_name, model, labeled_loader, val_loader,
                    teacher=teacher, unlabeled_loader=unlabeled_loader)
                save_cv()
            finally:
                if USE_WANDB and wandb.run is not None:
                    wandb.finish()
        else:
            print(f"Fold {fold+1} {frac_key} ssl_mt already done — skipping.")

        # ── Upper bound (run once at 10 % label fraction per fold) ───────
        if frac_key == "10pct" and "upper_bound" not in cv_results[fold_key]:
            run_name = f"upper_bound_f{fold+1}"
            if USE_WANDB:
                wandb.init(project=WANDB_PROJECT, name=run_name, reinit=True,
                           config={**FINETUNE_CFG, "fold": fold+1,
                                   "label_frac": 1.0, "method": "upper_bound"})
            try:
                print("\n--- Upper bound (ImageNet init, all labels) ---")
                ub_loader, _, _ = get_loaders(fold, label_frac=1.0)
                model = build_classifier(pretrained_backbone=True, load_simclr=False)
                cv_results[fold_key]["upper_bound"] = finetune(
                    run_name, model, ub_loader, val_loader)
                save_cv()
            finally:
                if USE_WANDB and wandb.run is not None:
                    wandb.finish()

# %% [markdown]
# ## 8 — Results Analysis

# %%
def compute_stats(cv_results: dict, method: str, label_frac: float,
                  metric: str = "best_bal_acc") -> dict:
    """Mean ± std for a method × label_frac across all folds."""
    frac_key = f"{int(label_frac*100)}pct"
    vals = []
    for fold in range(N_FOLDS):
        fold_key = f"fold_{fold}_{frac_key}"
        v = cv_results.get(fold_key, {}).get(method, {}).get(metric, None)
        if v is not None and not np.isnan(v):
            vals.append(v)
    if not vals:
        return {"mean": float("nan"), "std": float("nan"), "n": 0}
    return {"mean": float(np.mean(vals)),
            "std":  float(np.std(vals, ddof=1) if len(vals) > 1 else 0.0),
            "n":    len(vals)}


# Print main results table (10% label fraction)
methods = {
    "baseline":    "Baseline (ImageNet, labeled only)",
    "ssl_only":    "SSL-only (SimCLR, labeled only)",
    "ssl_mt":      "SSL + Mean Teacher",
    "upper_bound": "Supervised upper bound (100%)",
}

print("\n" + "=" * 75)
print("  HAM10000 Results — 10% Labeled (3-fold CV)")
print("=" * 75)
print(f"  {'Method':<40} {'Bal. Acc (mean±std)':<22} {'AUC (mean±std)'}")
print("-" * 75)

for key, label in methods.items():
    acc_s = compute_stats(cv_results, key, 0.10, "best_bal_acc")
    auc_s = compute_stats(cv_results, key, 0.10, "best_auc")
    acc_str = (f"{acc_s['mean']:.3f} ± {acc_s['std']:.3f}"
               if acc_s["n"] > 0 else "N/A")
    auc_str = (f"{auc_s['mean']:.3f} ± {auc_s['std']:.3f}"
               if auc_s["n"] > 0 else "N/A")
    print(f"  {label:<40} {acc_str:<22} {auc_str}")
print("=" * 75)

# Label fraction ablation
print("\nLabel fraction ablation — SSL + Mean Teacher:")
print(f"  {'Label %':<12} {'Bal. Acc':<22} {'AUC'}")
print("-" * 55)
for lf in LABEL_FRACTIONS:
    s_acc = compute_stats(cv_results, "ssl_mt", lf, "best_bal_acc")
    s_auc = compute_stats(cv_results, "ssl_mt", lf, "best_auc")
    acc_str = f"{s_acc['mean']:.3f} ± {s_acc['std']:.3f}" if s_acc["n"] else "N/A"
    auc_str = f"{s_auc['mean']:.3f} ± {s_auc['std']:.3f}" if s_auc["n"] else "N/A"
    print(f"  {lf*100:<5.0f}%       {acc_str:<22} {auc_str}")

# %% [markdown]
# ## 9 — Publication Figures

# %%
try:
    plt.style.use("seaborn-v0_8-whitegrid")
except OSError:
    plt.style.use("seaborn-whitegrid")
plt.rcParams.update({"font.size": 12, "axes.titlesize": 13,
                     "axes.labelsize": 12, "figure.titlesize": 14})

method_labels = ["Baseline", "SSL-only", "SSL + MT", "Upper Bound"]
method_keys   = ["baseline", "ssl_only", "ssl_mt", "upper_bound"]
colors        = ["#e07070", "#70a0e0", "#70c870", "#c070c0"]

# ── Figure A: Main comparison bar chart ─────────────────────────────────────
acc_means = [compute_stats(cv_results, k, 0.10)["mean"] for k in method_keys]
acc_stds  = [compute_stats(cv_results, k, 0.10)["std"]  for k in method_keys]
auc_means = [compute_stats(cv_results, k, 0.10, "best_auc")["mean"] for k in method_keys]
auc_stds  = [compute_stats(cv_results, k, 0.10, "best_auc")["std"]  for k in method_keys]

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle("HAM10000 — 10% Labeled Data (3-fold CV)", fontsize=14)

for ax, means, stds, ylabel, title in [
    (axes[0], acc_means, acc_stds, "Balanced Accuracy", "Balanced Accuracy"),
    (axes[1], auc_means, auc_stds, "AUC (macro)", "AUC (macro-averaged)"),
]:
    bars = ax.bar(method_labels, means, yerr=stds, capsize=6,
                  color=colors, alpha=0.85, edgecolor="black", linewidth=0.8)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", alpha=0.4)
    for bar, m, s in zip(bars, means, stds):
        if not np.isnan(m):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + s + 0.01,
                    f"{m:.3f}", ha="center", va="bottom",
                    fontsize=9, fontweight="bold")

plt.tight_layout()
fig.savefig(FIG_DIR / "results_comparison.png", dpi=300, bbox_inches="tight")
fig.savefig(FIG_DIR / "results_comparison.pdf", bbox_inches="tight")
plt.show()
print(f"Saved → {FIG_DIR / 'results_comparison.png'}")

# ── Figure B: Label fraction ablation curve ──────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 5))

for key, label, color in [
    ("baseline", "Baseline (ImageNet)", "#e07070"),
    ("ssl_only", "SSL-only (SimCLR)",   "#70a0e0"),
    ("ssl_mt",   "SSL + Mean Teacher",  "#70c870"),
]:
    means, stds, xs = [], [], []
    for lf in LABEL_FRACTIONS:
        s = compute_stats(cv_results, key, lf)
        if s["n"] > 0:
            xs.append(lf * 100)
            means.append(s["mean"])
            stds.append(s["std"])
    if xs:
        xs, means, stds = np.array(xs), np.array(means), np.array(stds)
        ax.plot(xs, means, marker="o", label=label, color=color, linewidth=2)
        ax.fill_between(xs, means - stds, means + stds, alpha=0.15, color=color)

ax.set_xlabel("Labeled data (%)")
ax.set_ylabel("Balanced Accuracy")
ax.set_title("Performance vs. Label Fraction — SSL gain at low labels")
ax.legend(fontsize=11)
ax.grid(alpha=0.4)
ax.set_xlim(0, 22)
plt.tight_layout()
fig.savefig(FIG_DIR / "label_fraction_ablation.png", dpi=300, bbox_inches="tight")
fig.savefig(FIG_DIR / "label_fraction_ablation.pdf", bbox_inches="tight")
plt.show()
print(f"Saved → {FIG_DIR / 'label_fraction_ablation.png'}")

# %% [markdown]
# ## 10 — Visualizations: GradCAM, t-SNE, Confusion Matrix, FlashTorch

# %%
# ── GradCAM implementation using torchvision feature extraction + hooks ──────

class GradCAM:
    """Gradient-weighted Class Activation Mapping for EfficientNet-B3.

    Hooks into the last convolutional block (model.backbone[0][-1]) and
    computes the weighted activation map for any target class.
    """

    def __init__(self, model: HAM10000Classifier):
        self.model       = model
        self._activations = None
        self._gradients   = None

        target_layer = model.backbone[0][-1]   # last block in EfficientNet features
        target_layer.register_forward_hook(self._save_activation)
        target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, inp, out):
        self._activations = out.detach()

    def _save_gradient(self, module, grad_in, grad_out):
        self._gradients = grad_out[0].detach()

    def __call__(self, img_tensor: torch.Tensor,
                 class_idx: int = None) -> tuple:
        """Return (cam_np, predicted_class).

        img_tensor : [C, H, W] single image, already normalised.
        """
        self.model.eval()
        x = img_tensor.unsqueeze(0).to(DEVICE).requires_grad_(True)

        self.model.zero_grad()
        logits = self.model(x)
        if class_idx is None:
            class_idx = logits.argmax(dim=1).item()

        logits[0, class_idx].backward()

        weights = self._gradients.mean(dim=(-2, -1), keepdim=True)  # [1, C, 1, 1]
        cam     = (weights * self._activations).sum(dim=1)           # [1, H, W]
        cam     = F.relu(cam).squeeze(0).cpu().numpy()               # [H, W]
        cam     = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam, class_idx


def unnormalise(tensor: torch.Tensor) -> np.ndarray:
    """Reverse ImageNet normalisation → [H, W, 3] uint8."""
    mean = torch.tensor(_MEAN).view(3, 1, 1)
    std  = torch.tensor(_STD).view(3, 1, 1)
    img  = (tensor * std + mean).clamp(0, 1)
    return (img.permute(1, 2, 0).numpy() * 255).astype(np.uint8)


# ──  Load best checkpoints from fold 0, 10% label fraction ──────────────────
_viz_models: dict = {}
for _mkey in ["baseline", "ssl_only", "ssl_mt"]:
    _ckpt = cv_results.get("fold_0_10pct", {}).get(_mkey, {}).get("ckpt", "")
    if _ckpt and Path(_ckpt).exists():
        _m = build_classifier(pretrained_backbone=False, load_simclr=False)
        _m.load_state_dict(torch.load(_ckpt, map_location=DEVICE))
        _m.eval()
        _viz_models[_mkey] = _m
        print(f"  Loaded {_mkey} checkpoint for visualisation.")
    else:
        print(f"  {_mkey}: checkpoint not found — skipping visualisation.")

# ── Figure C: GradCAM comparison ─────────────────────────────────────────────
if _viz_models:
    _n_samples = 4
    _viz_labels_map = {"baseline": "Baseline", "ssl_only": "SSL-only",
                       "ssl_mt":   "SSL + MT"}
    _sample_rows = (metadata.groupby("dx", observed=True)
                    .apply(lambda g: g.sample(min(1, len(g)), random_state=SEED))
                    .reset_index(drop=True)
                    .head(_n_samples))

    n_rows = len(_viz_models)
    n_cols = len(_sample_rows) * 2  # image + cam per sample
    fig, axes = plt.subplots(n_rows, n_cols,
                              figsize=(n_cols * 2.2, n_rows * 2.5))
    fig.suptitle("GradCAM — What each model attends to\n"
                 "(left: input image, right: gradient activation map)",
                 fontsize=13, y=1.01)

    for row, (mkey, model) in enumerate(_viz_models.items()):
        gradcam = GradCAM(model)
        for col_pair, (_, sample) in enumerate(_sample_rows.iterrows()):
            path = _path_cache.get(sample["image_id"])
            if path is None:
                continue
            img_pil = Image.open(path).convert("RGB")
            img_t   = VAL_TRANSFORM(img_pil)
            cam, pred = gradcam(img_t)

            img_np = unnormalise(img_t)
            cam_resized = np.array(
                Image.fromarray((cam * 255).astype(np.uint8)).resize(
                    (img_np.shape[1], img_np.shape[0]),
                    Image.BILINEAR
                )
            ).astype(float) / 255.0

            # Overlay: jet colormap blended on original image
            heatmap = plt.cm.jet(cam_resized)[..., :3]
            overlay = (0.55 * img_np / 255.0 + 0.45 * heatmap).clip(0, 1)

            col_img = col_pair * 2
            col_cam = col_pair * 2 + 1
            ax_img  = axes[row, col_img] if n_rows > 1 else axes[col_img]
            ax_cam  = axes[row, col_cam] if n_rows > 1 else axes[col_cam]

            ax_img.imshow(img_np)
            ax_img.set_title(f"{sample['dx']}", fontsize=8)
            ax_img.axis("off")

            ax_cam.imshow(overlay)
            ax_cam.set_title(f"pred: {IDX_TO_CLS.get(pred, '?')}", fontsize=8)
            ax_cam.axis("off")

        row_ax = axes[row, 0] if n_rows > 1 else axes[0]
        row_ax.set_ylabel(_viz_labels_map.get(mkey, mkey),
                          fontsize=11, fontweight="bold")

    plt.tight_layout()
    fig.savefig(FIG_DIR / "gradcam_comparison.png", dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Saved → {FIG_DIR / 'gradcam_comparison.png'}")

# ── Figure D: Confusion matrix (best method at 10% labels) ───────────────────
if "ssl_mt" in _viz_models:
    _, _, val_loader_viz = get_loaders(0, 0.10)
    metrics_viz = evaluate(_viz_models["ssl_mt"], val_loader_viz)
    cm = confusion_matrix(metrics_viz["labels"], metrics_viz["preds"],
                          normalize="true")

    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(cm, annot=True, fmt=".2f", cmap="Blues",
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
                linewidths=0.5, ax=ax)
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")
    ax.set_title("Confusion Matrix — SSL + Mean Teacher (10% labels, fold 1)")
    plt.tight_layout()
    fig.savefig(FIG_DIR / "confusion_matrix.png", dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Saved → {FIG_DIR / 'confusion_matrix.png'}")

# ── Figure E: t-SNE feature space comparison ────────────────────────────────
if _viz_models:
    _, _, val_loader_tsne = get_loaders(0, 0.10)

    fig, axes = plt.subplots(1, len(_viz_models),
                              figsize=(6.5 * len(_viz_models), 5.5))
    fig.suptitle("t-SNE of learned features — validation set\n"
                 "(colour = true class)", fontsize=13)

    _tsne_colors = np.array(sns.color_palette("tab10", NUM_CLASSES))

    for col, (mkey, model) in enumerate(_viz_models.items()):
        feats, lbls = [], []
        model.eval()
        with torch.no_grad():
            for imgs, labels in val_loader_tsne:
                f = model.get_features(imgs.to(DEVICE)).cpu().numpy()
                feats.append(f)
                lbls.extend(labels.numpy())
        feats = np.concatenate(feats, axis=0)
        lbls  = np.array(lbls)

        # Sub-sample for speed
        if len(feats) > 2000:
            idx   = np.random.choice(len(feats), 2000, replace=False)
            feats = feats[idx]
            lbls  = lbls[idx]

        coords = TSNE(n_components=2, random_state=SEED,
                      perplexity=30, n_iter=500).fit_transform(feats)

        ax = axes[col] if len(_viz_models) > 1 else axes
        for i, cls in enumerate(CLASS_NAMES):
            mask = lbls == i
            ax.scatter(coords[mask, 0], coords[mask, 1],
                       c=[_tsne_colors[i]], label=cls, alpha=0.55, s=12)
        ax.set_title({"baseline": "Baseline", "ssl_only": "SSL-only",
                      "ssl_mt": "SSL + Mean Teacher"}.get(mkey, mkey),
                     fontsize=12)
        ax.axis("off")
        if col == len(_viz_models) - 1:
            ax.legend(markerscale=2, loc="upper right",
                      bbox_to_anchor=(1.25, 1.0), fontsize=9)

    plt.tight_layout()
    fig.savefig(FIG_DIR / "tsne_features.png", dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Saved → {FIG_DIR / 'tsne_features.png'}")

# ── Figure F: FlashTorch Backprop saliency ───────────────────────────────────
try:
    from flashtorch.saliency import Backprop as _FTBackprop

    if _viz_models:
        _n_ft_samples = 3
        _ft_df = metadata.groupby("dx", observed=True).apply(
            lambda g: g.sample(1, random_state=SEED)
        ).reset_index(drop=True).head(_n_ft_samples)

        fig, axes = plt.subplots(len(_viz_models), _n_ft_samples * 2,
                                  figsize=(_n_ft_samples * 5, len(_viz_models) * 2.8))
        fig.suptitle("FlashTorch Gradient Saliency\n"
                     "(left: input, right: vanilla gradients, target = predicted class)",
                     fontsize=12, y=1.01)

        for row, (mkey, model) in enumerate(_viz_models.items()):
            bp = _FTBackprop(model)
            for col_pair, (_, sample) in enumerate(_ft_df.iterrows()):
                path = _path_cache.get(sample["image_id"])
                if path is None:
                    continue
                img_t = VAL_TRANSFORM(Image.open(path).convert("RGB"))

                pred  = model(img_t.unsqueeze(0).to(DEVICE)).argmax(1).item()
                try:
                    grads = bp.calculate_gradients(
                        img_t.unsqueeze(0), target_class=pred, guided=False)
                    grad_np = grads[0].abs().mean(0).numpy() if grads is not None \
                        else np.zeros(img_t.shape[1:])
                except Exception:
                    grad_np = np.zeros(img_t.shape[1:])

                col_i = col_pair * 2
                col_g = col_pair * 2 + 1
                ax_i  = axes[row, col_i] if len(_viz_models) > 1 else axes[col_i]
                ax_g  = axes[row, col_g] if len(_viz_models) > 1 else axes[col_g]

                ax_i.imshow(unnormalise(img_t))
                ax_i.set_title(f"{sample['dx']}", fontsize=8)
                ax_i.axis("off")

                ax_g.imshow(grad_np, cmap="hot")
                ax_g.set_title(f"→ {IDX_TO_CLS.get(pred, '?')}", fontsize=8)
                ax_g.axis("off")

            row_ax = axes[row, 0] if len(_viz_models) > 1 else axes[0]
            row_ax.set_ylabel({"baseline": "Baseline", "ssl_only": "SSL-only",
                               "ssl_mt": "SSL + MT"}.get(mkey, mkey),
                              fontsize=10, fontweight="bold")

        plt.tight_layout()
        fig.savefig(FIG_DIR / "flashtorch_saliency.png", dpi=150, bbox_inches="tight")
        plt.show()
        print(f"Saved → {FIG_DIR / 'flashtorch_saliency.png'}")
except ImportError:
    print("flashtorch not installed — skipping saliency maps.")
except Exception as _ft_err:
    print(f"FlashTorch saliency skipped: {_ft_err}")

# %% [markdown]
# ## 11 — Upload All Paper Figures to WandB

# %%
_paper_figures = {
    "eda_class_dist.png":         "EDA — class distribution",
    "eda_sample_grid.png":        "EDA — sample images per class",
    "eda_demographics.png":       "EDA — age and sex distribution",
    "results_comparison.png":     "Main result — method comparison bar chart",
    "label_fraction_ablation.png": "Ablation — balanced accuracy vs label fraction",
    "gradcam_comparison.png":     "GradCAM — model attention per method",
    "confusion_matrix.png":       "Confusion matrix — SSL + Mean Teacher",
    "tsne_features.png":          "t-SNE — learned feature spaces",
    "flashtorch_saliency.png":    "FlashTorch gradient saliency",
    "results_comparison.pdf":     "Main result PDF (for paper)",
    "label_fraction_ablation.pdf":"Ablation PDF (for paper)",
}

if USE_WANDB:
    with wandb.init(project=WANDB_PROJECT, name="paper-figures",
                    reinit=True, job_type="figures") as run:
        for fname, caption in _paper_figures.items():
            fpath = FIG_DIR / fname
            if fpath.exists():
                # Log as WandB Image (PNG) or artifact (PDF)
                if fname.endswith(".png"):
                    wandb.log({caption: wandb.Image(str(fpath), caption=caption)})
                save_checkpoint(fpath, f"fig-{fpath.stem}", "", "")
                print(f"  Uploaded: {fname}")
            else:
                print(f"  Skipped (not found): {fname}")
    print("All available figures uploaded to WandB.")
else:
    print(f"WandB not available. Figures saved locally at: {FIG_DIR}")

# %%
print("\n" + "=" * 60)
print("  HAM10000 Experiment Complete")
print("=" * 60)
print(f"  Figures directory : {FIG_DIR}")
print(f"  CV results file   : {CV_RESULTS_FILE}")
print(f"  WandB project     : {WANDB_PROJECT}")
print("\n  Paper-ready figures generated:")
for fname, caption in _paper_figures.items():
    fpath = FIG_DIR / fname
    status = "✅" if fpath.exists() else "⬜ (pending experiment completion)"
    print(f"  {status}  {fname:<45} {caption}")
print("=" * 60)
