"""
Medical Segmentation Decathlon — MONAI-based dataset loaders.

Tasks used in this project (all from http://medicaldecathlon.com/):
  Task03_Liver    — CT,  2 classes (liver, tumour)   → binary liver mask
  Task07_Pancreas — CT,  2 classes (pancreas, mass)  → binary pancreas mask
  Task02_Heart    — MRI, 1 class  (left atrium)

Kaggle dataset slugs (add each as a separate dataset input):
  liver:    vivekprajapati2048/medical-segmentation-decathlon-3dliver
  heart:    vivekprajapati2048/medical-segmentation-decathlon-heart
  pancreas: lnguynquangbnh/task07-pancreas

Note: all three Kaggle datasets store files as .nii (not .nii.gz).
glob_nii() handles both extensions automatically.

Each Kaggle dataset mounts at /kaggle/input/<slug>/.
The folder layout inside may be either:
  (A) /kaggle/input/<slug>/Task03_Liver/imagesTr/   (task subfolder preserved)
  (B) /kaggle/input/<slug>/imagesTr/                (files at root)
resolve_task_dir() handles both automatically.
"""

import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from monai.data import DataLoader, CacheDataset
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, Spacingd,
    Orientationd, ScaleIntensityRanged, CropForegroundd,
    RandCropByPosNegLabeld, RandFlipd, RandRotate90d, ToTensord,
    NormalizeIntensityd, RandGaussianNoised, RandScaleIntensityd,
    RandShiftIntensityd, Lambdad, SpatialPadd,
)


# ── Task definitions ──────────────────────────────────────────────────────────

TASKS = {
    "liver": {
        "task_id":       0,
        "task_folder":   "Task03_Liver",
        "kaggle_slug":   "vivekprajapati2048/medical-segmentation-decathlon-3dliver",
        "kaggle_input":  "medical-segmentation-decathlon-3dliver",
        "modality":      "CT",
        "spacing":       (1.5, 1.5, 1.5),  # FIXED: Changed from (1.5, 1.5, 2.0) to isotropic
        "intensity_range": (-175, 250),
        "label_value":   1,
        "n_train_approx": 131,
    },
    "pancreas": {
        "task_id":       1,
        "task_folder":   "Task07_Pancreas",
        "kaggle_slug":   "lnguynquangbnh/task07-pancreas",
        "kaggle_input":  "task07-pancreas",
        "modality":      "CT",
        "spacing":       (1.5, 1.5, 1.5),  # FIXED: Changed from (1.5, 1.5, 2.0) to isotropic
        "intensity_range": (-125, 275),
        "label_value":   1,
        "n_train_approx": 281,
    },
    "heart": {
        "task_id":       2,
        "task_folder":   "Task02_Heart",
        "kaggle_slug":   "vivekprajapati2048/medical-segmentation-decathlon-heart",
        "kaggle_input":  "medical-segmentation-decathlon-heart",
        "modality":      "MRI",
        "spacing":       (1.5, 1.5, 1.5),  # FIXED: Changed from (1.25, 1.25, 1.37) to isotropic
        "intensity_range": None,
        "label_value":   1,
        "n_train_approx": 20,
    },
}

TASK_ORDER = ["liver", "pancreas", "heart"]


# ── Path resolution ───────────────────────────────────────────────────────────

def resolve_task_dir(task_root: str, task_name: str) -> Path:
    root = Path(task_root)
    task_folder = TASKS[task_name]["task_folder"]

    candidate_a = root / task_folder / "imagesTr"
    if candidate_a.exists():
        return root / task_folder

    candidate_b = root / "imagesTr"
    if candidate_b.exists():
        return root

    raise FileNotFoundError(
        f"Cannot find imagesTr/ in '{task_root}'.\n"
        f"Tried:\n  {candidate_a}\n  {candidate_b}\n"
        f"Make sure you added the dataset '{TASKS[task_name]['kaggle_slug']}' "
        f"as a Kaggle input."
    )


def build_task_roots(base: str) -> Dict[str, str]:
    return dict.fromkeys(TASKS, base)


def kaggle_task_roots() -> Dict[str, str]:
    return {
        name: f"/kaggle/input/{cfg['kaggle_input']}"
        for name, cfg in TASKS.items()
    }


_EXT_GZ  = ".nii.gz"
_EXT_NII = ".nii"


def _real_nii(paths: List[Path]) -> List[Path]:
    """Keep only real files — filter out directories and macOS resource-fork (._) entries."""
    return [p for p in paths if p.is_file() and not p.name.startswith("._")]


def glob_nii(directory: Path) -> List[Path]:
    """
    Find NIfTI files in a directory. Handles three layouts:
      Flat gz  : imagesTr/case.nii.gz
      Flat nii : imagesTr/case.nii           (Kaggle liver/heart)
      Nested   : imagesTr/case.nii/case.nii  (Kaggle nested packaging)
    """
    gz = _real_nii(sorted(directory.glob(f"*{_EXT_GZ}")))
    if gz:
        return gz
    flat = _real_nii(sorted(directory.glob(f"*{_EXT_NII}")))
    if flat:
        return flat
    return _real_nii(sorted(directory.glob(f"*/*{_EXT_NII}")))


# ── Dataset verification ───────────────────────────────────────────────────────

def verify_datasets(task_roots: Dict[str, str],
                    required: List[str] = None) -> bool:
    """Verify only the tasks actually needed. Defaults to all tasks in task_roots."""
    print("\n── Dataset verification ──────────────────────────────────────")
    all_ok = True
    tasks_to_check = required if required is not None else list(task_roots.keys())
    for task_name in tasks_to_check:
        root = task_roots.get(task_name, "")
        try:
            task_dir = resolve_task_dir(root, task_name)
            n_imgs = len(glob_nii(task_dir / "imagesTr"))
            n_lbls = len(glob_nii(task_dir / "labelsTr"))
            approx = TASKS[task_name]["n_train_approx"]
            ok = "✅" if n_imgs > 0 else "❌"
            warn = f"  (expected ~{approx})" if n_imgs != approx else ""
            print(f"  {ok}  {task_name:<10}  imgs={n_imgs:<4}  lbls={n_lbls:<4}"
                  f"  path={task_dir}{warn}")
        except FileNotFoundError:
            print(f"  ❌  {task_name:<10}  NOT FOUND — {TASKS[task_name]['kaggle_slug']}")
            all_ok = False
    print("─────────────────────────────────────────────────────────────\n")
    return all_ok


# ── File list builders ────────────────────────────────────────────────────────

def get_file_list(task_roots: Dict[str, str],
                  task_name: str) -> Tuple[List[dict], List[dict]]:
    task_dir = resolve_task_dir(task_roots[task_name], task_name)
    img_dir  = task_dir / "imagesTr"
    lbl_dir  = task_dir / "labelsTr"

    img_files = glob_nii(img_dir)
    files = [
        {"image": str(img),
         "label": str(lbl_dir / img.relative_to(img_dir)),
         "task":  TASKS[task_name]["task_id"]}
        for img in img_files
    ]

    rng   = np.random.default_rng(42)
    idx   = rng.permutation(len(files))
    split = max(1, int(0.8 * len(files)))
    return [files[i] for i in idx[:split]], [files[i] for i in idx[split:]]


def get_unlabelled_files(task_roots: Dict[str, str]) -> List[dict]:
    all_files = []
    for task_name in TASK_ORDER:
        try:
            task_dir = resolve_task_dir(task_roots[task_name], task_name)
            all_files += [{"image": str(p)} for p in glob_nii(task_dir / "imagesTr")]
        except FileNotFoundError:
            print(f"  ⚠️  Skipping {task_name} for SSL pretraining (not found)")
    return all_files


# ── Transforms ────────────────────────────────────────────────────────────────

def _binarize_label(x):
    """Map all non-zero labels to 1. Handles liver/pancreas (0/1/2) → binary (0/1)."""
    return (x > 0).float()


def _ct_transforms(task_name: str, train: bool) -> Compose:
    cfg    = TASKS[task_name]
    a_min, a_max = cfg["intensity_range"]
    keys   = ["image", "label"]
    base   = [
        LoadImaged(keys=keys),
        EnsureChannelFirstd(keys=keys),
        Lambdad(keys=["label"], func=_binarize_label),
        Spacingd(keys=keys, pixdim=cfg["spacing"], mode=("bilinear", "nearest")),
        Orientationd(keys=keys, axcodes="RAS"),
        ScaleIntensityRanged(keys=["image"], a_min=a_min, a_max=a_max,
                             b_min=0.0, b_max=1.0, clip=True),
        CropForegroundd(keys=keys, source_key="image"),
    ]
    if train:
        base += [
            SpatialPadd(keys=keys, spatial_size=(96, 96, 96)),
            RandCropByPosNegLabeld(keys=keys, label_key="label",
                                   spatial_size=(96, 96, 96),
                                   pos=1, neg=1, num_samples=8,
                                   allow_smaller=True),
            RandFlipd(keys=keys, prob=0.5, spatial_axis=0),
            RandRotate90d(keys=keys, prob=0.5, max_k=3),
            # Intensity augmentation — improves generalisation across scanners
            RandGaussianNoised(keys=["image"], prob=0.2, std=0.01),
            RandScaleIntensityd(keys=["image"], factors=0.1, prob=0.5),
            RandShiftIntensityd(keys=["image"], offsets=0.1, prob=0.5),
        ]
    base.append(ToTensord(keys=keys))
    return Compose(base)


def _mri_transforms(task_name: str, train: bool) -> Compose:
    cfg  = TASKS[task_name]
    keys = ["image", "label"]
    base = [
        LoadImaged(keys=keys),
        EnsureChannelFirstd(keys=keys),
        Lambdad(keys=["label"], func=_binarize_label),
        Spacingd(keys=keys, pixdim=cfg["spacing"], mode=("bilinear", "nearest")),
        Orientationd(keys=keys, axcodes="RAS"),
        NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True),
        CropForegroundd(keys=keys, source_key="image"),
    ]
    if train:
        base += [
            SpatialPadd(keys=keys, spatial_size=(96, 96, 96)),
            RandCropByPosNegLabeld(keys=keys, label_key="label",
                                   spatial_size=(96, 96, 96),
                                   pos=1, neg=1, num_samples=8,
                                   allow_smaller=True),
            RandFlipd(keys=keys, prob=0.5, spatial_axis=0),
            # MRI has high inter-scanner variability — stronger intensity aug
            RandGaussianNoised(keys=["image"], prob=0.2, std=0.05),
            RandScaleIntensityd(keys=["image"], factors=0.1, prob=0.5),
            RandShiftIntensityd(keys=["image"], offsets=0.1, prob=0.5),
        ]
    base.append(ToTensord(keys=keys))
    return Compose(base)


def get_transforms(task_name: str, train: bool) -> Compose:
    if TASKS[task_name]["modality"] == "CT":
        return _ct_transforms(task_name, train)
    return _mri_transforms(task_name, train)


# ── DataLoader builder ────────────────────────────────────────────────────────

def get_loaders(task_roots: Dict[str, str],
                task_name:  str,
                batch_size: int = 2,
                num_workers: int = 4,
                cache_rate: float = 0.1,
                pin_memory: bool = False) -> Tuple[DataLoader, DataLoader]:
    """
    Build train and validation dataloaders.
    
    Args:
        pin_memory: If False (default), disables pin_memory to prevent MetaTensor corruption.
                   Set to True only if you have sufficient GPU memory and stable CUDA setup.
    """
    train_files, val_files = get_file_list(task_roots, task_name)
    train_ds = CacheDataset(train_files,
                            transform=get_transforms(task_name, train=True),
                            cache_rate=cache_rate)
    val_ds   = CacheDataset(val_files,
                            transform=get_transforms(task_name, train=False),
                            cache_rate=1.0)
    train_loader = DataLoader(train_ds, batch_size=batch_size,
                              shuffle=True,  num_workers=num_workers,
                              pin_memory=pin_memory)
    val_loader   = DataLoader(val_ds,   batch_size=1,
                              shuffle=False, num_workers=num_workers,
                              pin_memory=pin_memory)
    return train_loader, val_loader


def validate_batch(batch: dict, task_name: str) -> bool:
    """
    Validate batch tensor shapes before training.
    
    Raises:
        ValueError: If tensor shapes are invalid
    """
    img = batch["image"]
    lbl = batch["label"]
    
    # Expected shape: [B, C, D, H, W]
    if img.dim() != 5:
        raise ValueError(
            f"[{task_name}] Image tensor has {img.dim()} dimensions, expected 5. "
            f"Shape: {img.shape}"
        )
    
    if lbl.dim() != 4:
        raise ValueError(
            f"[{task_name}] Label tensor has {lbl.dim()} dimensions, expected 4. "
            f"Shape: {lbl.shape}"
        )
    
    if img.shape[2:] != lbl.shape[1:]:
        raise ValueError(
            f"[{task_name}] Image spatial dims {img.shape[2:]} don't match "
            f"label spatial dims {lbl.shape[1:]}."
        )
    
    if torch.isnan(img).any():
        raise ValueError(f"[{task_name}] NaN values detected in image tensor")
    
    if torch.isnan(lbl).any():
        raise ValueError(f"[{task_name}] NaN values detected in label tensor")
    
    return True
