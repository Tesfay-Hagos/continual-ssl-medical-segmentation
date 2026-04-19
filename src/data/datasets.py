"""
Medical Segmentation Decathlon — MONAI-based dataset loaders.

Tasks used in this project (all from http://medicaldecathlon.com/):
  Task03_Liver    — CT,  2 classes (liver, tumour)   → we use binary liver mask
  Task07_Pancreas — CT,  2 classes (pancreas, mass)  → we use binary pancreas mask
  Task02_Heart    — MRI, 1 class  (left atrium)

Each task is treated as a separate continual learning step.
"""

import os
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
from monai.data import Dataset, DataLoader, CacheDataset
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, Spacingd,
    Orientationd, ScaleIntensityRanged, CropForegroundd,
    RandCropByPosNegLabeld, RandFlipd, RandRotate90d, ToTensord,
    NormalizeIntensityd,
)


# ── Task definitions ──────────────────────────────────────────────────────────

TASKS = {
    "liver": {
        "task_id": 0,
        "folder": "Task03_Liver",
        "modality": "CT",
        "spacing": (1.5, 1.5, 2.0),
        "intensity_range": (-175, 250),
        "label_value": 1,
    },
    "pancreas": {
        "task_id": 1,
        "folder": "Task07_Pancreas",
        "modality": "CT",
        "spacing": (1.5, 1.5, 2.0),
        "intensity_range": (-125, 275),
        "label_value": 1,
    },
    "heart": {
        "task_id": 2,
        "folder": "Task02_Heart",
        "modality": "MRI",
        "spacing": (1.25, 1.25, 1.37),
        "intensity_range": None,   # MRI: use NormalizeIntensity
        "label_value": 1,
    },
}

TASK_ORDER = ["liver", "pancreas", "heart"]   # sequential training order


def get_file_list(data_root: str, task_name: str) -> Tuple[List[dict], List[dict]]:
    """Return (train_files, val_files) dicts for a given task."""
    task_cfg = TASKS[task_name]
    task_dir = Path(data_root) / task_cfg["folder"]
    img_dir   = task_dir / "imagesTr"
    lbl_dir   = task_dir / "labelsTr"

    cases = sorted([f.stem.replace(".nii", "") for f in img_dir.glob("*.nii.gz")])
    files = [
        {"image": str(img_dir / f"{c}.nii.gz"),
         "label": str(lbl_dir / f"{c}.nii.gz"),
         "task":  task_cfg["task_id"]}
        for c in cases
    ]

    # 80/20 split (fixed seed)
    rng = np.random.default_rng(42)
    idx = rng.permutation(len(files))
    split = int(0.8 * len(files))
    return [files[i] for i in idx[:split]], [files[i] for i in idx[split:]]


def get_unlabelled_files(data_root: str) -> List[dict]:
    """Collect all images (ignoring labels) from all tasks for SSL pretraining."""
    all_files = []
    for task_name, cfg in TASKS.items():
        img_dir = Path(data_root) / cfg["folder"] / "imagesTr"
        all_files += [{"image": str(p)} for p in sorted(img_dir.glob("*.nii.gz"))]
    return all_files


def _ct_transforms(task_name: str, train: bool) -> Compose:
    cfg = TASKS[task_name]
    a_min, a_max = cfg["intensity_range"]
    keys = ["image", "label"]
    base = [
        LoadImaged(keys=keys),
        EnsureChannelFirstd(keys=keys),
        Spacingd(keys=keys, pixdim=cfg["spacing"], mode=("bilinear", "nearest")),
        Orientationd(keys=keys, axcodes="RAS"),
        ScaleIntensityRanged(keys=["image"], a_min=a_min, a_max=a_max,
                             b_min=0.0, b_max=1.0, clip=True),
        CropForegroundd(keys=keys, source_key="image"),
    ]
    if train:
        base += [
            RandCropByPosNegLabeld(keys=keys, label_key="label",
                                   spatial_size=(96, 96, 96),
                                   pos=1, neg=1, num_samples=4),
            RandFlipd(keys=keys, prob=0.5, spatial_axis=0),
            RandRotate90d(keys=keys, prob=0.5, max_k=3),
        ]
    base.append(ToTensord(keys=keys))
    return Compose(base)


def _mri_transforms(task_name: str, train: bool) -> Compose:
    cfg = TASKS[task_name]
    keys = ["image", "label"]
    base = [
        LoadImaged(keys=keys),
        EnsureChannelFirstd(keys=keys),
        Spacingd(keys=keys, pixdim=cfg["spacing"], mode=("bilinear", "nearest")),
        Orientationd(keys=keys, axcodes="RAS"),
        NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True),
        CropForegroundd(keys=keys, source_key="image"),
    ]
    if train:
        base += [
            RandCropByPosNegLabeld(keys=keys, label_key="label",
                                   spatial_size=(96, 96, 96),
                                   pos=1, neg=1, num_samples=4),
            RandFlipd(keys=keys, prob=0.5, spatial_axis=0),
        ]
    base.append(ToTensord(keys=keys))
    return Compose(base)


def get_transforms(task_name: str, train: bool) -> Compose:
    if TASKS[task_name]["modality"] == "CT":
        return _ct_transforms(task_name, train)
    return _mri_transforms(task_name, train)


def get_loaders(data_root: str, task_name: str,
                batch_size: int = 2,
                num_workers: int = 4,
                cache_rate: float = 0.1) -> Tuple[DataLoader, DataLoader]:
    train_files, val_files = get_file_list(data_root, task_name)
    train_ds = CacheDataset(train_files, transform=get_transforms(task_name, train=True),
                            cache_rate=cache_rate)
    val_ds   = CacheDataset(val_files,   transform=get_transforms(task_name, train=False),
                            cache_rate=1.0)
    train_loader = DataLoader(train_ds, batch_size=batch_size,
                              shuffle=True,  num_workers=num_workers, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=1,
                              shuffle=False, num_workers=num_workers, pin_memory=True)
    return train_loader, val_loader
