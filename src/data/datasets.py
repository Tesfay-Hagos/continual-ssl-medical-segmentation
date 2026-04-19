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
from monai.data import DataLoader, CacheDataset
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, Spacingd,
    Orientationd, ScaleIntensityRanged, CropForegroundd,
    RandCropByPosNegLabeld, RandFlipd, RandRotate90d, ToTensord,
    NormalizeIntensityd,
)


# ── Task definitions ──────────────────────────────────────────────────────────

TASKS = {
    "liver": {
        "task_id":       0,
        "task_folder":   "Task03_Liver",
        "kaggle_slug":   "vivekprajapati2048/medical-segmentation-decathlon-3dliver",
        "kaggle_input":  "medical-segmentation-decathlon-3dliver",
        "modality":      "CT",
        "spacing":       (1.5, 1.5, 2.0),
        "intensity_range": (-175, 250),
        "label_value":   1,
        "n_train_approx": 131,   # MSD Task03: 131 training volumes
    },
    "pancreas": {
        "task_id":       1,
        "task_folder":   "Task07_Pancreas",
        "kaggle_slug":   "lnguynquangbnh/task07-pancreas",
        "kaggle_input":  "task07-pancreas",
        "modality":      "CT",
        "spacing":       (1.5, 1.5, 2.0),
        "intensity_range": (-125, 275),
        "label_value":   1,
        "n_train_approx": 281,   # MSD Task07: 281 training volumes
    },
    "heart": {
        "task_id":       2,
        "task_folder":   "Task02_Heart",
        "kaggle_slug":   "vivekprajapati2048/medical-segmentation-decathlon-heart",
        "kaggle_input":  "medical-segmentation-decathlon-heart",
        "modality":      "MRI",
        "spacing":       (1.25, 1.25, 1.37),
        "intensity_range": None,
        "label_value":   1,
        "n_train_approx": 20,    # MSD Task02: 20 training volumes
    },
}

TASK_ORDER = ["liver", "pancreas", "heart"]


# ── Path resolution: handles both Kaggle layout variants ──────────────────────

def resolve_task_dir(task_root: str, task_name: str) -> Path:
    """
    Given the mounted dataset root (e.g. /kaggle/input/medical-segmentation-decathlon-heart),
    find the actual directory containing imagesTr/.

    Handles two layouts:
      (A) task_root/Task02_Heart/imagesTr/   ← task subfolder present
      (B) task_root/imagesTr/               ← files directly at root
    Raises FileNotFoundError if neither exists.
    """
    root = Path(task_root)
    task_folder = TASKS[task_name]["task_folder"]

    # Layout A: task subfolder
    candidate_a = root / task_folder / "imagesTr"
    if candidate_a.exists():
        return root / task_folder

    # Layout B: root directly contains imagesTr
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
    """
    Build per-task roots from a single base path (for local use where all
    task folders live under one directory, e.g. /data/decathlon/Task03_Liver).
    """
    return dict.fromkeys(TASKS, base)


def kaggle_task_roots() -> Dict[str, str]:
    """Return the standard /kaggle/input/<slug> paths for each task."""
    return {
        name: f"/kaggle/input/{cfg['kaggle_input']}"
        for name, cfg in TASKS.items()
    }


_EXT_GZ  = ".nii.gz"
_EXT_NII = ".nii"


def glob_nii(directory: Path) -> List[Path]:
    """Glob NIfTI files from a directory — tries .nii.gz first, falls back to .nii."""
    gz = sorted(directory.glob(f"*{_EXT_GZ}"))
    return gz if gz else sorted(directory.glob(f"*{_EXT_NII}"))


# ── Dataset verification ───────────────────────────────────────────────────────

def verify_datasets(task_roots: Dict[str, str]) -> bool:
    """Print a verification table and return True only if all tasks are found."""
    print("\n── Dataset verification ──────────────────────────────────────")
    all_ok = True
    for task_name in TASK_ORDER:
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
    """Return (train_files, val_files) for one task."""
    task_dir = resolve_task_dir(task_roots[task_name], task_name)
    img_dir  = task_dir / "imagesTr"
    lbl_dir  = task_dir / "labelsTr"

    img_files = glob_nii(img_dir)
    ext       = img_files[0].suffix if img_files else _EXT_GZ
    if ext == ".gz":                         # Path.suffix of "foo.nii.gz" is ".gz"
        ext = _EXT_GZ
    cases = [f.name.replace(_EXT_GZ, "").replace(_EXT_NII, "") for f in img_files]
    files = [
        {"image": str(img_dir / f"{c}{ext}"),
         "label": str(lbl_dir / f"{c}{ext}"),
         "task":  TASKS[task_name]["task_id"]}
        for c in cases
    ]

    rng   = np.random.default_rng(42)
    idx   = rng.permutation(len(files))
    split = max(1, int(0.8 * len(files)))
    return [files[i] for i in idx[:split]], [files[i] for i in idx[split:]]


def get_unlabelled_files(task_roots: Dict[str, str]) -> List[dict]:
    """Collect all images from all tasks for SSL pretraining (no labels needed)."""
    all_files = []
    for task_name in TASK_ORDER:
        try:
            task_dir = resolve_task_dir(task_roots[task_name], task_name)
            all_files += [{"image": str(p)} for p in glob_nii(task_dir / "imagesTr")]
        except FileNotFoundError:
            print(f"  ⚠️  Skipping {task_name} for SSL pretraining (not found)")
    return all_files


# ── Transforms ────────────────────────────────────────────────────────────────

def _ct_transforms(task_name: str, train: bool) -> Compose:
    cfg    = TASKS[task_name]
    a_min, a_max = cfg["intensity_range"]
    keys   = ["image", "label"]
    base   = [
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
    cfg  = TASKS[task_name]
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


# ── DataLoader builder ────────────────────────────────────────────────────────

def get_loaders(task_roots: Dict[str, str],
                task_name:  str,
                batch_size: int = 2,
                num_workers: int = 4,
                cache_rate: float = 0.1) -> Tuple[DataLoader, DataLoader]:
    train_files, val_files = get_file_list(task_roots, task_name)
    train_ds = CacheDataset(train_files,
                            transform=get_transforms(task_name, train=True),
                            cache_rate=cache_rate)
    val_ds   = CacheDataset(val_files,
                            transform=get_transforms(task_name, train=False),
                            cache_rate=1.0)
    train_loader = DataLoader(train_ds, batch_size=batch_size,
                              shuffle=True,  num_workers=num_workers,
                              pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=1,
                              shuffle=False, num_workers=num_workers,
                              pin_memory=True)
    return train_loader, val_loader
