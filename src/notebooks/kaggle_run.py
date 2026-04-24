# %% [markdown]
# # Continual Self-Supervised Learning for Medical Image Segmentation
#
# **Method:** U-Net encoder + SparK masked pretraining + EWC / LwF / Experience Replay
#
# **Paper:** *Continual Self-Supervised Learning for Medical Image Segmentation:
# A Simplified Framework Without Federated Components*
#
# ---
# ### Before running
# 1. Add **three separate Kaggle dataset inputs**:
#    - `vivekprajapati2048/medical-segmentation-decathlon-heart` (Heart / Task02)
#    - `vivekprajapati2048/medical-segmentation-decathlon-3dliver` (Liver / Task03)
#    - `eliasmarcon/pancreas` (Pancreas / Task07)
# 2. Enable GPU accelerator (P100 or T4)
# 3. Run all cells top-to-bottom

# %% [markdown]
# ## 0 — Environment setup

# %%
import os
import sys
import subprocess

# Detect whether we are on Kaggle or local
ON_KAGGLE = os.path.exists("/kaggle/working")

REPO_URL  = "https://github.com/Tesfay-Hagos/continual-ssl-medical-segmentation.git"
REPO_DIR  = "/kaggle/working/project" if ON_KAGGLE else os.path.abspath(
                os.path.join(os.path.dirname(__file__), "..", ".."))
OUT_DIR   = "/kaggle/working/checkpoints" if ON_KAGGLE else "/tmp/cssl_ckpts"

print(f"ON_KAGGLE : {ON_KAGGLE}")
print(f"REPO_DIR  : {REPO_DIR}")
print(f"OUT_DIR   : {OUT_DIR}")

# %%
# Clone or pull repo on Kaggle so we always have the latest src/ code
if ON_KAGGLE:
    if not os.path.exists(REPO_DIR):
        result = subprocess.run(["git", "clone", REPO_URL, REPO_DIR],
                                capture_output=True, text=True)
    else:
        result = subprocess.run(["git", "-C", REPO_DIR, "pull"],
                                capture_output=True, text=True)
    print(result.stdout or result.stderr)
    # Remove stale .pyc files so updated modules are recompiled fresh
    subprocess.run(["find", REPO_DIR, "-type", "d", "-name", "__pycache__",
                    "-exec", "rm", "-rf", "{}", "+"],
                   capture_output=True)
else:
    print(f"Using repo at: {REPO_DIR}")

# %%
# Install dependencies
if ON_KAGGLE:
    subprocess.run(
        [sys.executable, "-m", "pip", "install",
         "monai[all]", "nibabel", "scipy", "scikit-image", "pyyaml",
         "wandb", "--quiet"],
        check=True
    )
    print("Dependencies installed.")
else:
    print("Skipping pip install (local run — assumed already installed).")

# %%
# Add src to Python path
import importlib

SRC_DIR = os.path.join(REPO_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)
importlib.invalidate_caches()

import torch
import numpy as np
from data.datasets import kaggle_task_roots, build_task_roots, verify_datasets

TASK_ROOTS = (kaggle_task_roots() if ON_KAGGLE
              else build_task_roots(os.environ.get("DATA_ROOT", "/data/decathlon")))

gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU only"
print(f"PyTorch  : {torch.__version__}")
print(f"Device   : {gpu_name}")
print(f"CUDA     : {torch.version.cuda}")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.makedirs(OUT_DIR, exist_ok=True)

# %%
# WandB — reads 'wandb_token' from Kaggle Secrets (Add-ons → Secrets → wandb_token)
import wandb

WANDB_PROJECT = "cssl-medical"
try:
    if ON_KAGGLE:
        from kaggle_secrets import UserSecretsClient
        _secrets = UserSecretsClient()
        _key = _secrets.get_secret("WANDB_API_KEY")
        wandb.login(key=_key)
    else:
        wandb.login()          # uses WANDB_API_KEY env var locally
    USE_WANDB = True
    print(f"WandB logged in. Project: {WANDB_PROJECT}")
except Exception as e:
    USE_WANDB = False
    print(f"WandB login failed ({e}) — running without logging.")

# Set to True only after running src/utils/gdrive_setup.py and adding
# GDRIVE_CREDENTIALS + GDRIVE_FOLDER_ID to Kaggle secrets.
USE_GDRIVE = False

GDRIVE_FOLDER_ID   = ""
GDRIVE_CREDENTIALS = ""
if USE_GDRIVE:
    try:
        GDRIVE_FOLDER_ID   = _secrets.get_secret("GDRIVE_FOLDER_ID")
        GDRIVE_CREDENTIALS = _secrets.get_secret("GDRIVE_CREDENTIALS")
        print(f"Google Drive backup enabled (folder: {GDRIVE_FOLDER_ID[:8]}...)")
    except Exception as e:
        print(f"Google Drive setup failed ({e}) — using WandB only.")
        USE_GDRIVE = False
else:
    print("Google Drive disabled — checkpoints saved to WandB artifacts only.")

# %% [markdown]
# ## 1 — Dataset verification

# %%
if not verify_datasets(TASK_ROOTS):
    raise RuntimeError(
        "One or more datasets are missing. "
        "Add all three Kaggle dataset inputs (heart, liver, pancreas)."
    )
print("Dataset OK — all three tasks found.")

# %% [markdown]
# ## 2 — SparK Pretraining
#
# Pretrain the U-Net encoder using sparse masked image modeling on all
# unlabelled volumes from all three tasks combined.
#
# *Skip this cell if a pretrained checkpoint already exists.*

# %%
import yaml
from pretraining.pretrain import pretrain

PRETRAIN_CKPT = os.path.join(OUT_DIR, "pretrain", "best.pth")
PRETRAIN_DONE = os.path.join(OUT_DIR, "pretrain", "pretrain_done.json")

# Try to restore the completion marker from WandB so we can tell whether
# pretraining truly finished (best.pth is created after epoch 1, so it
# cannot be used as a "done" signal).
from utils.storage import restore_checkpoint
from pathlib import Path
restore_checkpoint("pretrain_done.json",
                   Path(os.path.join(OUT_DIR, "pretrain")),
                   "pretrain-checkpoint", WANDB_PROJECT,
                   GDRIVE_FOLDER_ID, GDRIVE_CREDENTIALS)

if os.path.exists(PRETRAIN_DONE):
    import json as _json
    _info = _json.load(open(PRETRAIN_DONE))
    print(f"Pretraining already complete "
          f"({_info['epochs_completed']} epochs, best_loss={_info['best_loss']:.5f})")
    print("Delete pretrain_done.json to re-run pretraining.")
else:
    cfg_path = os.path.join(SRC_DIR, "configs", "pretraining.yaml")
    with open(cfg_path) as f:
        pretrain_cfg = yaml.safe_load(f)

    pretrain_cfg["task_roots"]        = TASK_ROOTS
    pretrain_cfg["output_dir"]        = os.path.join(OUT_DIR, "pretrain")
    pretrain_cfg["epochs"]            = 100
    pretrain_cfg["batch_size"]        = 2
    pretrain_cfg["num_workers"]       = 2 if ON_KAGGLE else 0
    pretrain_cfg["use_wandb"]         = USE_WANDB
    pretrain_cfg["wandb_project"]     = WANDB_PROJECT
    pretrain_cfg["wandb_run"]         = "spark-pretrain"
    pretrain_cfg["gdrive_folder_id"]  = GDRIVE_FOLDER_ID
    pretrain_cfg["gdrive_credentials"] = GDRIVE_CREDENTIALS

    pretrain(pretrain_cfg)
    print(f"\nPretraining complete. Checkpoint: {PRETRAIN_CKPT}")

# %% [markdown]
# ## 3 — Baseline: Fine-tune only (no CL, no pretraining)
#
# Shows the full extent of catastrophic forgetting.
# Trains sequentially on each task with a vanilla fine-tune — no regularization,
# no replay, no pretrained weights.  This is the lower bound.

# %%
from scripts.train_continual import run as run_continual

baseline_cfg = {
    "strategy":        "none",
    "use_pretrained":  False,
    "task_roots":      TASK_ROOTS,
    "output_dir":      os.path.join(OUT_DIR, "baseline_finetune"),
    "task_order":      ["liver", "pancreas", "heart"],
    "channels":        [32, 64, 128, 256, 512],
    "strides":         [2, 2, 2, 2],
    "epochs_per_task": 15,
    "batch_size":      2,
    "lr":              1e-4,
    "weight_decay":    1e-5,
    "num_workers":     2 if ON_KAGGLE else 0,
    "cache_rate":      0.1,
    "use_wandb":       USE_WANDB,
    "wandb_project":   WANDB_PROJECT,
    "wandb_run":       "baseline_finetune",
}

print("=== BASELINE: Fine-tune only (catastrophic forgetting) ===")
run_continual(baseline_cfg)

# %% [markdown]
# ## 4 — EWC experiments
#
# **4a** — EWC *without* pretraining (random init encoder)
# **4b** — EWC *with* SparK pretraining
#
# Comparing 4a vs 4b directly answers **RQ3**:
# does SSL pretraining act as an implicit anti-forgetting regularizer?

# %%
cfg_path = os.path.join(SRC_DIR, "configs", "ewc.yaml")
with open(cfg_path) as f:
    ewc_base = yaml.safe_load(f)

ewc_base.update({
    "task_roots":         TASK_ROOTS,
    "num_workers":        2 if ON_KAGGLE else 0,
    "use_wandb":          USE_WANDB,
    "wandb_project":      WANDB_PROJECT,
    "gdrive_folder_id":   GDRIVE_FOLDER_ID,
    "gdrive_credentials": GDRIVE_CREDENTIALS,
})

# 4a — EWC, no pretraining
ewc_no_ssl = {**ewc_base,
              "use_pretrained": False,
              "wandb_run":      "ewc_no_ssl",
              "output_dir":     os.path.join(OUT_DIR, "ewc_no_ssl")}
print("=== EWC — no pretraining ===")
run_continual(ewc_no_ssl)

# %%
# 4b — EWC, with SparK pretraining
ewc_ssl = {**ewc_base,
           "use_pretrained":  True,
           "pretrained_ckpt": PRETRAIN_CKPT,
           "wandb_run":       "ewc_ssl",
           "output_dir":      os.path.join(OUT_DIR, "ewc_ssl")}
print("=== EWC — SparK pretrained ===")
run_continual(ewc_ssl)

# %% [markdown]
# ## 5 — LwF experiments
#
# **5a** — LwF *without* pretraining
# **5b** — LwF *with* SparK pretraining

# %%
cfg_path = os.path.join(SRC_DIR, "configs", "lwf.yaml")
with open(cfg_path) as f:
    lwf_base = yaml.safe_load(f)

lwf_base.update({
    "task_roots":         TASK_ROOTS,
    "num_workers":        2 if ON_KAGGLE else 0,
    "use_wandb":          USE_WANDB,
    "wandb_project":      WANDB_PROJECT,
    "gdrive_folder_id":   GDRIVE_FOLDER_ID,
    "gdrive_credentials": GDRIVE_CREDENTIALS,
})

# 5a — LwF, no pretraining
lwf_no_ssl = {**lwf_base,
              "use_pretrained": False,
              "wandb_run":      "lwf_no_ssl",
              "output_dir":     os.path.join(OUT_DIR, "lwf_no_ssl")}
print("=== LwF — no pretraining ===")
run_continual(lwf_no_ssl)

# %%
# 5b — LwF, with SparK pretraining
lwf_ssl = {**lwf_base,
           "use_pretrained":  True,
           "pretrained_ckpt": PRETRAIN_CKPT,
           "wandb_run":       "lwf_ssl",
           "output_dir":      os.path.join(OUT_DIR, "lwf_ssl")}
print("=== LwF — SparK pretrained ===")
run_continual(lwf_ssl)

# %% [markdown]
# ## 6 — Experience Replay experiments
#
# **6a** — Replay *without* pretraining
# **6b** — Replay *with* SparK pretraining
#
# Also runs the **RQ4 buffer-size ablation**: 50 / 100 / 200 / 500 samples.

# %%
cfg_path = os.path.join(SRC_DIR, "configs", "replay.yaml")
with open(cfg_path) as f:
    replay_base = yaml.safe_load(f)

replay_base.update({
    "task_roots":         TASK_ROOTS,
    "num_workers":        2 if ON_KAGGLE else 0,
    "use_wandb":          USE_WANDB,
    "wandb_project":      WANDB_PROJECT,
    "gdrive_folder_id":   GDRIVE_FOLDER_ID,
    "gdrive_credentials": GDRIVE_CREDENTIALS,
})

# 6a — Replay, no pretraining
replay_no_ssl = {**replay_base,
                 "use_pretrained": False,
                 "wandb_run":      "replay_no_ssl",
                 "output_dir":     os.path.join(OUT_DIR, "replay_no_ssl")}
print("=== Replay — no pretraining ===")
run_continual(replay_no_ssl)

# %%
# 6b — Replay, with SparK pretraining
replay_ssl = {**replay_base,
              "use_pretrained":  True,
              "pretrained_ckpt": PRETRAIN_CKPT,
              "wandb_run":       "replay_ssl",
              "output_dir":      os.path.join(OUT_DIR, "replay_ssl")}
print("=== Replay — SparK pretrained ===")
run_continual(replay_ssl)

# %% [markdown]
# ### RQ4 Buffer-size ablation
# Minimum buffer size at which Replay matches multi-task upper bound within 5% DSC.

# %%
for buf_size in [100, 500]:
    cfg = {**replay_base,
           "use_pretrained":  True,
           "pretrained_ckpt": PRETRAIN_CKPT,
           "buffer_capacity": buf_size,
           "wandb_run":       f"replay_buf{buf_size}",
           "output_dir":      os.path.join(OUT_DIR, f"replay_buf{buf_size}")}
    print(f"\n=== Replay buf={buf_size} ===")
    run_continual(cfg)

# %% [markdown]
# ## 7 — Upper bound: Multi-task learning
#
# Train on all three tasks simultaneously (all data visible at once).
# This is the best achievable DSC — the ceiling for any CL strategy.

# %%
# Multi-task training: combine all task loaders and train jointly.
from torch.utils.data import ConcatDataset, DataLoader
from data.datasets import get_loaders, TASK_ORDER, get_transforms, get_file_list
from monai.data import CacheDataset
from models.unet import build_unet, UNetWithEncoder
from monai.losses import DiceCELoss
from monai.inferers import sliding_window_inference
from evaluation.metrics import SegmentationEvaluator, print_cl_metrics

multitask_out = os.path.join(OUT_DIR, "multitask")
os.makedirs(multitask_out, exist_ok=True)

# Build combined training set
all_train, all_val = [], {}
for task_name in ["liver", "pancreas", "heart"]:
    tr_files, val_files = get_file_list(TASK_ROOTS, task_name)
    all_train += tr_files
    val_ds = CacheDataset(val_files,
                          transform=get_transforms(task_name, train=False),
                          cache_rate=1.0)
    all_val[task_name] = DataLoader(val_ds, batch_size=1, num_workers=0)

# NOTE: multi-task training uses a shared label space (binary per task),
# so we train with task-specific heads only on separate val loaders.
# For simplicity here we train with a single shared binary head.
print(f"Multi-task training set: {len(all_train)} volumes")

unet = build_unet(in_channels=1, out_channels=2,
                  channels=(32, 64, 128, 256, 512), strides=(2, 2, 2, 2))
mt_model = UNetWithEncoder(unet).to(DEVICE)
if os.path.exists(PRETRAIN_CKPT):
    mt_model.load_pretrained_encoder(PRETRAIN_CKPT)

criterion = DiceCELoss(to_onehot_y=True, softmax=True)
optimizer = torch.optim.AdamW(mt_model.parameters(), lr=1e-4, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)

# Single combined loader (mix all tasks each epoch)
import random
for epoch in range(1, 16):
    mt_model.train()
    random.shuffle(all_train)
    epoch_loss = 0.0
    subset = all_train[:100]
    for item in subset:
        task = ["liver", "pancreas", "heart"][item["task"]]
        tfm  = get_transforms(task, train=True)
        out  = tfm(item)
        if isinstance(out, list):
            out = out[0]
        img = out["image"].unsqueeze(0).to(DEVICE)
        lbl = out["label"].unsqueeze(0).long().to(DEVICE)
        optimizer.zero_grad()
        pred = mt_model(img)
        loss = criterion(pred, lbl)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    scheduler.step()
    if epoch % 5 == 0:
        print(f"  Multi-task epoch {epoch}/15 | loss={epoch_loss/len(subset):.4f}")

# Evaluate upper bound
R_mt = np.zeros((1, 3))
for i, task_name in enumerate(["liver", "pancreas", "heart"]):
    ev = SegmentationEvaluator(num_classes=2)
    mt_model.eval()
    with torch.no_grad():
        for batch in all_val[task_name]:
            pred = sliding_window_inference(
                batch["image"].to(DEVICE), (96, 96, 96),
                sw_batch_size=2, predictor=mt_model, overlap=0.25)
            ev.update(pred, batch["label"].to(DEVICE))
    m = ev.aggregate()
    R_mt[0, i] = m["dice"]
    print(f"  Multi-task {task_name:<10}: DSC={m['dice']:.4f}")

import json
with open(os.path.join(multitask_out, "cl_results.json"), "w") as f:
    json.dump({"strategy": "multitask", "R_matrix": R_mt.tolist()}, f, indent=2)

# %% [markdown]
# ## 8 — Results: full comparison table
#
# Reproduces Table 1 from the paper.

# %%
import json
import numpy as np
from evaluation.metrics import (backward_transfer, forgetting_measure,
                                 average_accuracy)

EXPERIMENTS = {
    "Fine-tune (baseline)":    "baseline_finetune",
    "EWC (no SSL)":            "ewc_no_ssl",
    "EWC + SparK":             "ewc_ssl",
    "LwF (no SSL)":            "lwf_no_ssl",
    "LwF + SparK":             "lwf_ssl",
    "Replay (no SSL)":         "replay_no_ssl",
    "Replay + SparK":          "replay_ssl",
    "Multi-task (upper bound)":"multitask",
}
TASKS = ["Liver", "Pancreas", "Heart"]

print(f"\n{'Method':<28} {'Liver':>7} {'Pancreas':>9} {'Heart':>7} "
      f"{'AA':>7} {'BWT':>8} {'F':>8}")
print("-" * 80)

for label, folder in EXPERIMENTS.items():
    result_path = os.path.join(OUT_DIR, folder, "cl_results.json")
    if not os.path.exists(result_path):
        print(f"  {label:<28}  [not run yet]")
        continue
    with open(result_path) as f:
        data = json.load(f)
    R = np.array(data["R_matrix"])
    last_row = R[-1]
    aa  = average_accuracy(R)
    bwt = backward_transfer(R) if R.shape[0] > 1 else float("nan")
    fm  = forgetting_measure(R) if R.shape[0] > 1 else float("nan")
    print(f"  {label:<28} {last_row[0]:>6.3f}  {last_row[1]:>8.3f}  "
          f"{last_row[2]:>6.3f}  {aa:>6.3f}  {bwt:>7.3f}  {fm:>7.3f}")

# %% [markdown]
# ## 9 — RQ3 analysis: SSL as implicit anti-forgetting regularizer

# %%
print("\n=== RQ3: Effect of SparK pretraining on forgetting ===\n")
print(f"{'Strategy':<12} {'No SSL BWT':>12} {'SSL BWT':>10} {'Delta BWT':>11}")
print("-" * 48)

for strategy in ["ewc", "lwf", "replay"]:
    bwt_vals = {}                           # reset per strategy
    for ssl_tag, folder in [("no_ssl", f"{strategy}_no_ssl"),
                             ("ssl",    f"{strategy}_ssl")]:
        p = os.path.join(OUT_DIR, folder, "cl_results.json")
        if not os.path.exists(p):
            continue
        with open(p) as f:
            R = np.array(json.load(f)["R_matrix"])
        bwt_vals[ssl_tag] = backward_transfer(R)

    if "no_ssl" in bwt_vals and "ssl" in bwt_vals:
        delta = bwt_vals["ssl"] - bwt_vals["no_ssl"]
        print(f"  {strategy:<12} {bwt_vals['no_ssl']:>12.4f} "
              f"{bwt_vals['ssl']:>10.4f} {delta:>+11.4f}")

print("\n  Positive delta = SSL pretraining reduces forgetting (supports RQ3).")

# %% [markdown]
# ## 10 — RQ4 buffer-size ablation

# %%
print("\n=== RQ4: Minimum replay buffer size ===\n")

# Load multi-task upper bound as reference
mt_path = os.path.join(OUT_DIR, "multitask", "cl_results.json")
mt_aa   = None
if os.path.exists(mt_path):
    mt_aa = average_accuracy(np.array(json.load(open(mt_path))["R_matrix"]))
    print(f"Multi-task upper bound AA: {mt_aa:.4f}")

print(f"\n{'Buffer':>8} {'AA':>8} {'BWT':>8} {'vs upper (%)':>14}")
print("-" * 44)

for buf in [50, 100, 200, 500]:
    p = os.path.join(OUT_DIR, f"replay_buf{buf}", "cl_results.json")
    if not os.path.exists(p):
        print(f"  {buf:>7}  [not run]")
        continue
    with open(p) as f:
        R = np.array(json.load(f)["R_matrix"])
    aa  = average_accuracy(R)
    bwt = backward_transfer(R)
    gap = ((mt_aa - aa) / mt_aa * 100) if mt_aa else float("nan")
    marker = " ← within 5%" if abs(gap) <= 5.0 else ""
    print(f"  {buf:>7}  {aa:>7.4f}  {bwt:>7.4f}  {gap:>12.1f}%{marker}")

# %% [markdown]
# ## 11 — Save all results to /kaggle/working/results/

# %%
import shutil

RESULTS_DIR = "/kaggle/working/results" if ON_KAGGLE else os.path.join(OUT_DIR, "final_results")
os.makedirs(RESULTS_DIR, exist_ok=True)

for label, folder in EXPERIMENTS.items():
    src = os.path.join(OUT_DIR, folder, "cl_results.json")
    if os.path.exists(src):
        dst = os.path.join(RESULTS_DIR, f"{folder}_results.json")
        shutil.copy(src, dst)

print(f"All results saved to: {RESULTS_DIR}")
print("Done.")
