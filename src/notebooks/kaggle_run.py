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
restore_checkpoint("best.pth",
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

TASK_ORDER = ["liver", "heart"]   # core 2-task run; add "pancreas" later

baseline_cfg = {
    "strategy":        "none",
    "use_pretrained":  False,
    "task_roots":      TASK_ROOTS,
    "output_dir":      os.path.join(OUT_DIR, "baseline_finetune"),
    "task_order":      TASK_ORDER,
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
    "task_order":         TASK_ORDER,
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
    "task_order":         TASK_ORDER,
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

# %% [markdown]
# ## 6 — Results

# %%
import json
import numpy as np
from evaluation.metrics import (backward_transfer, forgetting_measure,
                                 average_accuracy)

EXPERIMENTS = {
    "Fine-tune (baseline)": "baseline_finetune",
    "EWC (no SSL)":         "ewc_no_ssl",
    "EWC + SparK":          "ewc_ssl",
    "LwF (no SSL)":         "lwf_no_ssl",
}
TASKS = TASK_ORDER

print(f"\n{'Method':<24} " + "  ".join(f"{t:>8}" for t in TASKS) +
      f"  {'AA':>7} {'BWT':>8} {'F':>8}")
print("-" * 70)

for label, folder in EXPERIMENTS.items():
    result_path = os.path.join(OUT_DIR, folder, "cl_results.json")
    if not os.path.exists(result_path):
        print(f"  {label:<24}  [not run yet]")
        continue
    with open(result_path) as f:
        data = json.load(f)
    R = np.array(data["R_matrix"])
    last_row = R[-1]
    aa  = average_accuracy(R)
    bwt = backward_transfer(R) if R.shape[0] > 1 else float("nan")
    fm  = forgetting_measure(R) if R.shape[0] > 1 else float("nan")
    scores = "  ".join(f"{last_row[i]:>8.3f}" for i in range(len(TASKS)))
    print(f"  {label:<24} {scores}  {aa:>6.3f}  {bwt:>7.3f}  {fm:>7.3f}")

# %%
# RQ3: does SSL pretraining reduce forgetting?
print("\n=== RQ3: SparK pretraining vs no pretraining (EWC) ===\n")
print(f"{'':12} {'No-SSL BWT':>12} {'SSL BWT':>10} {'Delta':>10}")
print("-" * 48)

for strategy in ["ewc"]:
    bwt_vals = {}
    for tag, folder in [("no_ssl", f"{strategy}_no_ssl"),
                        ("ssl",    f"{strategy}_ssl")]:
        p = os.path.join(OUT_DIR, folder, "cl_results.json")
        if not os.path.exists(p):
            continue
        with open(p) as f:
            R = np.array(json.load(f)["R_matrix"])
        bwt_vals[tag] = backward_transfer(R)
    if "no_ssl" in bwt_vals and "ssl" in bwt_vals:
        delta = bwt_vals["ssl"] - bwt_vals["no_ssl"]
        print(f"  {strategy:<12} {bwt_vals['no_ssl']:>12.4f} "
              f"{bwt_vals['ssl']:>10.4f} {delta:>+10.4f}")
print("\n  Positive delta = SSL reduces forgetting.")

# %%
import shutil

RESULTS_DIR = "/kaggle/working/results" if ON_KAGGLE else os.path.join(OUT_DIR, "final_results")
os.makedirs(RESULTS_DIR, exist_ok=True)

for label, folder in EXPERIMENTS.items():
    src = os.path.join(OUT_DIR, folder, "cl_results.json")
    if os.path.exists(src):
        shutil.copy(src, os.path.join(RESULTS_DIR, f"{folder}_results.json"))

print(f"Results saved to: {RESULTS_DIR}")
print("Done.")
