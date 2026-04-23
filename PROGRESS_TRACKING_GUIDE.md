# Progress Tracking & Continuous Logging Guide

## Overview

The training system maintains **persistent progress tracking** across all experiments, allowing you to:
- ✅ Resume interrupted training from the exact checkpoint
- ✅ Build complete experiment graphs across multiple runs
- ✅ Log all consecutive runs to a unified results database
- ✅ Track metrics across all tasks and strategies

## Architecture

### 1. Progress File (`cl_progress.json`)

Located in the output directory (e.g., `/kaggle/working/checkpoints/ewc/`), this file tracks:

```json
{
  "completed_tasks": ["liver", "pancreas"],
  "R_dsc": [[0.85, 0.0, 0.0], [0.82, 0.78, 0.0], [0.0, 0.0, 0.0]],
  "R_hd95": [[5.2, null, null], [6.1, 4.8, null], [null, null, null]],
  "per_task_best": {
    "liver": {"best_dsc": 0.85, "best_hd95": 5.2},
    "pancreas": {"best_dsc": 0.78, "best_hd95": 4.8}
  },
  "wandb_run_id": "abc123xyz"
}
```

**Key fields:**
- `completed_tasks`: List of tasks already trained (skipped on resume)
- `R_dsc` / `R_hd95`: Results matrices (rows=after task t, cols=task i)
- `per_task_best`: Best metrics achieved during training for each task
- `wandb_run_id`: WandB run ID for resuming the same experiment

### 2. Checkpoint System

**Three-tier checkpoint hierarchy:**

```
checkpoints/
├── ewc/
│   ├── cl_progress.json          ← Progress file (restored from WandB/Drive)
│   ├── cl_results.json           ← Final results (uploaded after completion)
│   ├── latest.pth                ← Latest checkpoint (model + optimizer state)
│   ├── liver/
│   │   ├── best.pth              ← Best model for this task
│   │   ├── epoch-5.pth           ← Periodic checkpoints
│   │   └── epoch-10.pth
│   ├── pancreas/
│   │   ├── best.pth
│   │   └── ...
│   └── heart/
│       ├── best.pth
│       └── ...
```

### 3. Storage Backends

**Priority order for persistence:**

1. **WandB Artifacts** (Primary)
   - Versioned, 100 GB free storage
   - Auto-downloaded on resume
   - Synced across runs
   - Artifact names: `cl-{strategy}-progress`, `cl-{strategy}-results`

2. **Google Drive** (Backup)
   - Optional secondary storage
   - Survives WandB outages
   - Requires one-time setup (see below)

3. **Local Disk** (Temporary)
   - Working directory during training
   - Cleared between runs if not backed up

## How Progress Tracking Works

### Resume Flow

When you restart training with the same config:

```python
# 1. Try to restore progress from WandB
restore_checkpoint("cl_progress.json", out_dir, "cl-ewc-progress", project)

# 2. Load progress file if it exists
if progress_file.exists():
    prog = json.load(open(progress_file))
    completed_tasks = prog.get("completed_tasks", [])
    R_dsc = np.array(prog["R_dsc"])
    R_hd95 = np.array(prog["R_hd95"])
    saved_run_id = prog.get("wandb_run_id")

# 3. Resume WandB run with same ID
if saved_run_id:
    wandb.init(id=saved_run_id, resume="must")

# 4. Skip completed tasks, continue from next task
for t, task_name in enumerate(tasks):
    if task_name in completed_tasks:
        print(f"✅ Already completed — skipping")
        continue
    # Train this task...
```

### Continuous Logging

**During training, metrics are logged to:**

1. **WandB Dashboard** (Real-time)
   - Per-task training loss
   - Per-task validation metrics (DSC, HD95)
   - Cross-task evaluation results
   - Summary metrics (AA, BWT, FM)

2. **Local JSON Files**
   - `cl_progress.json` - Updated after each task
   - `cl_results.json` - Final results after all tasks

3. **Task Checkpoints**
   - `latest.pth` - Current model state
   - `best.pth` - Best model for each task
   - Periodic checkpoints every N epochs

### Results Matrix Building

The system maintains a **results matrix** that grows as tasks are completed:

```
After Task 1 (Liver):
R_dsc = [[0.85, 0.0, 0.0],
         [0.0,  0.0, 0.0],
         [0.0,  0.0, 0.0]]

After Task 2 (Pancreas):
R_dsc = [[0.82, 0.78, 0.0],      ← Liver performance after Pancreas training
         [0.0,  0.85, 0.0],      ← Pancreas performance
         [0.0,  0.0,  0.0]]

After Task 3 (Heart):
R_dsc = [[0.80, 0.75, 0.72],     ← Liver performance after all tasks (forgetting)
         [0.78, 0.83, 0.70],     ← Pancreas performance
         [0.0,  0.0,  0.88]]     ← Heart performance
```

**Metrics computed from matrix:**
- **Average Accuracy (AA)**: Mean of diagonal (current task performance)
- **Backward Transfer (BWT)**: How much previous tasks degrade
- **Forgetting (FM)**: How much performance drops after learning new tasks

## Running Experiments with Progress Tracking

### First Run

```bash
cd continual_self_supervised_learning
python src/scripts/train_continual.py --config src/configs/ewc.yaml
```

**What happens:**
1. Creates output directory: `/kaggle/working/checkpoints/ewc/`
2. Initializes WandB run
3. Trains Task 1 (Liver)
4. Saves progress to `cl_progress.json`
5. Uploads to WandB artifact `cl-ewc-progress`
6. Continues to Task 2, 3, etc.

### Resume After Interruption

```bash
# Same command - automatically resumes!
python src/scripts/train_continual.py --config src/configs/ewc.yaml
```

**What happens:**
1. Restores `cl_progress.json` from WandB
2. Loads completed tasks: `["liver", "pancreas"]`
3. Resumes WandB run with same ID
4. Skips Liver and Pancreas training
5. Continues from Heart (Task 3)
6. Updates results matrix with new evaluations

### Run Multiple Strategies

```bash
# Run EWC
python src/scripts/train_continual.py --config src/configs/ewc.yaml

# Run LwF (separate experiment)
python src/scripts/train_continual.py --config src/configs/lwf.yaml

# Run Replay (separate experiment)
python src/scripts/train_continual.py --config src/configs/replay.yaml

# Run Baseline (no continual learning)
python src/scripts/train_continual.py --config src/configs/ewc.yaml --strategy none
```

**Each strategy maintains separate:**
- Output directory: `/kaggle/working/checkpoints/{strategy}/`
- Progress file: `cl_progress.json`
- WandB artifact: `cl-{strategy}-progress`
- Results file: `cl_results.json`

## Building Complete Experiment Graphs

### Collecting Results from All Runs

After all experiments complete, results are available in:

```
checkpoints/
├── ewc/cl_results.json
├── lwf/cl_results.json
├── replay/cl_results.json
└── none/cl_results.json
```

Each file contains:
```json
{
  "strategy": "ewc",
  "use_pretrained": true,
  "tasks": ["liver", "pancreas", "heart"],
  "R_dsc": [[0.85, 0.82, 0.80], ...],
  "R_hd95": [[5.2, 6.1, 7.5], ...],
  "per_task_best": {...},
  "metrics_dsc": {
    "AA": 0.82,
    "BWT": -0.03,
    "FM": 0.05
  }
}
```

### Creating Comparison Plots

```python
import json
import numpy as np
import matplotlib.pyplot as plt

# Load results from all strategies
strategies = ["ewc", "lwf", "replay", "none"]
results = {}
for strategy in strategies:
    with open(f"checkpoints/{strategy}/cl_results.json") as f:
        results[strategy] = json.load(f)

# Plot Average Accuracy comparison
fig, ax = plt.subplots(figsize=(10, 6))
aa_values = [results[s]["metrics_dsc"]["AA"] for s in strategies]
ax.bar(strategies, aa_values)
ax.set_ylabel("Average Accuracy (DSC)")
ax.set_title("Continual Learning Strategy Comparison")
plt.savefig("comparison_aa.png")

# Plot Forgetting comparison
fm_values = [results[s]["metrics_dsc"]["FM"] for s in strategies]
ax.bar(strategies, fm_values)
ax.set_ylabel("Forgetting Measure")
ax.set_title("Catastrophic Forgetting Comparison")
plt.savefig("comparison_fm.png")
```

## Google Drive Setup (Optional)

For backup persistence across WandB outages:

### One-time Setup (Local Machine)

```bash
# Install Google Drive libraries
pip install google-auth-oauthlib google-api-python-client

# Run setup script
python src/utils/gdrive_setup.py
```

This creates `credentials.json` with your Google Drive access token.

### Configure Kaggle Secrets

1. Copy contents of `credentials.json`
2. Create Kaggle secret: `GDRIVE_CREDENTIALS` with the JSON content
3. Create a folder in Google Drive for backups
4. Copy folder ID to Kaggle secret: `GDRIVE_FOLDER_ID`

### Enable in Config

```yaml
# src/configs/ewc.yaml
gdrive_folder_id: "your-folder-id"
gdrive_credentials: "your-credentials-json"
```

Now checkpoints are backed up to both WandB and Google Drive.

## Monitoring Progress

### WandB Dashboard

View real-time metrics:
- **Training Loss**: Per-task training loss over epochs
- **Validation Metrics**: DSC and HD95 for each task
- **Cross-Task Evaluation**: How each task performs after learning new tasks
- **Summary Metrics**: AA, BWT, FM for continual learning analysis

### Local Progress File

Check progress anytime:

```bash
cat checkpoints/ewc/cl_progress.json | python -m json.tool
```

Shows:
- Which tasks are completed
- Current results matrix
- Best metrics per task
- WandB run ID

### Results File

After completion:

```bash
cat checkpoints/ewc/cl_results.json | python -m json.tool
```

Shows:
- Full results matrix (R_dsc, R_hd95)
- Continual learning metrics (AA, BWT, FM)
- Per-task best performance

## Troubleshooting

### Progress File Not Restoring

**Problem**: Training starts from Task 1 instead of resuming

**Solution**:
1. Check WandB is configured: `wandb login`
2. Verify artifact exists: `wandb artifact list`
3. Check local progress file: `ls -la checkpoints/ewc/cl_progress.json`
4. Manually restore: `wandb artifact get cl-ewc-progress:latest`

### Metrics Not Logging to WandB

**Problem**: No metrics appear in WandB dashboard

**Solution**:
1. Enable WandB in config: `use_wandb: true`
2. Check WandB login: `wandb login`
3. Verify project name in config: `wandb_project: "cssl-medical"`
4. Check WandB run: `wandb runs --project cssl-medical`

### Results Matrix Not Building

**Problem**: R_dsc shows zeros instead of metrics

**Solution**:
1. Ensure all tasks complete: Check `completed_tasks` in progress file
2. Verify val_loaders are created: Check console output for "Eval {task}"
3. Check for NaN values: `np.isnan(R_dsc).any()`

## Best Practices

1. **Always use WandB** for experiment tracking
   - Enables automatic resume on interruption
   - Provides real-time monitoring
   - Stores results for later analysis

2. **Save progress frequently**
   - Progress file updated after each task
   - Checkpoints saved every N epochs
   - Latest checkpoint always available

3. **Use consistent config files**
   - Same config = same experiment
   - Different configs = different experiments
   - Prevents accidental overwrites

4. **Monitor metrics during training**
   - Check WandB dashboard for anomalies
   - Watch for NaN or zero metrics
   - Verify cross-task evaluation is running

5. **Back up results**
   - Download `cl_results.json` after completion
   - Export WandB runs for archival
   - Keep local copies of important experiments

## Summary

The progress tracking system ensures:
- ✅ **Resilience**: Resume from any interruption
- ✅ **Continuity**: Build complete experiment graphs
- ✅ **Persistence**: All metrics logged and backed up
- ✅ **Reproducibility**: Same config = same results
- ✅ **Scalability**: Run multiple strategies in parallel

All consecutive runs are automatically logged and tracked, building a complete picture of your continual learning experiments.
