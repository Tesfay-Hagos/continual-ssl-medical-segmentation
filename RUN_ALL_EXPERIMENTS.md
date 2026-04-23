# Running All Experiments & Building Complete Graphs

## Quick Start

Run all continual learning strategies with automatic progress tracking:

```bash
cd continual_self_supervised_learning

# Run all strategies sequentially
python src/scripts/train_continual.py --config src/configs/ewc.yaml
python src/scripts/train_continual.py --config src/configs/lwf.yaml
python src/scripts/train_continual.py --config src/configs/replay.yaml
python src/scripts/train_continual.py --config src/configs/ewc.yaml --strategy none

# Or run in parallel (if GPU memory allows)
# python src/scripts/train_continual.py --config src/configs/ewc.yaml &
# python src/scripts/train_continual.py --config src/configs/lwf.yaml &
# python src/scripts/train_continual.py --config src/configs/replay.yaml &
# wait
```

## Experiment Configurations

### 1. EWC (Elastic Weight Consolidation)

**Config**: `src/configs/ewc.yaml`

```yaml
strategy: ewc
ewc_lambda: 1000.0
fisher_batches: 100
epochs_per_task: 15
```

**What it does:**
- Protects important weights from previous tasks
- Uses Fisher information matrix to identify critical parameters
- Adds penalty term to loss function

**Expected results:**
- Low forgetting (FM < 0.05)
- Moderate backward transfer (BWT ≈ -0.02)
- Good average accuracy (AA > 0.80)

**Run**:
```bash
python src/scripts/train_continual.py --config src/configs/ewc.yaml
```

### 2. LwF (Learning without Forgetting)

**Config**: `src/configs/lwf.yaml`

```yaml
strategy: lwf
lwf_alpha: 1.0
lwf_temperature: 2.0
epochs_per_task: 15
```

**What it does:**
- Uses knowledge distillation from previous model
- Maintains soft targets from old tasks
- Balances new task learning with old task preservation

**Expected results:**
- Very low forgetting (FM < 0.02)
- Minimal backward transfer (BWT ≈ 0.0)
- Slightly lower average accuracy (AA > 0.78)

**Run**:
```bash
python src/scripts/train_continual.py --config src/configs/lwf.yaml
```

### 3. Experience Replay

**Config**: `src/configs/replay.yaml`

```yaml
strategy: replay
buffer_capacity: 200
replay_batch_size: 2
epochs_per_task: 15
```

**What it does:**
- Stores samples from previous tasks
- Replays them during new task training
- Prevents catastrophic forgetting through rehearsal

**Expected results:**
- Very low forgetting (FM < 0.01)
- Excellent backward transfer (BWT ≈ 0.01)
- Best average accuracy (AA > 0.82)

**Run**:
```bash
python src/scripts/train_continual.py --config src/configs/replay.yaml
```

### 4. Baseline (No Continual Learning)

**Config**: `src/configs/ewc.yaml` with `--strategy none`

```bash
python src/scripts/train_continual.py --config src/configs/ewc.yaml --strategy none
```

**What it does:**
- Standard sequential training without any continual learning strategy
- Baseline for comparison

**Expected results:**
- High forgetting (FM > 0.10)
- Negative backward transfer (BWT < -0.05)
- Lower average accuracy (AA < 0.75)

## Experiment Execution Plan

### Phase 1: Single Task Baseline (Optional)

Train on each task individually to establish upper bounds:

```bash
# Create baseline config
cat > src/configs/baseline.yaml << 'EOF'
output_dir: "/kaggle/working/checkpoints/baseline"
pretrained_ckpt: "/kaggle/working/checkpoints/pretrain/best.pth"
use_pretrained: true
task_order: ["liver"]
channels: [32, 64, 128, 256, 512]
strides: [2, 2, 2, 2]
epochs_per_task: 20
batch_size: 2
lr: 1.0e-4
weight_decay: 1.0e-5
num_workers: 2
cache_rate: 0.1
pin_memory: false
EOF

# Run baseline
python src/scripts/train_continual.py --config src/configs/baseline.yaml
```

### Phase 2: Continual Learning Strategies

Run each strategy in sequence:

```bash
# Strategy 1: EWC
echo "Starting EWC..."
python src/scripts/train_continual.py --config src/configs/ewc.yaml
echo "EWC complete. Results: checkpoints/ewc/cl_results.json"

# Strategy 2: LwF
echo "Starting LwF..."
python src/scripts/train_continual.py --config src/configs/lwf.yaml
echo "LwF complete. Results: checkpoints/lwf/cl_results.json"

# Strategy 3: Replay
echo "Starting Replay..."
python src/scripts/train_continual.py --config src/configs/replay.yaml
echo "Replay complete. Results: checkpoints/replay/cl_results.json"

# Strategy 4: Baseline (no CL)
echo "Starting Baseline..."
python src/scripts/train_continual.py --config src/configs/ewc.yaml --strategy none
echo "Baseline complete. Results: checkpoints/none/cl_results.json"
```

### Phase 3: Hyperparameter Ablations (Optional)

Test different hyperparameters:

```bash
# EWC with different lambda values
for lambda in 500 1000 5000 10000; do
  cat > src/configs/ewc_lambda_${lambda}.yaml << EOF
output_dir: "/kaggle/working/checkpoints/ewc_lambda_${lambda}"
ewc_lambda: ${lambda}
# ... rest of config
EOF
  python src/scripts/train_continual.py --config src/configs/ewc_lambda_${lambda}.yaml
done

# LwF with different temperatures
for temp in 2 3 4 5; do
  cat > src/configs/lwf_temp_${temp}.yaml << EOF
output_dir: "/kaggle/working/checkpoints/lwf_temp_${temp}"
lwf_temperature: ${temp}
# ... rest of config
EOF
  python src/scripts/train_continual.py --config src/configs/lwf_temp_${temp}.yaml
done
```

## Monitoring Progress

### Real-time Monitoring

```bash
# Watch progress file updates
watch -n 5 'cat checkpoints/ewc/cl_progress.json | python -m json.tool'

# Monitor GPU usage
watch -n 1 nvidia-smi

# Check WandB dashboard
# Open: https://wandb.ai/your-entity/cssl-medical
```

### Check Completed Tasks

```bash
# EWC progress
python -c "import json; print(json.load(open('checkpoints/ewc/cl_progress.json'))['completed_tasks'])"

# LwF progress
python -c "import json; print(json.load(open('checkpoints/lwf/cl_progress.json'))['completed_tasks'])"

# Replay progress
python -c "import json; print(json.load(open('checkpoints/replay/cl_progress.json'))['completed_tasks'])"
```

## Collecting Results

### After All Experiments Complete

```bash
# Create results directory
mkdir -p results

# Copy all results
cp checkpoints/ewc/cl_results.json results/ewc_results.json
cp checkpoints/lwf/cl_results.json results/lwf_results.json
cp checkpoints/replay/cl_results.json results/replay_results.json
cp checkpoints/none/cl_results.json results/baseline_results.json

# Create summary
python << 'EOF'
import json
import numpy as np

strategies = ["ewc", "lwf", "replay", "baseline"]
results = {}

for strategy in strategies:
    with open(f"results/{strategy}_results.json") as f:
        results[strategy] = json.load(f)

# Print summary
print("\n" + "="*60)
print("CONTINUAL LEARNING RESULTS SUMMARY")
print("="*60)

for strategy in strategies:
    r = results[strategy]
    metrics = r["metrics_dsc"]
    print(f"\n{strategy.upper()}:")
    print(f"  Average Accuracy (AA):  {metrics['AA']:.4f}")
    print(f"  Backward Transfer (BWT): {metrics['BWT']:.4f}")
    print(f"  Forgetting (FM):        {metrics['FM']:.4f}")

print("\n" + "="*60)
EOF
```

## Building Comparison Plots

### Create Comprehensive Comparison Script

```python
# save as: plot_results.py
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Load results
strategies = ["ewc", "lwf", "replay", "baseline"]
results = {}

for strategy in strategies:
    path = Path(f"checkpoints/{strategy}/cl_results.json")
    if path.exists():
        with open(path) as f:
            results[strategy] = json.load(f)

if not results:
    print("No results found. Run experiments first.")
    exit(1)

# Create figure with subplots
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("Continual Learning Strategy Comparison", fontsize=16, fontweight="bold")

# 1. Average Accuracy
ax = axes[0, 0]
aa_values = [results[s]["metrics_dsc"]["AA"] for s in strategies if s in results]
colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
ax.bar(strategies, aa_values, color=colors[:len(strategies)])
ax.set_ylabel("Average Accuracy (DSC)")
ax.set_title("Average Accuracy (Higher is Better)")
ax.set_ylim([0.7, 0.9])
for i, v in enumerate(aa_values):
    ax.text(i, v + 0.01, f"{v:.3f}", ha="center", fontweight="bold")

# 2. Backward Transfer
ax = axes[0, 1]
bwt_values = [results[s]["metrics_dsc"]["BWT"] for s in strategies if s in results]
ax.bar(strategies, bwt_values, color=colors[:len(strategies)])
ax.set_ylabel("Backward Transfer (DSC)")
ax.set_title("Backward Transfer (Higher is Better)")
ax.axhline(y=0, color="k", linestyle="--", alpha=0.3)
for i, v in enumerate(bwt_values):
    ax.text(i, v + 0.002, f"{v:.4f}", ha="center", fontweight="bold")

# 3. Forgetting
ax = axes[1, 0]
fm_values = [results[s]["metrics_dsc"]["FM"] for s in strategies if s in results]
ax.bar(strategies, fm_values, color=colors[:len(strategies)])
ax.set_ylabel("Forgetting Measure (DSC)")
ax.set_title("Forgetting (Lower is Better)")
for i, v in enumerate(fm_values):
    ax.text(i, v + 0.002, f"{v:.4f}", ha="center", fontweight="bold")

# 4. Results Matrix Heatmap (for first strategy)
ax = axes[1, 1]
first_strategy = list(results.keys())[0]
R_dsc = np.array(results[first_strategy]["R_dsc"])
tasks = results[first_strategy]["tasks"]
im = ax.imshow(R_dsc, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
ax.set_xticks(range(len(tasks)))
ax.set_yticks(range(len(tasks)))
ax.set_xticklabels(tasks)
ax.set_yticklabels([f"After Task {i+1}" for i in range(len(tasks))])
ax.set_title(f"Results Matrix ({first_strategy.upper()})")
plt.colorbar(im, ax=ax, label="DSC")

# Add values to heatmap
for i in range(R_dsc.shape[0]):
    for j in range(R_dsc.shape[1]):
        if not np.isnan(R_dsc[i, j]):
            ax.text(j, i, f"{R_dsc[i, j]:.2f}", ha="center", va="center",
                   color="white" if R_dsc[i, j] < 0.5 else "black", fontsize=9)

plt.tight_layout()
plt.savefig("comparison_results.png", dpi=300, bbox_inches="tight")
print("✅ Saved: comparison_results.png")

# Create detailed comparison table
print("\n" + "="*80)
print("DETAILED RESULTS COMPARISON")
print("="*80)

for strategy in strategies:
    if strategy not in results:
        continue
    r = results[strategy]
    print(f"\n{strategy.upper()}")
    print("-" * 80)
    print(f"  Average Accuracy (AA):    {r['metrics_dsc']['AA']:.4f}")
    print(f"  Backward Transfer (BWT):  {r['metrics_dsc']['BWT']:.4f}")
    print(f"  Forgetting (FM):          {r['metrics_dsc']['FM']:.4f}")
    print(f"  Tasks: {', '.join(r['tasks'])}")
    print(f"  Per-task best DSC:")
    for task, metrics in r["per_task_best"].items():
        print(f"    {task}: {metrics['best_dsc']:.4f}")

print("\n" + "="*80)
```

Run the plotting script:

```bash
python plot_results.py
```

## Expected Results

### Typical Performance Metrics

| Strategy | AA | BWT | FM | Notes |
|----------|----|----|----|----|
| Baseline | 0.72 | -0.08 | 0.12 | High forgetting, poor backward transfer |
| EWC | 0.80 | -0.02 | 0.05 | Good balance, moderate performance |
| LwF | 0.78 | -0.01 | 0.02 | Very low forgetting, slightly lower AA |
| Replay | 0.82 | 0.01 | 0.01 | Best performance, requires memory |

### Results Matrix Example (EWC)

```
After Task 1 (Liver):
  Liver: 0.85

After Task 2 (Pancreas):
  Liver:    0.82  (↓ 0.03 forgetting)
  Pancreas: 0.78

After Task 3 (Heart):
  Liver:    0.80  (↓ 0.05 total forgetting)
  Pancreas: 0.75  (↓ 0.03 forgetting)
  Heart:    0.81
```

## Troubleshooting

### Experiment Interrupted

**Resume automatically:**
```bash
python src/scripts/train_continual.py --config src/configs/ewc.yaml
```

The system will:
1. Restore progress from WandB
2. Skip completed tasks
3. Resume from the next task

### Results Not Saving

**Check:**
```bash
# Verify results file exists
ls -la checkpoints/ewc/cl_results.json

# Check WandB upload
wandb artifact list | grep cl-ewc-results

# Manually save
cp checkpoints/ewc/cl_results.json results/ewc_backup.json
```

### Metrics Look Wrong

**Verify:**
```bash
# Check results matrix
python -c "import json; r=json.load(open('checkpoints/ewc/cl_results.json')); print('R_dsc:', r['R_dsc'])"

# Check for NaN values
python -c "import json, numpy as np; r=json.load(open('checkpoints/ewc/cl_results.json')); print('NaN count:', np.isnan(r['R_dsc']).sum())"
```

## Next Steps

After collecting all results:

1. **Analyze results** - Compare strategies using plots
2. **Write paper** - Document findings and methodology
3. **Archive experiments** - Save all configs and results
4. **Publish code** - Share implementation on GitHub

See `PROGRESS_TRACKING_GUIDE.md` for detailed progress tracking information.
