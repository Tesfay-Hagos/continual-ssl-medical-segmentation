# Complete Experiment Workflow

## Overview

This document provides a complete workflow for running all experiments, tracking progress, and building comprehensive graphs for your continual learning research.

## Workflow Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│ PHASE 1: SETUP                                                  │
├─────────────────────────────────────────────────────────────────┤
│ ✓ Environment configured                                        │
│ ✓ Data pipeline fixed (isotropic spacing, pin_memory=false)    │
│ ✓ Pre-trained encoder available                                │
│ ✓ Configs ready (ewc.yaml, lwf.yaml, replay.yaml)             │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ PHASE 2: RUN EXPERIMENTS                                        │
├─────────────────────────────────────────────────────────────────┤
│ Strategy 1: EWC                                                 │
│   - Protects important weights                                  │
│   - Expected: AA≈0.80, FM≈0.05                                 │
│   - Output: checkpoints/ewc/cl_results.json                    │
│                                                                 │
│ Strategy 2: LwF                                                 │
│   - Knowledge distillation                                      │
│   - Expected: AA≈0.78, FM≈0.02                                 │
│   - Output: checkpoints/lwf/cl_results.json                    │
│                                                                 │
│ Strategy 3: Replay                                              │
│   - Experience replay                                           │
│   - Expected: AA≈0.82, FM≈0.01                                 │
│   - Output: checkpoints/replay/cl_results.json                 │
│                                                                 │
│ Strategy 4: Baseline                                            │
│   - No continual learning                                       │
│   - Expected: AA≈0.72, FM≈0.12                                 │
│   - Output: checkpoints/none/cl_results.json                   │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ PHASE 3: COLLECT RESULTS                                        │
├─────────────────────────────────────────────────────────────────┤
│ ✓ All cl_results.json files collected                          │
│ ✓ Progress files saved                                          │
│ ✓ WandB artifacts uploaded                                      │
│ ✓ Metrics computed (AA, BWT, FM)                               │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ PHASE 4: BUILD GRAPHS                                           │
├─────────────────────────────────────────────────────────────────┤
│ ✓ Comparison plots (AA, BWT, FM)                               │
│ ✓ Results matrices (heatmaps)                                   │
│ ✓ Per-task performance curves                                   │
│ ✓ Summary tables                                                │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ PHASE 5: ANALYSIS & PUBLICATION                                │
├─────────────────────────────────────────────────────────────────┤
│ ✓ Statistical significance testing                              │
│ ✓ Ablation studies                                              │
│ ✓ Paper writing                                                 │
│ ✓ Code repository preparation                                   │
└─────────────────────────────────────────────────────────────────┘
```

## Step-by-Step Execution

### Step 1: Verify Setup

```bash
cd continual_self_supervised_learning

# Check environment
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import monai; print(f'MONAI: {monai.__version__}')"

# Check configs exist
ls -la src/configs/

# Check pre-trained encoder
ls -la /kaggle/working/checkpoints/pretrain/best.pth

# Verify data pipeline
python -c "from src.data.datasets import get_loaders; print('✓ Data pipeline OK')"
```

### Step 2: Run Experiments

#### Option A: Sequential Execution (Recommended)

```bash
# Run each strategy one after another
# This ensures clean GPU memory between runs

echo "Starting EWC..."
python src/scripts/train_continual.py --config src/configs/ewc.yaml
echo "✓ EWC complete"

echo "Starting LwF..."
python src/scripts/train_continual.py --config src/configs/lwf.yaml
echo "✓ LwF complete"

echo "Starting Replay..."
python src/scripts/train_continual.py --config src/configs/replay.yaml
echo "✓ Replay complete"

echo "Starting Baseline..."
python src/scripts/train_continual.py --config src/configs/ewc.yaml --strategy none
echo "✓ Baseline complete"

echo "All experiments finished!"
```

**Estimated time**: 
- EWC: 2-3 hours (3 tasks × 15 epochs)
- LwF: 2-3 hours
- Replay: 2-3 hours
- Baseline: 2-3 hours
- **Total: 8-12 hours**

#### Option B: Parallel Execution (Advanced)

```bash
# Run multiple strategies in parallel (if GPU memory allows)
# Each strategy uses separate GPU or shares memory carefully

python src/scripts/train_continual.py --config src/configs/ewc.yaml &
EWC_PID=$!

python src/scripts/train_continual.py --config src/configs/lwf.yaml &
LWF_PID=$!

python src/scripts/train_continual.py --config src/configs/replay.yaml &
REPLAY_PID=$!

# Wait for all to complete
wait $EWC_PID $LWF_PID $REPLAY_PID

echo "All experiments finished!"
```

**Note**: Parallel execution requires careful GPU memory management. Sequential is safer.

### Step 3: Monitor Progress

#### Real-time Monitoring

```bash
# In a separate terminal, watch progress updates
watch -n 10 'cat checkpoints/ewc/cl_progress.json | python -m json.tool | head -20'

# Or check all strategies
for strategy in ewc lwf replay none; do
  echo "=== $strategy ==="
  python -c "import json; p=json.load(open('checkpoints/$strategy/cl_progress.json')); print(f'Completed: {p[\"completed_tasks\"]}')" 2>/dev/null || echo "Not started"
done
```

#### WandB Dashboard

```bash
# Open in browser
# https://wandb.ai/your-entity/cssl-medical

# Or view from command line
wandb runs --project cssl-medical
```

### Step 4: Collect Results

```bash
# Create results directory
mkdir -p results

# Copy all results
for strategy in ewc lwf replay none; do
  cp checkpoints/$strategy/cl_results.json results/${strategy}_results.json
  echo "✓ Copied $strategy results"
done

# Verify all files exist
ls -la results/
```

### Step 5: Generate Comparison Plots

```bash
# Create plotting script
cat > plot_comparison.py << 'EOF'
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Load results
strategies = ["ewc", "lwf", "replay", "none"]
results = {}

for strategy in strategies:
    path = Path(f"results/{strategy}_results.json")
    if path.exists():
        with open(path) as f:
            results[strategy] = json.load(f)

# Create comparison figure
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("Continual Learning Strategy Comparison", fontsize=16, fontweight="bold")

# Plot 1: Average Accuracy
ax = axes[0, 0]
strategies_present = [s for s in strategies if s in results]
aa_values = [results[s]["metrics_dsc"]["AA"] for s in strategies_present]
colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
ax.bar(strategies_present, aa_values, color=colors[:len(strategies_present)])
ax.set_ylabel("Average Accuracy (DSC)")
ax.set_title("Average Accuracy (Higher is Better)")
ax.set_ylim([0.65, 0.90])
for i, v in enumerate(aa_values):
    ax.text(i, v + 0.01, f"{v:.3f}", ha="center", fontweight="bold")

# Plot 2: Backward Transfer
ax = axes[0, 1]
bwt_values = [results[s]["metrics_dsc"]["BWT"] for s in strategies_present]
ax.bar(strategies_present, bwt_values, color=colors[:len(strategies_present)])
ax.set_ylabel("Backward Transfer (DSC)")
ax.set_title("Backward Transfer (Higher is Better)")
ax.axhline(y=0, color="k", linestyle="--", alpha=0.3)
for i, v in enumerate(bwt_values):
    ax.text(i, v + 0.002, f"{v:.4f}", ha="center", fontweight="bold")

# Plot 3: Forgetting
ax = axes[1, 0]
fm_values = [results[s]["metrics_dsc"]["FM"] for s in strategies_present]
ax.bar(strategies_present, fm_values, color=colors[:len(strategies_present)])
ax.set_ylabel("Forgetting Measure (DSC)")
ax.set_title("Forgetting (Lower is Better)")
for i, v in enumerate(fm_values):
    ax.text(i, v + 0.002, f"{v:.4f}", ha="center", fontweight="bold")

# Plot 4: Results Matrix (first strategy)
ax = axes[1, 1]
first_strategy = strategies_present[0]
R_dsc = np.array(results[first_strategy]["R_dsc"])
tasks = results[first_strategy]["tasks"]
im = ax.imshow(R_dsc, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
ax.set_xticks(range(len(tasks)))
ax.set_yticks(range(len(tasks)))
ax.set_xticklabels(tasks)
ax.set_yticklabels([f"After Task {i+1}" for i in range(len(tasks))])
ax.set_title(f"Results Matrix ({first_strategy.upper()})")
plt.colorbar(im, ax=ax, label="DSC")

for i in range(R_dsc.shape[0]):
    for j in range(R_dsc.shape[1]):
        if not np.isnan(R_dsc[i, j]):
            ax.text(j, i, f"{R_dsc[i, j]:.2f}", ha="center", va="center",
                   color="white" if R_dsc[i, j] < 0.5 else "black", fontsize=9)

plt.tight_layout()
plt.savefig("results/comparison_results.png", dpi=300, bbox_inches="tight")
print("✅ Saved: results/comparison_results.png")

# Print summary table
print("\n" + "="*80)
print("CONTINUAL LEARNING RESULTS SUMMARY")
print("="*80)
print(f"{'Strategy':<12} {'AA':<10} {'BWT':<10} {'FM':<10} {'Notes':<30}")
print("-"*80)

for strategy in strategies_present:
    r = results[strategy]
    metrics = r["metrics_dsc"]
    aa = metrics["AA"]
    bwt = metrics["BWT"]
    fm = metrics["FM"]
    
    # Determine notes
    if strategy == "none":
        notes = "Baseline (no CL)"
    elif fm < 0.02:
        notes = "Excellent forgetting"
    elif fm < 0.05:
        notes = "Good forgetting"
    else:
        notes = "Moderate forgetting"
    
    print(f"{strategy:<12} {aa:<10.4f} {bwt:<10.4f} {fm:<10.4f} {notes:<30}")

print("="*80)
EOF

# Run plotting script
python plot_comparison.py
```

### Step 6: Analyze Results

```bash
# Create analysis script
cat > analyze_results.py << 'EOF'
import json
import numpy as np
from pathlib import Path

strategies = ["ewc", "lwf", "replay", "none"]
results = {}

for strategy in strategies:
    path = Path(f"results/{strategy}_results.json")
    if path.exists():
        with open(path) as f:
            results[strategy] = json.load(f)

print("\n" + "="*80)
print("DETAILED ANALYSIS")
print("="*80)

# Find best strategy for each metric
best_aa = max(results.items(), key=lambda x: x[1]["metrics_dsc"]["AA"])
best_bwt = max(results.items(), key=lambda x: x[1]["metrics_dsc"]["BWT"])
best_fm = min(results.items(), key=lambda x: x[1]["metrics_dsc"]["FM"])

print(f"\nBest Average Accuracy: {best_aa[0].upper()} ({best_aa[1]['metrics_dsc']['AA']:.4f})")
print(f"Best Backward Transfer: {best_bwt[0].upper()} ({best_bwt[1]['metrics_dsc']['BWT']:.4f})")
print(f"Best Forgetting: {best_fm[0].upper()} ({best_fm[1]['metrics_dsc']['FM']:.4f})")

# Compare to baseline
baseline = results.get("none", {})
if baseline:
    baseline_aa = baseline["metrics_dsc"]["AA"]
    baseline_fm = baseline["metrics_dsc"]["FM"]
    
    print(f"\nComparison to Baseline:")
    print(f"  Baseline AA: {baseline_aa:.4f}")
    print(f"  Baseline FM: {baseline_fm:.4f}")
    
    for strategy in ["ewc", "lwf", "replay"]:
        if strategy in results:
            r = results[strategy]
            aa_improvement = (r["metrics_dsc"]["AA"] - baseline_aa) / baseline_aa * 100
            fm_improvement = (baseline_fm - r["metrics_dsc"]["FM"]) / baseline_fm * 100
            print(f"\n  {strategy.upper()}:")
            print(f"    AA improvement: {aa_improvement:+.1f}%")
            print(f"    FM improvement: {fm_improvement:+.1f}%")

print("\n" + "="*80)
EOF

python analyze_results.py
```

### Step 7: Generate Report

```bash
# Create comprehensive report
cat > generate_report.py << 'EOF'
import json
from pathlib import Path
from datetime import datetime

strategies = ["ewc", "lwf", "replay", "none"]
results = {}

for strategy in strategies:
    path = Path(f"results/{strategy}_results.json")
    if path.exists():
        with open(path) as f:
            results[strategy] = json.load(f)

# Generate report
report = f"""
# Continual Learning Experiment Report
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Executive Summary

This report presents results from continual learning experiments on medical image segmentation.
Four strategies were evaluated: EWC, LwF, Replay, and Baseline (no CL).

## Results

### Average Accuracy (AA)
Higher is better. Measures overall performance on all tasks.

"""

for strategy in ["ewc", "lwf", "replay", "none"]:
    if strategy in results:
        aa = results[strategy]["metrics_dsc"]["AA"]
        report += f"- **{strategy.upper()}**: {aa:.4f}\n"

report += """
### Backward Transfer (BWT)
Higher is better. Measures how much new tasks affect old tasks.

"""

for strategy in ["ewc", "lwf", "replay", "none"]:
    if strategy in results:
        bwt = results[strategy]["metrics_dsc"]["BWT"]
        report += f"- **{strategy.upper()}**: {bwt:.4f}\n"

report += """
### Forgetting Measure (FM)
Lower is better. Measures catastrophic forgetting.

"""

for strategy in ["ewc", "lwf", "replay", "none"]:
    if strategy in results:
        fm = results[strategy]["metrics_dsc"]["FM"]
        report += f"- **{strategy.upper()}**: {fm:.4f}\n"

report += """
## Conclusions

See comparison plots in `results/comparison_results.png` for visual analysis.

## Files Generated

- `results/ewc_results.json` - EWC results
- `results/lwf_results.json` - LwF results
- `results/replay_results.json` - Replay results
- `results/none_results.json` - Baseline results
- `results/comparison_results.png` - Comparison plots
- `results/report.md` - This report
"""

# Save report
Path("results/report.md").write_text(report)
print("✅ Report saved: results/report.md")
EOF

python generate_report.py
```

## Resuming Interrupted Experiments

If an experiment is interrupted:

```bash
# Simply re-run the same command
python src/scripts/train_continual.py --config src/configs/ewc.yaml

# The system will:
# 1. Restore progress from WandB
# 2. Skip completed tasks
# 3. Resume from the next task
# 4. Continue logging to the same WandB run
```

## Expected Timeline

| Phase | Duration | Tasks |
|-------|----------|-------|
| Setup | 30 min | Verify environment, configs, data |
| EWC | 2-3 hours | 3 tasks × 15 epochs |
| LwF | 2-3 hours | 3 tasks × 15 epochs |
| Replay | 2-3 hours | 3 tasks × 15 epochs |
| Baseline | 2-3 hours | 3 tasks × 15 epochs |
| Analysis | 1 hour | Generate plots, analyze results |
| **Total** | **10-13 hours** | All experiments complete |

## Output Files

After completion, you'll have:

```
results/
├── ewc_results.json          # EWC results
├── lwf_results.json          # LwF results
├── replay_results.json       # Replay results
├── none_results.json         # Baseline results
├── comparison_results.png    # Comparison plots
├── report.md                 # Summary report
└── analysis.txt              # Detailed analysis

checkpoints/
├── ewc/
│   ├── cl_results.json       # Final results
│   ├── cl_progress.json      # Progress tracking
│   └── {task}/               # Per-task checkpoints
├── lwf/
│   └── ...
├── replay/
│   └── ...
└── none/
    └── ...
```

## Next Steps

1. **Review results** - Check comparison plots and metrics
2. **Statistical analysis** - Perform significance testing
3. **Ablation studies** - Test hyperparameter sensitivity
4. **Paper writing** - Document findings
5. **Code publication** - Share implementation

See related guides:
- `PROGRESS_TRACKING_GUIDE.md` - Detailed progress tracking
- `RUN_ALL_EXPERIMENTS.md` - Experiment configurations
- `METRICS_INTERPRETATION.md` - Understanding metrics
