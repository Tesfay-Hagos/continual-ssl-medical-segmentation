# Continual Learning Metrics Interpretation Guide

## Overview

This guide explains the metrics used to evaluate continual learning performance and how to interpret them.

## Core Metrics

### 1. Dice Similarity Coefficient (DSC)

**Definition**: Overlap between predicted and ground truth segmentation

$$\text{DSC} = \frac{2|X \cap Y|}{|X| + |Y|}$$

**Range**: 0 to 1 (higher is better)

**Interpretation**:
- **0.90+**: Excellent segmentation
- **0.80-0.90**: Good segmentation
- **0.70-0.80**: Acceptable segmentation
- **<0.70**: Poor segmentation

**Example**:
```
Liver segmentation DSC = 0.85
→ 85% overlap between predicted and ground truth
→ Good performance
```

### 2. Hausdorff Distance (HD95)

**Definition**: 95th percentile of distances between predicted and ground truth boundaries

**Range**: 0 to ∞ (lower is better, in mm)

**Interpretation**:
- **<5 mm**: Excellent boundary accuracy
- **5-10 mm**: Good boundary accuracy
- **10-20 mm**: Acceptable boundary accuracy
- **>20 mm**: Poor boundary accuracy

**Example**:
```
Liver HD95 = 5.2 mm
→ 95% of boundary points are within 5.2 mm
→ Good boundary accuracy
```

## Continual Learning Metrics

### 1. Average Accuracy (AA)

**Definition**: Mean performance on all tasks after learning all tasks

$$\text{AA} = \frac{1}{T} \sum_{i=1}^{T} R[T, i]$$

Where `R[T, i]` is performance on task i after learning all T tasks.

**Range**: 0 to 1 (higher is better)

**Interpretation**:
- **>0.80**: Excellent continual learning
- **0.75-0.80**: Good continual learning
- **0.70-0.75**: Acceptable continual learning
- **<0.70**: Poor continual learning

**Example**:
```
Tasks: [Liver, Pancreas, Heart]
After all tasks:
  Liver:    0.80
  Pancreas: 0.75
  Heart:    0.81
AA = (0.80 + 0.75 + 0.81) / 3 = 0.787
```

### 2. Backward Transfer (BWT)

**Definition**: How much learning new tasks affects performance on old tasks

$$\text{BWT} = \frac{1}{T-1} \sum_{i=1}^{T-1} (R[T, i] - R[i, i])$$

Where:
- `R[T, i]` = performance on task i after learning all tasks
- `R[i, i]` = performance on task i right after learning it

**Range**: -1 to 1 (higher is better, 0 is neutral)

**Interpretation**:
- **>0.0**: Positive transfer (new tasks help old tasks)
- **≈0.0**: No transfer (new tasks don't affect old tasks)
- **<0.0**: Negative transfer (new tasks hurt old tasks)

**Example**:
```
Liver performance:
  After Task 1: 0.85
  After Task 3: 0.80
  Forgetting: 0.85 - 0.80 = 0.05

Pancreas performance:
  After Task 2: 0.78
  After Task 3: 0.75
  Forgetting: 0.78 - 0.75 = 0.03

BWT = -(0.05 + 0.03) / 2 = -0.04
→ Negative transfer: new tasks hurt old tasks by 4%
```

### 3. Forgetting Measure (FM)

**Definition**: Average performance drop on old tasks after learning new tasks

$$\text{FM} = \frac{1}{T-1} \sum_{i=1}^{T-1} (R[i, i] - R[T, i])$$

Where:
- `R[i, i]` = performance on task i right after learning it
- `R[T, i]` = performance on task i after learning all tasks

**Range**: 0 to 1 (lower is better)

**Interpretation**:
- **<0.02**: Excellent (minimal forgetting)
- **0.02-0.05**: Good (acceptable forgetting)
- **0.05-0.10**: Moderate (noticeable forgetting)
- **>0.10**: Poor (significant forgetting)

**Example**:
```
Liver forgetting: 0.85 - 0.80 = 0.05
Pancreas forgetting: 0.78 - 0.75 = 0.03
FM = (0.05 + 0.03) / 2 = 0.04
→ Average 4% performance drop (acceptable)
```

### 4. Forward Transfer (FWT)

**Definition**: How much learning previous tasks helps new tasks

$$\text{FWT} = \frac{1}{T-1} \sum_{i=2}^{T} (R[i, i] - R_{\text{scratch}}[i])$$

Where:
- `R[i, i]` = performance on task i after learning previous tasks
- `R_{\text{scratch}}[i]` = performance on task i trained from scratch

**Range**: -1 to 1 (higher is better)

**Interpretation**:
- **>0.0**: Positive transfer (previous tasks help)
- **≈0.0**: No transfer
- **<0.0**: Negative transfer (previous tasks hurt)

**Note**: Requires training each task from scratch for comparison.

## Results Matrix (R)

### Structure

The results matrix tracks performance on all tasks after learning each task:

```
       Task 1  Task 2  Task 3
After Task 1:  0.85    -       -
After Task 2:  0.82    0.78    -
After Task 3:  0.80    0.75    0.81
```

**Rows**: After learning task t
**Columns**: Performance on task i
**Diagonal**: Performance right after learning each task
**Below diagonal**: Performance after learning new tasks (forgetting)

### Reading the Matrix

```
R[1,1] = 0.85  → Liver performance after Task 1
R[2,1] = 0.82  → Liver performance after Task 2 (↓ 0.03 forgetting)
R[2,2] = 0.78  → Pancreas performance after Task 2
R[3,1] = 0.80  → Liver performance after Task 3 (↓ 0.05 total forgetting)
R[3,2] = 0.75  → Pancreas performance after Task 3 (↓ 0.03 forgetting)
R[3,3] = 0.81  → Heart performance after Task 3
```

## Strategy Comparison

### EWC (Elastic Weight Consolidation)

**Mechanism**: Protects important weights using Fisher information

**Expected metrics**:
- AA: 0.78-0.82
- BWT: -0.02 to 0.00
- FM: 0.03-0.08

**Characteristics**:
- Moderate forgetting
- Good backward transfer
- Reasonable average accuracy
- Computationally efficient

### LwF (Learning without Forgetting)

**Mechanism**: Uses knowledge distillation from previous model

**Expected metrics**:
- AA: 0.76-0.80
- BWT: -0.01 to 0.01
- FM: 0.01-0.03

**Characteristics**:
- Very low forgetting
- Minimal backward transfer
- Slightly lower average accuracy
- Fast training

### Experience Replay

**Mechanism**: Stores and replays samples from previous tasks

**Expected metrics**:
- AA: 0.80-0.85
- BWT: 0.00 to 0.02
- FM: 0.00-0.02

**Characteristics**:
- Minimal forgetting
- Positive backward transfer
- Best average accuracy
- Requires memory storage

### Baseline (No Continual Learning)

**Mechanism**: Standard sequential training without any strategy

**Expected metrics**:
- AA: 0.70-0.75
- BWT: -0.08 to -0.04
- FM: 0.10-0.20

**Characteristics**:
- High forgetting
- Negative backward transfer
- Lower average accuracy
- Baseline for comparison

## Interpreting Results

### Good Continual Learning

```
AA:  0.80+  (high average accuracy)
BWT: -0.02  (minimal negative transfer)
FM:  0.03   (low forgetting)
```

**Interpretation**: Strategy successfully learns new tasks while preserving old knowledge.

### Poor Continual Learning

```
AA:  0.70   (low average accuracy)
BWT: -0.10  (high negative transfer)
FM:  0.15   (high forgetting)
```

**Interpretation**: Strategy suffers from catastrophic forgetting.

### Trade-offs

```
Strategy A: AA=0.82, FM=0.05  (high accuracy, moderate forgetting)
Strategy B: AA=0.78, FM=0.02  (lower accuracy, low forgetting)
```

**Decision**: Choose based on application requirements:
- If accuracy is critical → Strategy A
- If stability is critical → Strategy B

## Practical Examples

### Example 1: Medical Imaging

```
Task sequence: Liver → Pancreas → Heart

Baseline (no CL):
  AA: 0.72, BWT: -0.08, FM: 0.12
  → Poor: loses 12% accuracy on old tasks

EWC:
  AA: 0.80, BWT: -0.02, FM: 0.05
  → Good: maintains 80% accuracy, only 5% forgetting

Replay:
  AA: 0.82, BWT: 0.01, FM: 0.02
  → Excellent: maintains 82% accuracy, minimal forgetting
```

**Recommendation**: Use Replay for best performance, EWC for memory efficiency.

### Example 2: Hyperparameter Sensitivity

```
EWC with different lambda values:

lambda=500:
  AA: 0.78, FM: 0.08
  → Weak regularization, more forgetting

lambda=1000:
  AA: 0.80, FM: 0.05
  → Good balance

lambda=5000:
  AA: 0.79, FM: 0.04
  → Strong regularization, less forgetting but lower AA
```

**Recommendation**: lambda=1000 provides best balance.

## Visualization Guide

### Plot 1: Average Accuracy Comparison

```
Strategy comparison:
  Baseline: 0.72
  EWC:      0.80 ✓
  LwF:      0.78
  Replay:   0.82 ✓✓
```

**Interpretation**: Replay and EWC outperform baseline.

### Plot 2: Forgetting Measure

```
Forgetting over tasks:
  Baseline: 0.12 (high)
  EWC:      0.05 (moderate)
  LwF:      0.02 (low)
  Replay:   0.01 (minimal)
```

**Interpretation**: Replay and LwF effectively prevent forgetting.

### Plot 3: Results Matrix Heatmap

```
Color intensity = performance (green=high, red=low)

Baseline:
  [0.85  -    -  ]
  [0.70  0.78 -  ]  ← Liver drops from 0.85 to 0.70
  [0.60  0.65 0.81]  ← Severe forgetting

EWC:
  [0.85  -    -  ]
  [0.82  0.78 -  ]  ← Liver drops from 0.85 to 0.82
  [0.80  0.75 0.81]  ← Minimal forgetting
```

**Interpretation**: EWC preserves old task performance much better.

## Common Issues and Solutions

### Issue 1: High Forgetting (FM > 0.10)

**Possible causes**:
- Continual learning strategy not working
- Learning rate too high
- Not enough regularization

**Solutions**:
- Increase regularization strength (lambda for EWC)
- Reduce learning rate
- Use stronger strategy (Replay instead of EWC)

### Issue 2: Low Average Accuracy (AA < 0.70)

**Possible causes**:
- Model not learning new tasks
- Too much regularization
- Data quality issues

**Solutions**:
- Reduce regularization strength
- Increase learning rate
- Check data preprocessing

### Issue 3: Negative Backward Transfer (BWT < -0.05)

**Possible causes**:
- New tasks interfering with old tasks
- Insufficient task separation
- Model capacity issues

**Solutions**:
- Use stronger continual learning strategy
- Increase model capacity
- Adjust task order

## Summary

**Key metrics for continual learning**:
1. **AA** - Overall performance (higher is better)
2. **BWT** - Backward transfer (higher is better)
3. **FM** - Forgetting (lower is better)

**Good continual learning**:
- AA > 0.80
- BWT > -0.05
- FM < 0.05

**Strategy selection**:
- **Baseline**: Comparison only
- **EWC**: Good balance, memory efficient
- **LwF**: Low forgetting, fast
- **Replay**: Best performance, requires memory

See `RUN_ALL_EXPERIMENTS.md` for running experiments and collecting results.
