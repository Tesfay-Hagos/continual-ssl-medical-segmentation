"""
Evaluation metrics for segmentation and continual learning.

Segmentation:
  - Dice Similarity Coefficient (DSC)
  - Hausdorff Distance 95th percentile (HD95)
  - Intersection over Union (IoU)

Continual Learning:
  - Backward Transfer  (BWT)  — did we forget previous tasks?
  - Forward Transfer   (FWT)  — did pretraining help unseen tasks?
  - Forgetting Measure (F)    — worst-case per-task forgetting
  - Average Accuracy   (AA)   — mean DSC across all tasks after final training

The R matrix:  R[t][i] = DSC on task i evaluated after training on task t.
"""

import numpy as np
from typing import Dict, List

import torch
from monai.metrics import DiceMetric, HausdorffDistanceMetric
from monai.transforms import AsDiscrete
from monai.data import decollate_batch


# ── Per-sample segmentation metrics ──────────────────────────────────────────

class SegmentationEvaluator:
    """Accumulates DSC and HD95 over a validation loader."""

    def __init__(self, num_classes: int = 2):
        self.num_classes   = num_classes
        self.dice_metric   = DiceMetric(include_background=False, reduction="mean")
        self.hd95_metric   = HausdorffDistanceMetric(include_background=False,
                                                      percentile=95,
                                                      reduction="mean")
        self.post_pred  = AsDiscrete(argmax=True, to_onehot=num_classes)
        self.post_label = AsDiscrete(to_onehot=num_classes)

    def reset(self):
        self.dice_metric.reset()
        self.hd95_metric.reset()

    @torch.no_grad()
    def update(self, pred: torch.Tensor, label: torch.Tensor):
        """
        pred:  raw logits  (B, C, D, H, W)
        label: integer mask (B, 1, D, H, W)
        """
        pred_list  = [self.post_pred(p)  for p in decollate_batch(pred)]
        label_list = [self.post_label(l) for l in decollate_batch(label)]
        self.dice_metric(y_pred=pred_list, y=label_list)
        self.hd95_metric(y_pred=pred_list, y=label_list)

    def aggregate(self) -> Dict[str, float]:
        dice = self.dice_metric.aggregate().item()
        try:
            hd95 = self.hd95_metric.aggregate().item()
        except Exception:
            hd95 = float("nan")
        self.reset()
        return {"dice": dice, "hd95": hd95}


# ── Continual learning metrics ────────────────────────────────────────────────

def backward_transfer(R: np.ndarray) -> float:
    """
    BWT = (1 / (T-1)) * Σ_{i=1}^{T-1}  (R[T][i] - R[i][i])

    Negative BWT = forgetting.
    R: (T, T) matrix, R[t][i] = DSC on task i after training on task t.
    """
    T = R.shape[0]
    if T < 2:
        return 0.0
    vals = [R[T - 1, i] - R[i, i] for i in range(T - 1)]
    return float(np.mean(vals))


def forward_transfer(R: np.ndarray, R0: np.ndarray) -> float:
    """
    FWT = (1 / (T-1)) * Σ_{i=2}^{T}  (R[i-1][i] - R0[i])

    R0[i] = DSC of random-init model on task i (no training).
    Positive FWT = useful knowledge transfer to unseen tasks.
    """
    T = R.shape[0]
    if T < 2:
        return 0.0
    vals = [R[i - 1, i] - R0[i] for i in range(1, T)]
    return float(np.mean(vals))


def forgetting_measure(R: np.ndarray) -> float:
    """
    F = (1 / (T-1)) * Σ_{i=1}^{T-1}  max_{j ∈ {1..T-1}} (R[j][i] - R[T-1][i])

    Measures the maximum performance drop on each previous task.
    """
    T = R.shape[0]
    if T < 2:
        return 0.0
    vals = []
    for i in range(T - 1):
        peak = max(R[j, i] for j in range(i, T))
        vals.append(peak - R[T - 1, i])
    return float(np.mean(vals))


def average_accuracy(R: np.ndarray) -> float:
    """Mean DSC across all tasks after training on the final task."""
    return float(np.mean(R[-1, :]))


def print_cl_metrics(R: np.ndarray, task_names: List[str], R0: np.ndarray = None):
    T = R.shape[0]
    print("\n── Continual Learning Results ──────────────────────────────")
    header = f"{'':15s}" + "".join(f"{n:>12s}" for n in task_names)
    print(header)
    for t in range(T):
        row = f"After task {t+1:<4}" + "".join(f"{R[t,i]:>12.3f}" for i in range(T))
        print(row)
    print()
    print(f"  Average Accuracy (AA)  : {average_accuracy(R):.4f}")
    print(f"  Backward Transfer (BWT): {backward_transfer(R):.4f}")
    print(f"  Forgetting Measure  (F): {forgetting_measure(R):.4f}")
    if R0 is not None:
        print(f"  Forward Transfer  (FWT): {forward_transfer(R, R0):.4f}")
    print("────────────────────────────────────────────────────────────\n")
