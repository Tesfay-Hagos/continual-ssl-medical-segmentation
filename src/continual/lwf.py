"""
Learning without Forgetting (LwF) — Li & Hoiem, ECCV 2016.
DOI: 10.1007/978-3-319-46493-0_37  |  5,411 citations

LwF uses knowledge distillation to preserve the model's predictions
on previous tasks while training on a new task.  No past data is stored.

For segmentation, we preserve the softmax output distribution of the
model at the end of each task t, and add a KL-divergence term:

    L_total = L_new(t+1) + α * Σ_{j≤t} KL(p_old_j || p_new_j)

where p_old_j is the teacher (frozen previous model) output on the
current batch and p_new_j is the current model's output.
"""

import copy
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F


class LwF:
    """
    LwF regularizer for segmentation.  Usage:

        lwf = LwF(alpha=1.0, temperature=2.0)
        # After finishing task t:
        lwf.register_task(model)
        # During task t+1 training, pass current batch images:
        loss = criterion(pred, target) + lwf.distillation_loss(model, imgs)
    """

    def __init__(self, alpha: float = 1.0, temperature: float = 2.0):
        self.alpha       = alpha
        self.temperature = temperature
        self._teachers: List[nn.Module] = []   # frozen snapshots of the model

    def register_task(self, model: nn.Module):
        """Save a frozen copy of the model after completing task t."""
        teacher = copy.deepcopy(model)
        teacher.eval()
        for p in teacher.parameters():
            p.requires_grad = False
        self._teachers.append(teacher)
        print(f"LwF: registered teacher #{len(self._teachers)}")

    def distillation_loss(self,
                          model: nn.Module,
                          imgs: torch.Tensor) -> torch.Tensor:
        """
        Compute KL-divergence distillation loss against all saved teachers.

        Args:
            model: current (student) model
            imgs:  current batch of images (no labels needed)
        """
        if not self._teachers:
            return torch.tensor(0.0, device=imgs.device)

        T = self.temperature
        loss = torch.tensor(0.0, device=imgs.device)

        with torch.no_grad():
            teacher_preds = [
                F.softmax(teacher(imgs) / T, dim=1)
                for teacher in self._teachers
            ]

        student_log = F.log_softmax(model(imgs) / T, dim=1)

        for t_pred in teacher_preds:
            # KL(teacher || student) with per-voxel scaling to match segmentation loss scale
            n_voxels = imgs.shape[2] * imgs.shape[3] * imgs.shape[4]  # D*H*W
            kl = F.kl_div(student_log, t_pred, reduction="sum") / (imgs.shape[0] * n_voxels)
            loss += (T ** 2) * kl   # temperature scaling restores gradient magnitude

        return self.alpha * loss / max(len(self._teachers), 1)
