"""
Elastic Weight Consolidation (EWC) — Kirkpatrick et al., PNAS 2017.
DOI: 10.1073/pnas.1611835114  |  9,523 citations

After completing task t, EWC estimates the Fisher information matrix F
for each parameter θ.  When training on task t+1, it adds a regularization
term that penalizes changes to parameters that were important for task t:

    L_total = L_new + (λ/2) * Σ_i  F_i * (θ_i - θ*_i)^2

where θ*_i are the optimal parameters after task t.
"""

import copy
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


class EWC:
    """
    EWC regularizer.  Usage:

        ewc = EWC(model, lambda_=1000)
        # After finishing task t:
        ewc.register_task(model, dataloader, device)
        # During task t+1 training:
        loss = criterion(pred, target) + ewc.penalty(model)
    """

    def __init__(self, model: nn.Module, lambda_: float = 1000.0):
        self.lambda_ = lambda_
        # List of (means, fishers) dicts — one entry per completed task
        self._task_params: list = []

    @torch.no_grad()
    def _copy_params(self, model: nn.Module) -> dict:
        return {n: p.clone() for n, p in model.named_parameters() if p.requires_grad}

    def register_task(self,
                      model: nn.Module,
                      dataloader: DataLoader,
                      device: torch.device,
                      criterion: Optional[nn.Module] = None,
                      num_batches: int = 200):
        """
        Compute Fisher information and save current parameters after task t.

        Args:
            model:       trained model at end of task t
            dataloader:  validation loader for task t (used to estimate Fisher)
            device:      compute device
            criterion:   loss used for Fisher estimation (default: CrossEntropy)
            num_batches: how many batches to use for Fisher estimation
        """
        if criterion is None:
            criterion = nn.CrossEntropyLoss()

        model.eval()
        means   = self._copy_params(model)
        fishers = {n: torch.zeros_like(p)
                   for n, p in model.named_parameters() if p.requires_grad}

        for i, batch in enumerate(dataloader):
            if i >= num_batches:
                break
            imgs   = batch["image"].to(device)
            labels = batch["label"].long().squeeze(1).to(device)

            model.zero_grad()
            output = model(imgs)
            loss   = criterion(output, labels)
            loss.backward()

            for n, p in model.named_parameters():
                if p.requires_grad and p.grad is not None:
                    fishers[n] += p.grad.detach() ** 2

        # Normalise
        for n in fishers:
            fishers[n] /= max(num_batches, 1)

        self._task_params.append({"means": means, "fishers": fishers})
        model.zero_grad()
        print(f"EWC: registered task {len(self._task_params)} "
              f"({len(means)} parameter tensors)")

    def penalty(self, model: nn.Module) -> torch.Tensor:
        """Return the EWC regularization loss term."""
        if not self._task_params:
            return torch.tensor(0.0)

        loss = torch.tensor(0.0, device=next(model.parameters()).device)
        for task in self._task_params:
            for n, p in model.named_parameters():
                if n in task["means"] and p.requires_grad:
                    fisher = task["fishers"][n].to(p.device)
                    mean   = task["means"][n].to(p.device)
                    loss  += (fisher * (p - mean) ** 2).sum()

        return (self.lambda_ / 2.0) * loss
