"""
Experience Replay buffer for continual learning.

Stores a fixed-size reservoir of (image, label, task_id) tuples from
previously seen tasks.  During new-task training, a random mini-batch
from the buffer is mixed into each training step.

Reservoir sampling ensures uniform coverage regardless of task length.
"""

import random
from typing import Optional, Tuple, List

import torch
from torch.utils.data import DataLoader


class ReplayBuffer:
    """
    Fixed-size experience replay buffer.

    Usage:
        buf = ReplayBuffer(capacity=200)
        # While training on task t, after each batch:
        buf.add_batch(imgs, labels, task_id=t)
        # During task t+1 training:
        r_imgs, r_labels = buf.sample(batch_size=4)
        loss += criterion(model(r_imgs), r_labels)
    """

    def __init__(self, capacity: int = 200, device: Optional[torch.device] = None):
        self.capacity = capacity
        self.device   = device or torch.device("cpu")
        self._storage: List[Tuple[torch.Tensor, torch.Tensor, int]] = []
        self._n_seen  = 0

    def __len__(self):
        return len(self._storage)

    def add_batch(self,
                  images: torch.Tensor,
                  labels: torch.Tensor,
                  task_id: int):
        """
        Add a batch to the buffer using reservoir sampling so that each
        observed example has equal probability of being in the buffer.
        """
        images = images.detach().cpu()
        labels = labels.detach().cpu()

        for i in range(images.shape[0]):
            self._n_seen += 1
            if len(self._storage) < self.capacity:
                self._storage.append((images[i], labels[i], task_id))
            else:
                j = random.randint(0, self._n_seen - 1)
                if j < self.capacity:
                    self._storage[j] = (images[i], labels[i], task_id)

    def sample(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return a random batch of (images, labels) from the buffer."""
        if len(self._storage) == 0:
            return None, None
        n      = min(batch_size, len(self._storage))
        items  = random.sample(self._storage, n)
        imgs   = torch.stack([x[0] for x in items]).to(self.device)
        labels = torch.stack([x[1] for x in items]).to(self.device)
        return imgs, labels

    def populate_from_loader(self,
                             loader: DataLoader,
                             task_id: int,
                             max_batches: int = 50):
        """Fill the buffer from a dataloader at the end of a task."""
        for i, batch in enumerate(loader):
            if i >= max_batches:
                break
            self.add_batch(batch["image"], batch["label"], task_id)
        print(f"Replay: buffer size after task {task_id}: {len(self._storage)}")

    def state_dict(self) -> dict:
        return {"storage": self._storage, "n_seen": self._n_seen,
                "capacity": self.capacity}

    def load_state_dict(self, state: dict):
        self._storage = state["storage"]
        self._n_seen  = state["n_seen"]
        self.capacity = state["capacity"]
