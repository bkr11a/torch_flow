from __future__ import annotations
from typing import List, Optional, Sequence
import numpy as np
import torch
from torch.utils.data import Dataset

class WeightedMixedFlowDataset(Dataset):
    """Replay-weighted optical-flow dataset wrapper.

    Unlike ConcatDataset, this samples child datasets according to explicit
    probabilities rather than according to dataset length.
    """
    def __init__(
        self,
        datasets: Sequence[Dataset],
        weights: Sequence[float],
        names: Optional[Sequence[str]] = None,
        epoch_size: int = 50000,
        deterministic: bool = False,
        seed: int = 12345,
        include_dataset_id: bool = True,
    ) -> None:
        if len(datasets) == 0:
            raise ValueError("WeightedMixedFlowDataset requires at least one child dataset")
        if len(datasets) != len(weights):
            raise ValueError(f"datasets/weights mismatch: {len(datasets)} vs {len(weights)}")
        empty = [i for i, ds in enumerate(datasets) if len(ds) == 0]
        if empty:
            raise ValueError(f"Empty child datasets: {empty}")

        w = np.asarray(weights, dtype=np.float64)
        if not np.all(np.isfinite(w)) or np.any(w < 0) or w.sum() <= 0:
            raise ValueError(f"Invalid weights: {weights}")

        self.datasets: List[Dataset] = list(datasets)
        self.weights_np = w / w.sum()
        self.probs = torch.as_tensor(self.weights_np, dtype=torch.float32)
        self.names = list(names) if names is not None else [f"dataset_{i}" for i in range(len(datasets))]
        self.epoch_size = int(epoch_size)
        self.deterministic = bool(deterministic)
        self.seed = int(seed)
        self.include_dataset_id = bool(include_dataset_id)
        if self.epoch_size <= 0:
            raise ValueError(f"epoch_size must be positive, got {epoch_size}")

        self._plan = None
        if self.deterministic:
            rng = np.random.default_rng(self.seed)
            dset_ids = rng.choice(len(self.datasets), size=self.epoch_size, replace=True, p=self.weights_np)
            self._plan = []
            for dset_id in dset_ids:
                sample_id = int(rng.integers(0, len(self.datasets[int(dset_id)])))
                self._plan.append((int(dset_id), sample_id))

    def __len__(self) -> int:
        return self.epoch_size

    def __getitem__(self, idx: int):
        if self.deterministic:
            assert self._plan is not None
            dset_id, sample_id = self._plan[int(idx) % self.epoch_size]
        else:
            dset_id = int(torch.multinomial(self.probs, 1, replacement=True).item())
            sample_id = int(torch.randint(len(self.datasets[dset_id]), (1,)).item())

        sample = self.datasets[dset_id][sample_id]
        if self.include_dataset_id:
            sample = dict(sample)
            sample["dataset_id"] = torch.tensor(dset_id, dtype=torch.long)
        return sample

    def summary(self) -> str:
        return "\n".join(
            f"{i:02d}: {name:24s} weight={w:.4f} len={len(ds)}"
            for i, (name, ds, w) in enumerate(zip(self.names, self.datasets, self.weights_np))
        )

    def __repr__(self) -> str:
        return (
            f"WeightedMixedFlowDataset(epoch_size={self.epoch_size}, "
            f"deterministic={self.deterministic}, datasets={len(self.datasets)})\n"
            + self.summary()
        )
