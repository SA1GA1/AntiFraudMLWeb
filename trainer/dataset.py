"""PyTorch Dataset that holds preprocessed tensors and implements
aggregate dropout for cold-start robustness."""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset


class FraudDataset(Dataset):
    def __init__(
        self,
        num: np.ndarray,        # (n, n_num) float32
        cat: np.ndarray,        # (n, n_cat) int64
        agg: np.ndarray,        # (n, n_agg) float32
        has_history: np.ndarray,  # (n, 1) float32
        target: np.ndarray | None,  # (n,) float32, or None for inference
        train_mode: bool = False,
        agg_dropout: float = 0.20,
        seed: int = 42,
    ) -> None:
        self.num = torch.from_numpy(np.ascontiguousarray(num))
        self.cat = torch.from_numpy(np.ascontiguousarray(cat))
        self.agg = torch.from_numpy(np.ascontiguousarray(agg))
        self.has_hist = torch.from_numpy(np.ascontiguousarray(has_history))
        if target is not None:
            self.target = torch.from_numpy(np.ascontiguousarray(target.astype(np.float32)))
        else:
            self.target = None
        self.train_mode = train_mode
        self.agg_dropout = float(agg_dropout)
        # Per-instance deterministic RNG so dropout is reproducible across epochs
        # when train_mode is False or we want eval determinism.
        self._rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return self.num.shape[0]

    def __getitem__(self, idx: int):
        num = self.num[idx]
        cat = self.cat[idx]
        agg = self.agg[idx]
        has_hist = self.has_hist[idx]

        if self.train_mode and self.agg_dropout > 0.0:
            # Per-sample fresh draw — torch will reshuffle each epoch, this
            # gives stochastic masking across epochs.
            if torch.rand(1).item() < self.agg_dropout:
                agg = torch.zeros_like(agg)
                has_hist = torch.zeros_like(has_hist)

        if self.target is None:
            return num, cat, agg, has_hist
        return num, cat, agg, has_hist, self.target[idx]
