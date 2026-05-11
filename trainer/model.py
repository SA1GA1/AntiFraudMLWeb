"""FraudMLP: categorical embeddings + numerical + aggregate + has_history → 1 logit."""

from __future__ import annotations

import math
from typing import List

import torch
from torch import nn


def _emb_dim(vocab_size: int) -> int:
    return min(32, max(2, int(math.ceil(math.sqrt(vocab_size)))))


class FraudMLP(nn.Module):
    def __init__(
        self,
        cat_vocab_sizes: List[int],
        n_numeric: int,
        n_aggregate: int,
        hidden: tuple[int, ...] = (256, 128, 64),
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.cat_vocab_sizes = list(cat_vocab_sizes)
        self.embeddings = nn.ModuleList(
            [nn.Embedding(v, _emb_dim(v)) for v in cat_vocab_sizes]
        )
        emb_total = sum(_emb_dim(v) for v in cat_vocab_sizes)
        input_dim = emb_total + n_numeric + n_aggregate + 1  # +1 for has_history

        layers: list[nn.Module] = []
        prev = input_dim
        for i, h in enumerate(hidden):
            layers.append(nn.Linear(prev, h))
            if i < len(hidden) - 1:
                layers.append(nn.BatchNorm1d(h))
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(dropout))
            else:
                layers.append(nn.ReLU())
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.body = nn.Sequential(*layers)

    def forward(
        self,
        num: torch.Tensor,
        cat: torch.Tensor,
        agg: torch.Tensor,
        has_hist: torch.Tensor,
    ) -> torch.Tensor:
        embs = [emb(cat[:, i]) for i, emb in enumerate(self.embeddings)]
        x = torch.cat(embs + [num, agg, has_hist], dim=1)
        return self.body(x).squeeze(-1)
