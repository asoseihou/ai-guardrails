"""PyTorch model: hashed n-gram bag (EmbeddingBag) followed by a small MLP.

An ``EmbeddingBag`` in ``mean`` mode over hashed n-gram ids is a linear layer
over a normalized bag-of-n-grams vector, computed sparsely (fastText-style).
"""

from __future__ import annotations

import torch
from torch import nn


class DocTypeMLP(nn.Module):
    """Mean EmbeddingBag over hashed n-grams followed by a one-hidden-layer MLP."""

    def __init__(
        self,
        n_buckets: int,
        n_classes: int,
        emb_dim: int = 48,
        hidden: int = 64,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.embed = nn.EmbeddingBag(n_buckets, emb_dim, mode="mean")
        self.mlp = nn.Sequential(
            nn.Linear(emb_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, n_classes),
        )

    def forward(self, ids: torch.Tensor, offsets: torch.Tensor) -> torch.Tensor:
        """Class logits for a packed batch (see :func:`pack`)."""
        logits: torch.Tensor = self.mlp(self.embed(ids, offsets))
        return logits


def pack(batch: list[list[int]]) -> tuple[torch.Tensor, torch.Tensor]:
    """Flatten a list of id lists into (ids, offsets) for EmbeddingBag."""
    offsets, total = [], 0
    for ids in batch:
        offsets.append(total)
        total += len(ids)
    flat = [i for ids in batch for i in ids]
    return torch.tensor(flat, dtype=torch.long), torch.tensor(offsets, dtype=torch.long)
