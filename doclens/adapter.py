"""Small trainable MaxSim-domain adapter inspired by parameter-efficient adaptation.

This is OUR lightweight experimental extension to ColPali, not part of the
original paper. Trains on pretrained frozen token embeddings, not VLM weights.
"""
from __future__ import annotations
import torch
from torch import nn
import torch.nn.functional as F
from .late_interaction import maxsim


class LowRankTokenAdapter(nn.Module):
    def __init__(self, dim: int, rank: int = 8):
        super().__init__()
        if dim < 1 or not (0 < rank <= dim):
            raise ValueError("Expected positive dim and rank <= dim")
        self.down = nn.Linear(dim, rank, bias=False)
        self.up = nn.Linear(rank, dim, bias=False)
        nn.init.normal_(self.down.weight, std=.02)
        nn.init.zeros_(self.up.weight)
        self.scale = 1 / rank

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        return F.normalize(embeddings + self.scale * self.up(self.down(embeddings)), dim=-1)


def adapter_scores(adapter: LowRankTokenAdapter,
                   queries: list[torch.Tensor], pages: list[torch.Tensor]) -> torch.Tensor:
    encoded_queries = [adapter(x) for x in queries]
    encoded_pages = [adapter(x) for x in pages]
    return torch.stack([torch.stack([maxsim(q, p, normalize=False) for p in encoded_pages])
                        for q in encoded_queries])


def inbatch_adapter_step(adapter: LowRankTokenAdapter, optimizer: torch.optim.Optimizer,
                         queries: list[torch.Tensor], pages: list[torch.Tensor],
                         temperature: float = 0.15) -> float:
    if len(queries) < 2 or len(pages) != len(queries) or temperature <= 0:
        raise ValueError("Provide >=2 aligned query-page pairs and positive temperature")
    scores = adapter_scores(adapter, queries, pages) / temperature
    gold = torch.arange(len(queries), device=scores.device)
    loss = (F.cross_entropy(scores, gold) + F.cross_entropy(scores.T, gold)) / 2
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.detach().item()
