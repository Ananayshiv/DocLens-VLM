"""Reimplementation of ColBERT/ColPali MaxSim late-interaction matching.

S(q,d) = sum_i max_j cosine(q_i, d_j) over VALID tokens.
Unlike a global image embedding, each query token may attend to a different
patch/token in the retrieved page. Excludes padded query/document tokens.
"""
from __future__ import annotations
import torch
import torch.nn.functional as F


def maxsim(query: torch.Tensor, document: torch.Tensor,
           query_mask: torch.Tensor | None = None,
           document_mask: torch.Tensor | None = None,
           normalize: bool = True, chunk_size: int = 512) -> torch.Tensor:
    """Compute unnormalized MaxSim score for one query-page pair.

    query: [Q, D]; document: [P, D]. A score is not a calibrated probability.
    """
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")
    if query.ndim != 2 or document.ndim != 2 or query.shape[1] != document.shape[1]:
        raise ValueError("Expected embeddings [query_tokens, dim] and [page_tokens, dim]")
    if query.shape[0] == 0 or document.shape[0] == 0:
        raise ValueError("Token sequences cannot be empty")
    if query_mask is None:
        query_mask = torch.ones(query.shape[0], dtype=torch.bool, device=query.device)
    if document_mask is None:
        document_mask = torch.ones(document.shape[0], dtype=torch.bool, device=document.device)
    if query_mask.shape != (query.shape[0],) or document_mask.shape != (document.shape[0],):
        raise ValueError("Mask lengths must match token counts")
    if not query_mask.any() or not document_mask.any():
        raise ValueError("Each sequence must have at least one unmasked token")
    q = query.float()
    d = document.to(query.device).float()
    if not torch.isfinite(q).all() or not torch.isfinite(d).all():
        raise ValueError("Embeddings must be finite")
    if normalize:
        q, d = F.normalize(q, p=2, dim=-1), F.normalize(d, p=2, dim=-1)
    valid_doc = document_mask.to(query.device).bool()
    # Prevent a Q×P dense allocation for high-resolution page embeddings.
    maxima = None
    for i in range(0, d.shape[0], chunk_size):
        sim = q @ d[i:i + chunk_size].T
        sim = sim.masked_fill(~valid_doc[i:i + chunk_size][None, :], -torch.inf)
        local = sim.max(dim=-1).values
        maxima = local if maxima is None else torch.maximum(maxima, local)
    return maxima[query_mask.to(query.device).bool()].sum()


def score_pages(query: torch.Tensor, pages: list[torch.Tensor]) -> torch.Tensor:
    if not pages:
        raise ValueError("No indexed pages")
    return torch.stack([maxsim(query, page) for page in pages])


def contrastive_loss(query_vectors: list[torch.Tensor],
                     page_vectors: list[torch.Tensor], temperature: float = 0.07) -> torch.Tensor:
    """In-batch InfoNCE with paired diagonal positives. Research training primitive.

    Requires training-time gradients; does NOT itself fine-tune a pretrained VLM.
    """
    if len(query_vectors) != len(page_vectors) or len(query_vectors) < 2:
        raise ValueError("Provide at least two matched query-page pairs")
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    logits = torch.stack([
        torch.stack([maxsim(q, p) for p in page_vectors])
        for q in query_vectors
    ]) / temperature
    targets = torch.arange(len(query_vectors), device=logits.device)
    return (F.cross_entropy(logits, targets) + F.cross_entropy(logits.T, targets)) / 2
