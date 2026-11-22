"""Hybrid page retrieval: reciprocal-rank fusion of OCR/BM25 and visual MaxSim.

Reciprocal Rank Fusion (Cormack, Clarke & Buettcher, SIGIR 2009) combines
rankings without calibrating their raw scores, which matters here because BM25
scores and MaxSim similarities live on unrelated scales:

    rrf(p) = sum_m  w_m / (rrf_k + rank_m(p))

Fusion is a RETRIEVAL heuristic only. The fused score is not a probability and
is not evidence that an answer exists on the page.
"""
from __future__ import annotations
from pathlib import Path
from typing import Callable, Sequence

DEFAULT_RRF_K = 60


def reciprocal_rank_fusion(rankings: Sequence[Sequence[dict]],
                           weights: Sequence[float] | None = None,
                           rrf_k: int = DEFAULT_RRF_K,
                           names: Sequence[str] | None = None) -> list[dict]:
    """Fuse ranked page lists (each a list of dicts with 'page_id').

    Pages missing from a ranking contribute nothing for that ranking. Ties are
    broken by best individual rank, then page_id, so output is deterministic.
    """
    if not rankings:
        raise ValueError("At least one ranking is required")
    weights = list(weights) if weights is not None else [1.0] * len(rankings)
    if len(weights) != len(rankings) or any(w < 0 for w in weights):
        raise ValueError("One non-negative weight per ranking is required")
    if rrf_k < 0:
        raise ValueError("rrf_k must be non-negative")
    names = list(names) if names is not None else [f"ranking_{i}" for i in range(len(rankings))]

    fused: dict[str, dict] = {}
    for name, weight, ranking in zip(names, weights, rankings):
        seen = set()
        for rank, row in enumerate(ranking, 1):
            pid = row["page_id"]
            if pid in seen:  # a duplicated page must not be counted twice
                continue
            seen.add(pid)
            entry = fused.setdefault(pid, {"row": row, "score": 0.0,
                                           "component_ranks": {}, "best_rank": rank})
            entry["score"] += weight / (rrf_k + rank)
            entry["component_ranks"][name] = rank
            entry["best_rank"] = min(entry["best_rank"], rank)
    ordered = sorted(fused.items(),
                     key=lambda kv: (-kv[1]["score"], kv[1]["best_rank"], kv[0]))
    return [{**entry["row"], "score": entry["score"], "mode": "hybrid-rrf",
             "component_ranks": entry["component_ranks"]} for _, entry in ordered]


def hybrid_search(question: str, corpus_dir: str | Path, backend=None, k: int = 3,
                  depth: int | None = None, rrf_k: int = DEFAULT_RRF_K,
                  weights: tuple[float, float] = (1.0, 1.0),
                  lexical_fn: Callable | None = None,
                  visual_fn: Callable | None = None) -> list[dict]:
    """Rank pages by fusing BM25 and visual MaxSim over the SAME page pool.

    `depth` is how many candidates each retriever contributes (default: all
    pages). If the corpus has no OCR/PDF text, BM25 raises and fusion degrades
    to visual-only with an explicit warning instead of silently pretending.
    """
    from .corpus import load_corpus
    if lexical_fn is None:
        from .search import lexical_search as lexical_fn
    if visual_fn is None:
        from .vision import retrieve_visual as visual_fn
    if backend is None and visual_fn.__module__.endswith("vision"):
        from .vision import ColSmolRetriever
        backend = ColSmolRetriever()
    pool = depth or len(load_corpus(corpus_dir))
    if pool <= 0:
        return []
    visual = visual_fn(question, corpus_dir, backend, pool)
    warning = ""
    try:
        lexical = lexical_fn(question, corpus_dir, k=pool)
        # Zero BM25 score means no term overlap; ranking those pages adds noise.
        lexical = [r for r in lexical if r.get("score", 0) > 0]
    except ValueError as exc:
        lexical, warning = [], f"BM25 unavailable ({exc}); visual-only ranking"
    fused = reciprocal_rank_fusion([lexical, visual], weights=weights, rrf_k=rrf_k,
                                   names=["bm25", "visual"])
    return [{**row, "fusion_warning": warning} for row in fused[:k]]
