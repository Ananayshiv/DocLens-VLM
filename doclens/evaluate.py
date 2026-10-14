"""Page-level retrieval benchmark. No fabricated model metrics."""
from __future__ import annotations
import json
import math
import random
from pathlib import Path
from statistics import mean


def reciprocal_rank(retrieved_ids: list[str], relevant_ids: set[str]) -> float:
    return next((1 / rank for rank, page_id in enumerate(retrieved_ids, 1)
                 if page_id in relevant_ids), 0.)


def recall_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> float:
    if not relevant_ids:
        raise ValueError("At least one gold relevant page needed")
    return len(set(retrieved_ids[:k]) & relevant_ids) / len(relevant_ids)


def ndcg_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> float:
    if not relevant_ids:
        raise ValueError("Empty gold qrels")
    dcg = sum(1 / math.log2(rank + 1) for rank, page_id
              in enumerate(retrieved_ids[:k], 1) if page_id in relevant_ids)
    ideal = sum(1 / math.log2(rank + 1) for rank in range(1, min(len(relevant_ids), k) + 1))
    return dcg / ideal if ideal else 0.


def paired_bootstrap_ci(first: list[float], second: list[float],
                        iterations: int = 2000, seed: int = 42) -> tuple[float, float]:
    """95% percentile CI for mean(second - first), resampling whole queries."""
    if len(first) != len(second) or not first:
        raise ValueError("Paired nonempty metric vectors required")
    delta = [b - a for a, b in zip(first, second)]
    rng = random.Random(seed)
    boots = sorted(mean(rng.choices(delta, k=len(delta))) for _ in range(iterations))
    return boots[int(.025 * (iterations-1))], boots[int(.975 * (iterations-1))]


def evaluate_queries(queries: list[dict], search_fn, k: int = 5) -> dict:
    """Rows must be {'query': str, 'relevant_page_ids': [str,...]}.

    Search callable returns ranked page dicts containing 'page_id'. No data
    from a question's relevance label may enter the retrieval process.
    """
    if not queries:
        raise ValueError("Provide benchmark rows")
    metrics = {"mrr": [], f"recall@{k}": [], f"ndcg@{k}": []}
    per_query = []
    for q in queries:
        relevant = set(q["relevant_page_ids"])
        if not relevant:
            raise ValueError(f"Missing gold relevance for {q['query']}")
        results = search_fn(q["query"], k)
        ranked = [r["page_id"] for r in results]
        row = {"query": q["query"], "gold": sorted(relevant), "retrieved": ranked,
               "mrr": reciprocal_rank(ranked, relevant),
               f"recall@{k}": recall_at_k(ranked, relevant, k),
               f"ndcg@{k}": ndcg_at_k(ranked, relevant, k)}
        for key in metrics:
            metrics[key].append(row[key])
        per_query.append(row)
    return {"n_queries": len(queries), "metrics": {key: mean(v) for key, v in metrics.items()},
            "per_query": per_query,
            "caution": "Only retrieval evaluated; no evidence of answer correctness"}


def run_evaluation(queries_json: str, corpus_dir: str, mode: str,
                   k: int = 5, output: str = "evaluation.json"):
    from .search import lexical_search
    queries = json.loads(Path(queries_json).read_text(encoding="utf-8"))
    if mode == "bm25":
        fn = lambda query, topk: lexical_search(query, corpus_dir, k=topk)
    elif mode == "visual":
        from .vision import ColSmolRetriever, retrieve_visual
        backend = ColSmolRetriever()
        fn = lambda query, topk: retrieve_visual(query, corpus_dir, backend, topk)
    else:
        raise ValueError(f"Unknown mode: {mode}")
    result = evaluate_queries(queries, fn, k=k)
    result.update({"mode": mode, "k": k, "corpus_dir": corpus_dir})
    Path(output).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result
