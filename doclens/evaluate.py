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


MODES = ("bm25", "visual", "hybrid")


def make_search_fn(mode: str, corpus_dir: str, backend=None):
    """Return search(query, k) for a mode. `backend` lets modes share one model."""
    from .search import lexical_search
    if mode == "bm25":
        return lambda query, topk: lexical_search(query, corpus_dir, k=topk)
    if mode not in MODES:
        raise ValueError(f"Unknown mode: {mode}")
    from .vision import ColSmolRetriever, retrieve_visual
    backend = backend or ColSmolRetriever()
    if mode == "visual":
        return lambda query, topk: retrieve_visual(query, corpus_dir, backend, topk)
    from .hybrid import hybrid_search
    return lambda query, topk: hybrid_search(query, corpus_dir, backend, k=topk)


def summarize_comparison(results_by_mode: dict[str, dict], baseline: str = "bm25",
                         iterations: int = 2000, seed: int = 42) -> dict:
    """Paired comparison of every mode against `baseline` on identical queries.

    Each value is an evaluate_queries() output. Deltas are mean(mode - baseline)
    with a 95% paired bootstrap CI resampling whole queries.
    """
    if baseline not in results_by_mode:
        raise ValueError(f"Baseline {baseline!r} missing")
    base = results_by_mode[baseline]
    base_queries = [r["query"] for r in base["per_query"]]
    metric_keys = list(base["metrics"])
    table = {}
    for mode, result in results_by_mode.items():
        if [r["query"] for r in result["per_query"]] != base_queries:
            raise ValueError(f"Mode {mode!r} was not evaluated on the baseline's query list")
        row = {"metrics": dict(result["metrics"]), "delta_vs_baseline": {}}
        if mode != baseline:
            for key in metric_keys:
                a = [r[key] for r in base["per_query"]]
                b = [r[key] for r in result["per_query"]]
                lo, hi = paired_bootstrap_ci(a, b, iterations=iterations, seed=seed)
                row["delta_vs_baseline"][key] = {
                    "mean": mean(b) - mean(a), "ci95": [lo, hi],
                    "significant": lo > 0 or hi < 0}
        table[mode] = row
    return {"baseline": baseline, "n_queries": len(base_queries), "modes": table}


def comparison_markdown(summary: dict) -> str:
    modes = summary["modes"]
    keys = list(next(iter(modes.values()))["metrics"])
    header = "| Mode | " + " | ".join(keys) + f" | Δ{keys[-1]} vs {summary['baseline']} (95% CI) |"
    lines = [header, "|" + "---|" * (len(keys) + 2)]
    for mode, row in modes.items():
        cells = [f"{row['metrics'][k]:.4f}" for k in keys]
        d = row["delta_vs_baseline"].get(keys[-1])
        delta = "—" if d is None else (f"{d['mean']:+.4f} [{d['ci95'][0]:+.4f}, {d['ci95'][1]:+.4f}]"
                                       + (" *" if d["significant"] else ""))
        lines.append(f"| {mode} | " + " | ".join(cells) + f" | {delta} |")
    lines.append(f"\nn = {summary['n_queries']} queries; * = 95% CI excludes zero.")
    return "\n".join(lines)


def run_benchmark(queries_json: str, corpus_dir: str, modes=MODES, k: int = 5,
                  out_dir: str = "runs/benchmark", iterations: int = 2000, seed: int = 42) -> dict:
    """Evaluate all modes on the same corpus/queries with ONE shared model load."""
    queries = json.loads(Path(queries_json).read_text(encoding="utf-8"))
    backend = None
    if any(m != "bm25" for m in modes):
        from .vision import ColSmolRetriever
        backend = ColSmolRetriever()
    results = {}
    for mode in modes:
        results[mode] = evaluate_queries(queries, make_search_fn(mode, corpus_dir, backend), k=k)
    summary = summarize_comparison(results, baseline=modes[0], iterations=iterations, seed=seed)
    summary.update({"k": k, "corpus_dir": corpus_dir, "queries": queries_json, "seed": seed,
                    "model_id": getattr(backend, "model_id", None)})
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (out / "per_query.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    (out / "results.md").write_text(comparison_markdown(summary) + "\n", encoding="utf-8")
    return summary


def run_evaluation(queries_json: str, corpus_dir: str, mode: str,
                   k: int = 5, output: str = "evaluation.json"):
    queries = json.loads(Path(queries_json).read_text(encoding="utf-8"))
    fn = make_search_fn(mode, corpus_dir)
    result = evaluate_queries(queries, fn, k=k)
    result.update({"mode": mode, "k": k, "corpus_dir": corpus_dir})
    Path(output).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result
