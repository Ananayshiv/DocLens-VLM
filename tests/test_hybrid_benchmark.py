import pytest
from doclens.hybrid import reciprocal_rank_fusion
from doclens.evaluate import evaluate_queries, summarize_comparison, comparison_markdown


def rows(*ids):
    return [{"page_id": i} for i in ids]


def test_rrf_rewards_agreement():
    fused = reciprocal_rank_fusion([rows("a", "b", "c"), rows("b", "c", "a")], rrf_k=60)
    assert [r["page_id"] for r in fused][0] == "b"
    assert fused[0]["component_ranks"] == {"ranking_0": 2, "ranking_1": 1}
    assert fused[0]["mode"] == "hybrid-rrf"


def test_rrf_formula_and_missing_pages():
    fused = {r["page_id"]: r for r in
             reciprocal_rank_fusion([rows("a"), rows("b", "a")], rrf_k=0, names=["bm25", "visual"])}
    assert fused["a"]["score"] == pytest.approx(1 / 1 + 1 / 2)
    assert fused["b"]["score"] == pytest.approx(1 / 1)
    assert fused["b"]["component_ranks"] == {"visual": 1}


def test_rrf_duplicates_weights_and_determinism():
    fused = reciprocal_rank_fusion([rows("a", "a", "b")], rrf_k=0)
    assert fused[0]["score"] == pytest.approx(1.0)  # duplicate not double counted
    only_visual = reciprocal_rank_fusion([rows("a", "b"), rows("b", "a")], weights=[0, 1])
    assert only_visual[0]["page_id"] == "b"
    tie = reciprocal_rank_fusion([rows("z", "y"), rows("y", "z")])
    assert [r["page_id"] for r in tie] == ["y", "z"]
    with pytest.raises(ValueError):
        reciprocal_rank_fusion([])
    with pytest.raises(ValueError):
        reciprocal_rank_fusion([rows("a")], weights=[1, 1])


def test_rrf_preserves_devanagari_ids():
    fused = reciprocal_rank_fusion([rows("सूचना-1"), rows("सूचना-1")])
    assert fused[0]["page_id"] == "सूचना-1"


def _evaluate(ranking_by_query):
    queries = [{"query": q, "relevant_page_ids": ["gold"]} for q in ranking_by_query]
    return evaluate_queries(queries, lambda q, k: rows(*ranking_by_query[q])[:k], k=3)


def test_summarize_comparison_paired_deltas():
    base = _evaluate({"q1": ["x", "gold"], "q2": ["x", "y", "gold"], "q3": ["x", "gold"]})
    better = _evaluate({"q1": ["gold"], "q2": ["gold"], "q3": ["gold"]})
    summary = summarize_comparison({"bm25": base, "hybrid": better}, iterations=200, seed=1)
    delta = summary["modes"]["hybrid"]["delta_vs_baseline"]["mrr"]
    assert delta["mean"] == pytest.approx(1 - (0.5 + 1 / 3 + 0.5) / 3)
    assert delta["ci95"][0] > 0 and delta["significant"]
    assert summary["modes"]["bm25"]["delta_vs_baseline"] == {}
    table = comparison_markdown(summary)
    assert "| hybrid |" in table and "*" in table


def test_summarize_rejects_mismatched_query_sets():
    a = _evaluate({"q1": ["gold"]})
    b = _evaluate({"q2": ["gold"]})
    with pytest.raises(ValueError):
        summarize_comparison({"bm25": a, "visual": b})
    with pytest.raises(ValueError):
        summarize_comparison({"visual": a}, baseline="bm25")
