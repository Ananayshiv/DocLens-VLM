import pytest
from doclens.lexical import BM25, tokenize
from doclens.evaluate import reciprocal_rank, recall_at_k, ndcg_at_k, paired_bootstrap_ci, evaluate_queries


def test_indic_combining_marks():
    assert "अंतिम" in tokenize("अंतिम तारीख क्या है?")
    assert "तारीख" in tokenize("अंतिम तारीख क्या है?")
    assert "बैंक" in tokenize("बैंक में नाम")


def test_bm25():
    m = BM25(["payment deadline October", "July meeting circular", "आवेदन अंतिम तारीख"])
    assert m.scores("deadline")[0] > m.scores("deadline")[1]
    assert m.scores("अंतिम")[2] > m.scores("अंतिम")[0]


def test_metrics():
    ranked = ["bad", "good", "extra"]
    gold = {"good"}
    assert reciprocal_rank(ranked, gold) == pytest.approx(.5)
    assert recall_at_k(ranked, gold, 1) == 0.
    assert recall_at_k(ranked, gold, 2) == 1.
    assert ndcg_at_k(ranked, gold, 2) == pytest.approx(1 / __import__('math').log2(3))
    with pytest.raises(ValueError): recall_at_k([], set(), 2)


def test_bootstrap_reproducibility():
    a = paired_bootstrap_ci([0., 1., 0.], [1., 1., 1.], iterations=50, seed=4)
    b = paired_bootstrap_ci([0., 1., 0.], [1., 1., 1.], iterations=50, seed=4)
    assert a == b
    assert a[0] >= 0.


def test_evaluate_queries():
    result = evaluate_queries([{"query":"date", "relevant_page_ids":["a"]}],
                              lambda query, k: [{"page_id":"a"}], k=1)
    assert result["metrics"]["recall@1"] == 1.
    with pytest.raises(ValueError): evaluate_queries([], lambda *_: [])
