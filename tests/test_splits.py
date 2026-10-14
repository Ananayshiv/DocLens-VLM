import pytest
from scripts.train_adapter import validate_split


def test_prevent_question_leakage():
    with pytest.raises(ValueError):
        validate_split([{"query":"date", "relevant_page_ids":["p1"]}],
                       [{"query":"date", "relevant_page_ids":["p2"]}])


def test_prevent_positive_doc_overlap():
    with pytest.raises(ValueError):
        validate_split([{"query":"date", "relevant_page_ids":["p1"]}],
                       [{"query":"meeting", "relevant_page_ids":["p1"]}])


def test_disjoint():
    validate_split([{"query":"date", "relevant_page_ids":["p1"]}],
                   [{"query":"meeting", "relevant_page_ids":["p2"]}])
