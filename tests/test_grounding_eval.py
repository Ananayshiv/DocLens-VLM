import pytest
from doclens.grounding_eval import evaluate_grounding


def sample_page():
    return {"page_id": "p1", "image_size": [200, 100],
            "words": [{"text": "25", "bbox": [30, 20, 55, 40], "line_id": "a", "origin": "pdf_text_layer"},
                      {"text": "October", "bbox": [58, 20, 120, 40], "line_id": "a", "origin": "pdf_text_layer"}]}


def test_correct_supported_and_absent_abstention():
    cases = [{"page_id": "p1", "model_proposal": "25 October", "expected_answer": "25 October",
              "gold_bbox_pixels": [30, 20, 120, 40]},
             {"page_id": "p1", "model_proposal": "Not in source", "expected_answer": None}]
    r = evaluate_grounding(cases, [sample_page()])
    m = r["metrics"]
    assert m["supported_rate"] == .5
    assert m["correct_supported_rate_all"] == .5
    assert m["false_support_on_absent"] == 0
    assert m["mean_bbox_iou_correct_supported"] == 1


def test_matching_some_other_line_is_not_answer_correctness():
    cases = [{"page_id": "p1", "model_proposal": "25", "expected_answer": "22"}]
    m = evaluate_grounding(cases, [sample_page()])["metrics"]
    assert m["supported_rate"] == 1
    assert m["supported_precision"] == 0


def test_abstention_metric_and_label_validation():
    cases = [{"page_id": "p1", "model_proposal": "NOVEMBER", "expected_answer": "25 October"}]
    m = evaluate_grounding(cases, [sample_page()])["metrics"]
    assert m["false_abstention_on_present"] == 1
    with pytest.raises(ValueError): evaluate_grounding([{"page_id": "p1", "model_proposal": "25"}], [sample_page()])
    with pytest.raises(ValueError): evaluate_grounding([], [sample_page()])
