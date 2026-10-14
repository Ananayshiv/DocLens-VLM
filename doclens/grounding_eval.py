"""Evaluation of literal grounded-answer proposals against curated gold labels.

Inputs are human-annotated; no dataset measurements are bundled or implied.
Each JSON row requires: page_id, model_proposal, expected_answer (str or null).
Optional: gold_page_id, gold_bbox_pixels for actual visual localization checks.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from .corpus import load_corpus
from .evidence import ground_proposal
from .fields import comparison_tokens


def _iou(a: list[float], b: list[float]) -> float:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    intersection = max(0., min(ax1, bx1) - max(ax0, bx0)) * max(0., min(ay1, by1) - max(ay0, by0))
    aa = max(0., ax1-ax0) * max(0., ay1-ay0)
    ba = max(0., bx1-bx0) * max(0., by1-by0)
    return intersection / (aa + ba - intersection) if aa + ba > intersection else 0.


def evaluate_grounding(rows: list[dict], corpus: list[dict]) -> dict:
    """Evaluate abstention and exact answer correctness; no implicit gold labels."""
    if not rows:
        raise ValueError("No annotated questions supplied")
    by_id = {record["page_id"]: record for record in corpus}
    out = []
    ious = []
    for i, row in enumerate(rows):
        if "expected_answer" not in row or "model_proposal" not in row or "page_id" not in row:
            raise ValueError(f"Row {i}: page_id, model_proposal and expected_answer required")
        if row["page_id"] not in by_id:
            raise ValueError(f"Row {i}: unknown page_id {row['page_id']}")
        page = by_id[row["page_id"]]
        if "image_size" not in page:
            raise ValueError("Re-index corpus with word geometry before grounding evaluation")
        answer = ground_proposal(row["model_proposal"], page, tuple(page["image_size"]))
        gold = row["expected_answer"]
        if gold is not None and not isinstance(gold, str):
            raise TypeError("expected_answer must be a string or null")
        if gold is not None and not comparison_tokens(gold):
            raise ValueError("Present-answer gold label cannot be empty")
        present = gold is not None
        supported = not answer["abstained"]
        matching = (supported and present and comparison_tokens(answer["answer"]) == comparison_tokens(gold)
                    and ("gold_page_id" not in row or row["gold_page_id"] == row["page_id"]))
        record = {"question": row.get("question", ""), "page_id": row["page_id"],
                  "gold_present": present, "supported": supported, "correct_supported": bool(matching),
                  "abstained": not supported, "reason": answer["reason"]}
        if row.get("gold_bbox_pixels") is not None and matching:
            location_iou = _iou(answer["evidence"]["bbox_pixels"], row["gold_bbox_pixels"])
            ious.append(location_iou)
            record["bbox_iou"] = location_iou
        out.append(record)
    n = len(rows)
    absent = [r for r in out if not r["gold_present"]]
    present = [r for r in out if r["gold_present"]]
    supported = [r for r in out if r["supported"]]
    return {"n_questions": n, "metrics": {
        "supported_rate": len(supported)/n,
        "abstention_rate": sum(r["abstained"] for r in out)/n,
        "correct_supported_rate_all": sum(r["correct_supported"] for r in out)/n,
        "supported_precision": sum(r["correct_supported"] for r in supported)/len(supported) if supported else None,
        "false_support_on_absent": sum(r["supported"] for r in absent)/len(absent) if absent else None,
        "false_abstention_on_present": sum(r["abstained"] for r in present)/len(present) if present else None,
        "mean_bbox_iou_correct_supported": sum(ious)/len(ious) if ious else None,
    }, "per_query": out,
        "warning": "Comparisons deliberately strip currencies and monetary signs; this is not financially safe numeric equality. Only manually labeled proposals evaluated; text match is not fact verification; evaluate real VLM generations separately."}


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate externally labeled answer and bbox correctness")
    parser.add_argument("--corpus", required=True)
    parser.add_argument("--labels", required=True,
                        help="JSON list of {page_id, model_proposal, expected_answer, gold_page_id?, gold_bbox_pixels?}")
    parser.add_argument("--out", default="grounding_eval.json")
    args = parser.parse_args()
    questions = json.loads(Path(args.labels).read_text(encoding="utf-8"))
    result = evaluate_grounding(questions, load_corpus(args.corpus))
    Path(args.out).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result["metrics"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
