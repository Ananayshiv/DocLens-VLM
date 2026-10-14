"""Deterministic evidence-gating regression tests. No pretrained model needed."""
from __future__ import annotations

import json
from pathlib import Path
import fitz
from PIL import Image

from doclens.corpus import ingest
from doclens.evidence import ABSTAIN, ground_proposal, locate_exact_span, tokens
from doclens.generator import answer_on_page


def word(text, x0, x1, line="1", y0=12, y1=30):
    return {"text": text, "bbox": [x0, y0, x1, y1], "line_id": line,
            "origin": "pdf_text_layer"}


def test_mixed_script_and_devanagari_combining_marks():
    assert tokens("वित्त विभाग INV-२०२६ | ₹ 2,500") == ["वित्त", "विभाग", "inv", "२०२६", "2", "500"]
    page = {"words": [word("आवेदन", 10, 60), word("तिथि:", 65, 95),
                       word("१५-१०-२०२६", 100, 190)]}
    result = ground_proposal("१५-१०-२०२६", page, (200, 100))
    assert result["answer"] == "१५-१०-२०२६"
    assert result["verification_status"] == "text_span_supported"
    assert result["evidence"]["bbox_pixels"] == [100, 12, 190, 30]
    assert result["evidence"]["bbox_normalized"][0] == .5


def test_no_guessed_bbox_on_missing_text():
    page = {"words": [word("Rs", 10, 25), word("1,000", 27, 70)]}
    for response in ["Rs 5,000", "NOT_FOUND", "", "The total is Rs 1,000"]:
        result = ground_proposal(response, page, (200, 100))
        assert result["answer"] == ABSTAIN
        assert result["evidence"] is None
        assert result["abstained"]


def test_duplicate_text_requires_abstention():
    page = {"words": [word("October", 10, 60), word("25", 65, 80),
                      word("October", 10, 60, line="2", y0=45, y1=60),
                      word("25", 65, 80, line="2", y0=45, y1=60)]}
    assert locate_exact_span("October 25", page["words"], (200, 100)) is None


def test_never_join_different_lines_or_invalid_bbox():
    words = [word("17", 20, 50, "1"), word("June", 10, 50, "2", y0=50, y1=65)]
    assert locate_exact_span("17 June", words, (200, 100)) is None
    words[1]["line_id"] = "1"
    words[1]["bbox"] = [-3, 50, 50, 65]
    assert locate_exact_span("June", words, (200, 100)) is None


def test_ingest_pdf_has_real_pixel_box_and_grounding(tmp_path):
    path = tmp_path / "doc.pdf"
    doc = fitz.open()
    page = doc.new_page(width=310, height=180)
    page.insert_text((40, 60), "Due date: October 25")
    doc.save(path)
    doc.close()
    rows = ingest([path], tmp_path / "corpus")
    r = rows[0]
    assert r["words"] and r["words"][0]["origin"] == "pdf_text_layer"
    assert ground_proposal("October 25", r, tuple(r["image_size"]))["abstained"] is False
    assert ground_proposal("December 20", r, tuple(r["image_size"]))["abstained"] is True
    assert json.loads((tmp_path / "corpus" / "corpus.json").read_text())[0]["words"] == r["words"]


def test_rotated_pdf_box_inside_page(tmp_path):
    path = tmp_path / "rot.pdf"
    doc = fitz.open()
    page = doc.new_page(width=300, height=200)
    page.insert_text((50, 50), "GOVERNMENT NOTICE")
    page.set_rotation(90)
    doc.save(path)
    doc.close()
    row = ingest([path], tmp_path / "corpus")[0]
    assert row["words"]
    assert ground_proposal("NOTICE", row, tuple(row["image_size"]))["evidence"] is not None


def test_proposed_answer_is_never_exposed_without_located_span(tmp_path):
    im = Image.new("RGB", (200, 80), "white")
    path = tmp_path / "page.png"
    im.save(path)
    corpus = tmp_path / "corpus"
    row = ingest([path], corpus)[0]

    class MockVLM:
        def answer(self, question, image):
            return "Amount: ₹99,999"

    result = answer_on_page("How much?", row, corpus, MockVLM())
    assert result["answer"] == ABSTAIN
    assert result["model_proposal"] == "Amount: ₹99,999"
    assert result["page"] == 1
    assert result["verification_status"] == "abstained_unverified"
