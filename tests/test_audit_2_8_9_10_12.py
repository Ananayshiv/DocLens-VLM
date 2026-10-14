"""Regression tests for 2026-10-14 audit items 2, 8, 9, 10 and 12."""
import json
import subprocess
import sys

import pytest
from PIL import Image

from doclens.corpus import (ingest, load_corpus, load_ingestion_manifest,
                              ingestion_coverage_notice, render_document)
from doclens.fields import comparison_tokens, value_appears_in_text
from doclens.grounding_eval import evaluate_grounding
from doclens.lexical import BM25
from doclens.search import lexical_search


@pytest.mark.parametrize("alternative", ["-500", "$500", "₹500", "($500)",
                                         "+500", "500", "५００", "५००"])
def test_audit2_symmetric_sign_and_currency_normalization(alternative):
    assert comparison_tokens(alternative) == ["500"]
    assert value_appears_in_text(alternative, "कुल देय: 500")
    assert value_appears_in_text("500", f"Total payable: {alternative}")
    assert not value_appears_in_text("500", "Total payable: 5000")


def test_audit2_punctuation_currency_comparison_and_word_boundary():
    assert comparison_tokens("(₹1,250.00)") == ["1250"]
    assert value_appears_in_text("₹1,250.00", "Amount: 1250")
    assert value_appears_in_text("INR 860", "Total payable: INR    860")
    assert not value_appears_in_text("INV-20", "Invoice: INV-204")
    assert comparison_tokens("अंतिम तारीख") == comparison_tokens("अंतिम तारीख")


def test_audit2_grounding_gold_and_proposal_use_identical_rules():
    page = {"page_id": "amount", "image_size": [200, 100], "words": [
        {"text": "$500", "bbox": [5, 10, 70, 30], "line_id": "line", "origin": "pdf_text_layer"}]}
    rows = [
        {"page_id": "amount", "model_proposal": "500", "expected_answer": "-500"},
        {"page_id": "amount", "model_proposal": "₹500", "expected_answer": "$500"},
        {"page_id": "amount", "model_proposal": "500", "expected_answer": "600"},
    ]
    result = evaluate_grounding(rows, [page])
    assert result["metrics"]["supported_rate"] == 1.0
    assert result["metrics"]["supported_precision"] == pytest.approx(2 / 3)
    assert [r["correct_supported"] for r in result["per_query"]] == [True, True, False]
    assert "signs" in result["warning"]


def test_audit9_non_destructive_incremental_merge(tmp_path):
    a = tmp_path / "one.png"
    b = tmp_path / "two.png"
    Image.new("RGB", (24, 18), "red").save(a)
    Image.new("RGB", (24, 18), "green").save(b)
    out = tmp_path / "corpus"
    first = ingest([a], out)
    assert len(first) == 1
    with pytest.raises(ValueError, match="Empty BM25 text index"):
        lexical_search("amount", out)
    combined = ingest([b], out)
    assert len(combined) == 2
    assert {r["page_id"] for r in combined} >= {first[0]["page_id"]}
    again = ingest([a], out)
    assert len(again) == 2
    assert load_corpus(out) == again
    assert all((out / r["image"]).exists() for r in again)


def test_audit9_reindex_never_erases_old_ocr(monkeypatch, tmp_path):
    path = tmp_path / "doc.png"
    Image.new("RGB", (30, 20), "yellow").save(path)
    out = tmp_path / "corpus"
    import doclens.corpus as corpus
    monkeypatch.setattr(corpus, "extract_ocr_words", lambda im, languages:
                        ("देय राशि 500", [{"text": "500", "bbox": [2, 3, 20, 12],
                                           "line_id": "1", "origin": "test"}]))
    enriched = ingest([path], out, ocr=True)
    assert enriched[0]["text"] == "देय राशि 500"
    not_enriched = ingest([path], out, ocr=False)
    assert not_enriched[0]["text"] == "देय राशि 500"
    assert len(not_enriched[0]["words"]) == 1


def test_audit9_reject_corrupt_existing_duplicates_without_replacing(tmp_path):
    path = tmp_path / "img.png"
    Image.new("RGB", (20, 20), "black").save(path)
    out = tmp_path / "corpus"
    ingest([path], out)
    rows = load_corpus(out)
    corrupt = json.dumps(rows * 2)
    (out / "corpus.json").write_text(corrupt)
    with pytest.raises(ValueError, match="Duplicate page_id"):
        ingest([path], out)
    assert (out / "corpus.json").read_text() == corrupt


def test_audit10_exif_orientation_correct_before_ingest_and_ocr(monkeypatch, tmp_path):
    src = tmp_path / "phone.jpg"
    image = Image.new("RGB", (60, 24), "white")
    exif = image.getexif()
    exif[274] = 6  # Rotate displayed image 90 degrees clockwise
    image.save(src, exif=exif)
    from doclens import corpus
    seen = []
    def spy_ocr(im, languages):
        seen.append(im.size)
        return "Hindi English", []
    monkeypatch.setattr(corpus, "extract_ocr_words", spy_ocr)
    rows = ingest([src], tmp_path / "corpus", ocr=True)
    assert seen == [(24, 60)]
    assert rows[0]["image_size"] == [24, 60]
    with Image.open(tmp_path / "corpus" / rows[0]["image"]) as stored:
        assert stored.size == (24, 60)
        assert stored.getexif().get(274) in (None, 1)


def test_audit8_pdf_truncation_warns_and_notice_persists(tmp_path):
    import fitz
    source = tmp_path / "long.pdf"
    with fitz.open() as pdf:
        for i in range(31):
            page = pdf.new_page(width=120, height=90)
            page.insert_text((10, 40), f"PageNumber {i+1}", fontsize=8)
        pdf.save(source)
    out = tmp_path / "corpus"
    with pytest.warns(UserWarning, match="exceeds the 30-page ingestion limit"):
        capped = ingest([source], out)
    assert len(capped) == 30
    manifest = load_ingestion_manifest(out)
    entry = manifest["sources"][str(source.resolve())]
    assert entry["total_pages"] == 31
    assert entry["truncated_pages"] == 1
    assert "Not visible" in ingestion_coverage_notice(out)
    with pytest.warns(UserWarning, match="INCOMPLETE CORPUS"):
        hits = lexical_search("PageNumber", out)
    assert hits and "INCOMPLETE CORPUS" in hits[0]["coverage_warning"]
    # Re-ingest with explicit unlimited option: it must preserve existing
    # records and clear the persistent warning.
    all_pages = ingest([source], out, max_pages_per_file=0)
    assert len(all_pages) == 31
    assert ingestion_coverage_notice(out) is None
    assert load_ingestion_manifest(out)["sources"][str(source.resolve())]["truncated_pages"] == 0
    assert any(r["page"] == 31 for r in all_pages)


def test_audit8_invalid_limits_and_cli_unlimited_argument(tmp_path):
    image = tmp_path / "test.png"
    Image.new("RGB", (10, 10)).save(image)
    with pytest.raises(ValueError, match="max_pages"):
        list(render_document(image, max_pages=-1))
    with pytest.raises(ValueError, match="max_pages_per_file"):
        ingest([image], tmp_path / "corpus", max_pages_per_file=-1)
    proc = subprocess.run([sys.executable, "-m", "doclens.cli", "ingest",
                           str(image), "--out", str(tmp_path / "cli"), "--max-pages", "0"],
                          capture_output=True, text=True, check=True)
    assert "Corpus now contains 1 unique pages" in proc.stdout


def test_audit12_bm25_rejects_wholly_unsearchable_corpus():
    with pytest.raises(ValueError, match="Empty BM25 text index"):
        BM25([])
    with pytest.raises(ValueError, match="Empty BM25 text index"):
        BM25(["", "!$₹", " \n "])
    model = BM25(["", "अंतिम 25 October"])
    assert model.scores("अंतिम")[1] > model.scores("अंतिम")[0]
    with pytest.raises(ValueError, match="Empty BM25 query"):
        model.scores("!$₹")


def test_audit9_vidore_import_refuses_to_clobber_another_corpus(tmp_path):
    from scripts.import_vidore import prepare
    dest = tmp_path / "existing"
    dest.mkdir()
    (dest / "corpus.json").write_text('[{"page_id":"my-page"}]', encoding="utf-8")
    before = (dest / "corpus.json").read_bytes()
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        prepare(8, str(dest))
    assert (dest / "corpus.json").read_bytes() == before


def test_audit9_vidore_import_idempotent_same_run(tmp_path):
    from scripts.import_vidore import prepare
    dest = tmp_path / "existing"
    dest.mkdir()
    (dest / "corpus.json").write_text('[{"page_id":"same-run"}]', encoding="utf-8")
    (dest / "queries.json").write_text('[]', encoding="utf-8")
    meta = {"dataset": "vidore/docvqa_test_subsampled", "split": "test", "seed": 42,
            "requested_n": 8, "n_questions": 8, "n_candidate_pages": 8}
    (dest / "metadata.json").write_text(json.dumps(meta), encoding="utf-8")
    assert prepare(8, str(dest), seed=42) == meta
    with pytest.raises(FileExistsError):
        prepare(8, str(dest), seed=43)
