import json
import pytest
from PIL import Image
from doclens.corpus import ingest, load_corpus
from doclens.search import lexical_search


def test_ingest_png_and_duplicate(tmp_path):
    image = Image.new("RGB", (30, 30), color="white")
    path = tmp_path / "page.png"
    image.save(path)
    out = tmp_path / "output"
    rows = ingest([path, path], out)
    assert len(rows) == 1
    assert (out / rows[0]["image"]).exists()
    assert load_corpus(out) == rows
    with pytest.raises(ValueError, match="Empty BM25 text index"):
        lexical_search("anything", out)


def test_ingest_pdf_with_selectable_text(tmp_path):
    import fitz
    p = tmp_path / "sample.pdf"
    pdf = fitz.open()
    page = pdf.new_page()
    page.insert_text((50, 70), "Fictional deadline October 25")
    pdf.save(p)
    pdf.close()
    rows = ingest([p], tmp_path / "index")
    assert len(rows) == 1
    assert "October 25" in rows[0]["text"]
    assert lexical_search("deadline", tmp_path / "index")[0]["score"] > 0.


def test_bad_suffix(tmp_path):
    wrong = tmp_path / "bad.txt"
    wrong.write_text("invalid")
    with pytest.raises(ValueError): ingest([wrong], tmp_path / "index")


def test_empty_corpus(tmp_path):
    base = tmp_path / "idx"
    base.mkdir()
    (base / "corpus.json").write_text("[]")
    assert lexical_search("anything", base) == []
