"""OCR every page of an imported corpus so BM25 sees the same page pool as visual retrieval.

python scripts/ocr_corpus.py --corpus runs/vidore --languages eng
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

from PIL import Image

from doclens.corpus import _atomic_json_write, extract_ocr_words


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--corpus", default="runs/vidore")
    p.add_argument("--languages", default="eng")
    a = p.parse_args()
    folder = Path(a.corpus)
    records = json.loads((folder / "corpus.json").read_text(encoding="utf-8"))
    empty = 0
    for i, r in enumerate(records, 1):
        with Image.open(folder / r["image"]) as im:
            r["text"], r["words"] = extract_ocr_words(im, a.languages)
            r["image_size"] = list(im.size)
        empty += not r["text"].strip()
        if i % 10 == 0 or i == len(records):
            print(f"OCR {i}/{len(records)}", flush=True)
    _atomic_json_write(folder / "corpus.json", records)
    print(json.dumps({"pages": len(records), "pages_without_ocr_text": empty, "languages": a.languages}))


if __name__ == "__main__":
    main()
