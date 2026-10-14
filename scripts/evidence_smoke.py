"""Reproducible CPU OCR/PDF box smoke test; NOT VLM or benchmark evidence."""
from __future__ import annotations
import json
from pathlib import Path
from PIL import Image, ImageDraw
from doclens.corpus import ingest
from doclens.evidence import ground_proposal


def main():
    root = Path(__file__).resolve().parents[1]
    out = root / "runs" / "evidence_smoke"
    rows = ingest([root / "assets" / "notice_deadline.png"], out,
                  ocr=True, ocr_languages="eng+hin")
    row = rows[0]
    inputs = ["25 अक्टूबर 2026", "25 October 2026", "No such date 2047"]
    results = []
    for proposal in inputs:
        record = ground_proposal(proposal, row, tuple(row["image_size"]))
        results.append({"proposal": proposal, **record,
                        "source": row["source"], "page": row["page"], "page_id": row["page_id"]})
    out.mkdir(parents=True, exist_ok=True)
    (out / "smoke_results.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    found = results[0]["evidence"]
    if found:
        with Image.open(out / row["image"]) as im:
            im = im.convert("RGB")
            ImageDraw.Draw(im).rectangle(found["bbox_pixels"], outline="red", width=4)
            im.save(root / "examples" / "fictional_notice_evidence_box.png")
    print("Synthetic demo only (no VLM):", [(r["proposal"], r["abstained"]) for r in results])
    print("Saved to:", out)


if __name__ == "__main__":
    main()
