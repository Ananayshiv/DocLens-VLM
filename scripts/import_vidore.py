"""Download a SMALL reproducible ViDoRe DocVQA test subset as a real image retrieval benchmark.

No 'official benchmark' claim: this builds a smaller candidate pool and reports
its true pool size. Labels are held separate from model input.

python scripts/import_vidore.py --n 80 --out runs/vidore --seed 42
"""
from __future__ import annotations
import argparse
import hashlib
import io
import json
from pathlib import Path


def prepare(n: int, out: str, seed: int = 42) -> dict:
    if n < 2 or n > 500:
        raise ValueError("Choose between 2 and 500 test questions")
    # Keep benchmark candidate pools immutable: mixing different seeds / n
    # into an existing corpus invalidates retrieval comparisons.
    folder = Path(out)
    corpus_path = folder / "corpus.json"
    if corpus_path.exists():
        metadata_path = folder / "metadata.json"
        queries_path = folder / "queries.json"
        if metadata_path.exists() and queries_path.exists():
            existing = json.loads(metadata_path.read_text(encoding="utf-8"))
            if (existing.get("dataset") == "vidore/docvqa_test_subsampled"
                    and existing.get("split") == "test"
                    and existing.get("seed") == seed
                    and existing.get("requested_n", existing.get("n_questions")) == n):
                return existing  # Idempotent rerun; do not overwrite existing pages.
        raise FileExistsError(
            f"{corpus_path} already exists: refusing to overwrite an existing "
            "corpus or mix evaluation pools. Use a fresh --out directory."
        )
    from datasets import load_dataset
    from PIL import ImageOps
    ds = load_dataset("vidore/docvqa_test_subsampled", split="test")
    # deterministic sampling, not selecting examples based on performance
    shuffled = ds.shuffle(seed=seed).select(range(min(n, len(ds))))
    pages = folder / "pages"
    pages.mkdir(parents=True, exist_ok=True)
    records, queries = {}, []
    for row in shuffled:
        image = ImageOps.exif_transpose(row["image"]).convert("RGB")
        buf = io.BytesIO()
        image.save(buf, format="PNG")
        blob = buf.getvalue()
        identifier = hashlib.sha256(blob).hexdigest()[:20]
        if identifier not in records:
            (pages / f"{identifier}.png").write_bytes(blob)
            records[identifier] = {"page_id": identifier, "source": "ViDoRe DocVQA test",
                                   "page": str(row.get("docId", identifier)),
                                   "image": f"pages/{identifier}.png", "text": ""}
        queries.append({"query": str(row["query"]), "relevant_page_ids": [identifier],
                        "benchmark_question_id": str(row.get("questionId", ""))})
    from doclens.corpus import _atomic_json_write
    _atomic_json_write(folder / "corpus.json", list(records.values()))
    _atomic_json_write(folder / "queries.json", queries)
    meta = {"dataset": "vidore/docvqa_test_subsampled", "split": "test",
            "seed": seed, "requested_n": n,
            "n_questions": len(queries), "n_candidate_pages": len(records),
            "evaluation": "retrieval only", "warning": "subsampled candidate pool; NOT official ViDoRe leaderboard"}
    _atomic_json_write(folder / "metadata.json", meta)
    return meta


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=80)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default="runs/vidore")
    a = p.parse_args()
    print(json.dumps(prepare(a.n, a.out, a.seed), indent=2))
