"""Page search; explicit zero-score handling for the lexical baseline."""
from pathlib import Path
import warnings
from .corpus import load_corpus, ingestion_coverage_notice
from .lexical import BM25


def lexical_search(question: str, corpus_dir: str | Path, k: int = 3) -> list[dict]:
    coverage_notice = ingestion_coverage_notice(corpus_dir)
    if coverage_notice:
        warnings.warn(coverage_notice, UserWarning, stacklevel=2)
    pages = load_corpus(corpus_dir)
    if not pages:
        return []
    bm25 = BM25([p["text"] for p in pages])
    scores = bm25.scores(question)
    ranked = sorted(zip(pages, scores), key=lambda a: a[1], reverse=True)
    # Return zero-match pages for transparency; NOT evidence of relevance.
    return [{**p, "score": float(s), "mode": "text-bm25",
             "warning": "no lexical match" if s == 0 else "",
             "coverage_warning": coverage_notice or ""} for p, s in ranked[:k]]
