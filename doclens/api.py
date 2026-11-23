"""HTTP service for DocLens retrieval and evidence-grounded answers.

    pip install -e '.[api]'            # add '.[vision]' for visual/hybrid modes
    DOCLENS_CORPUS=runs/demo uvicorn doclens.api:app --port 8000

Endpoints
    GET  /health  -> corpus size, loaded models
    POST /search  {"question": str, "mode": "bm25"|"visual"|"hybrid", "k": int}
    POST /ask     {"question": str, "mode": ...}  -> grounded answer or abstention

Models load lazily on first use and are then reused across requests. The
service never returns an ungrounded VLM answer: /ask inherits the same
unique-span abstention contract as the CLI (`Not visible in the document.`).
"""
from __future__ import annotations
import os
import threading
from pathlib import Path

try:
    from fastapi import FastAPI, HTTPException
    from pydantic import BaseModel, Field
except ImportError as exc:  # pragma: no cover - optional dependency
    raise RuntimeError("Install the API extra: pip install -e '.[api]'") from exc

from .corpus import load_corpus

CORPUS_DIR = Path(os.environ.get("DOCLENS_CORPUS", "runs/corpus"))
MAX_K = 20

app = FastAPI(title="DocLens-VLM", version="0.2.0",
              description="Visual document retrieval with evidence-grounded, abstaining answers")
_lock = threading.Lock()
_models: dict[str, object] = {}


def _get(name: str):
    """Load each pretrained model at most once (thread-safe)."""
    with _lock:
        if name not in _models:
            if name == "retriever":
                from .vision import ColSmolRetriever
                _models[name] = ColSmolRetriever()
            elif name == "answerer":
                from .generator import SmolVLMAnswerer
                _models[name] = SmolVLMAnswerer()
        return _models[name]


class SearchRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    mode: str = Field(default="bm25", pattern="^(bm25|visual|hybrid)$")
    k: int = Field(default=3, ge=1, le=MAX_K)


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    # Visual-only is the default: it beat both BM25 and hybrid fusion on ViDoRe (see README).
    mode: str = Field(default="visual", pattern="^(bm25|visual|hybrid)$")


def _search(question: str, mode: str, k: int) -> list[dict]:
    if not load_corpus(CORPUS_DIR):
        raise HTTPException(404, f"No pages indexed in {CORPUS_DIR}; run `doclens ingest` first")
    try:
        if mode == "bm25":
            from .search import lexical_search
            return lexical_search(question, CORPUS_DIR, k=k)
        if mode == "visual":
            from .vision import retrieve_visual
            return retrieve_visual(question, CORPUS_DIR, _get("retriever"), k)
        from .hybrid import hybrid_search
        return hybrid_search(question, CORPUS_DIR, _get("retriever"), k=k)
    except (ValueError, FileNotFoundError) as exc:
        raise HTTPException(422, str(exc)) from exc


def _public(row: dict) -> dict:
    keep = ("page_id", "source", "page", "score", "mode", "component_ranks",
            "warning", "coverage_warning", "fusion_warning")
    return {key: row[key] for key in keep if key in row}


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "corpus": str(CORPUS_DIR),
            "pages": len(load_corpus(CORPUS_DIR)), "models_loaded": sorted(_models)}


@app.post("/search")
def search(req: SearchRequest) -> dict:
    return {"question": req.question, "mode": req.mode,
            "results": [_public(r) for r in _search(req.question, req.mode, req.k)]}


@app.post("/ask")
def ask(req: AskRequest) -> dict:
    from .generator import answer_on_page
    pages = _search(req.question, req.mode, 1)
    if not pages:
        raise HTTPException(404, "No pages retrieved")
    result = answer_on_page(req.question, pages[0], CORPUS_DIR, _get("answerer"))
    return {"question": req.question, "retrieval": _public(pages[0]), **result}
