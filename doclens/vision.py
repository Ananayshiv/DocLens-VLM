"""Real pretrained 500M-parameter visual VLM retriever (ColSmol / ColPali)."""
from __future__ import annotations
from pathlib import Path
import numpy as np
import torch
from PIL import Image, ImageOps
from .late_interaction import maxsim

DEFAULT_MODEL = "vidore/colSmol-500M"


class ColSmolRetriever:
    """GPU-backed pretrained page and query embedder; loads weights from HF on first use.

    ColSmol (500M) is deliberately chosen for a 16GB T4; it is NOT a
    from-scratch VLM training reproduction. The MaxSim algorithm is local.
    """
    def __init__(self, model_id: str = DEFAULT_MODEL, device: str | None = None):
        try:
            from colpali_engine.models import ColIdefics3, ColIdefics3Processor
        except ImportError as exc:
            raise RuntimeError("Install: pip install -e '.[vision]'") from exc
        self.model_id = model_id
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        dtype = torch.float16 if self.device.startswith("cuda") else torch.float32
        self.model = ColIdefics3.from_pretrained(model_id, torch_dtype=dtype, attn_implementation="eager").to(self.device).eval()
        self.processor = ColIdefics3Processor.from_pretrained(model_id)

    @torch.inference_mode()
    def encode_image(self, image: Image.Image) -> torch.Tensor:
        inputs = self.processor.process_images([ImageOps.exif_transpose(image).convert("RGB")]).to(self.device)
        vectors = self.model(**inputs)[0].float().cpu()
        return vectors

    @torch.inference_mode()
    def encode_query(self, query: str) -> torch.Tensor:
        inputs = self.processor.process_queries([query]).to(self.device)
        vectors = self.model(**inputs)[0].float().cpu()
        return vectors


def build_index(corpus_dir: str | Path, backend: ColSmolRetriever,
                overwrite: bool = False) -> dict:
    from .corpus import load_corpus
    base = Path(corpus_dir)
    output = base / "visual"
    output.mkdir(parents=True, exist_ok=True)
    records = load_corpus(base)
    completed = 0
    for record in records:
        dest = output / f"{record['page_id']}.npz"
        if dest.exists() and not overwrite:
            continue
        with Image.open(base / record["image"]) as image:
            vectors = backend.encode_image(ImageOps.exif_transpose(image).convert("RGB"))
        # No pickle loading; .npz supports safe numeric arrays only.
        np.savez_compressed(dest, embeddings=vectors.numpy().astype(np.float16))
        completed += 1
    metadata = {"model_id": backend.model_id,
                "page_count": len(records)}
    (output / "metadata.json").write_text(__import__("json").dumps(metadata, indent=2))
    return {"indexed": completed, "total": len(records)}


def retrieve_visual(query: str, corpus_dir: str | Path,
                    backend: ColSmolRetriever, k: int = 3) -> list[dict]:
    from .corpus import load_corpus, ingestion_coverage_notice
    import warnings
    base = Path(corpus_dir)
    coverage_notice = ingestion_coverage_notice(base)
    if coverage_notice:
        warnings.warn(coverage_notice, UserWarning, stacklevel=2)
    records = load_corpus(base)
    if not records:
        return []
    metadata_path = base / "visual" / "metadata.json"
    if metadata_path.exists():
        import json
        meta = json.loads(metadata_path.read_text())
        if meta.get("model_id") != backend.model_id:
            raise ValueError("Visual index was built with a different checkpoint; rebuild the index")
    q = backend.encode_query(query)
    # Stream one indexed page at a time (peak memory O(Q×chunk), not all pages).
    scores = []
    for row in records:
        item = base / "visual" / f"{row['page_id']}.npz"
        if not item.exists():
            raise FileNotFoundError(f"Missing visual index: {item} (run index --mode visual)")
        with np.load(item, allow_pickle=False) as package:
            document = torch.from_numpy(package["embeddings"].astype("float32"))
        scores.append(float(maxsim(q, document, chunk_size=512).item()))
    best = sorted(zip(records, scores), key=lambda x: x[1], reverse=True)[:k]
    return [{**row, "score": float(score), "mode": "visual-maxsim",
             "coverage_warning": coverage_notice or ""} for row, score in best]
