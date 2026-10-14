"""Document page rendering and persisted local corpus; no external document upload."""
from __future__ import annotations
import hashlib
import io
import json
import math
import os
import tempfile
import warnings
from pathlib import Path
from PIL import Image, ImageOps

SUPPORTED = {".pdf", ".png", ".jpg", ".jpeg", ".webp"}
MANIFEST_FILE = "ingestion_manifest.json"


def render_document(path: Path, dpi: int = 130, max_pages: int = 30):
    """Yield (1-based page, RGB image, embedded selectable PDF text)."""
    if not path.is_file() or path.suffix.lower() not in SUPPORTED:
        raise ValueError(f"Unsupported document: {path}")
    if not isinstance(max_pages, int) or max_pages < 0:
        raise ValueError("max_pages must be >= 0; use 0 to index every page")
    if path.suffix.lower() == ".pdf":
        import fitz
        with fitz.open(path) as pdf:
            if max_pages and len(pdf) > max_pages:
                warnings.warn(
                    f"PDF {path} has {len(pdf)} pages and exceeds the {max_pages}-page "
                    f"ingestion limit; pages {max_pages + 1}-{len(pdf)} were NOT indexed. "
                    "A 'Not visible' answer is not reliable for omitted pages. "
                    "Re-ingest with --max-pages 0 to include all pages.",
                    UserWarning, stacklevel=2,
                )
            for i in range(len(pdf) if max_pages == 0 else min(len(pdf), max_pages)):
                page = pdf[i]
                pix = page.get_pixmap(matrix=fitz.Matrix(dpi / 72, dpi / 72), alpha=False)
                image = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
                yield i + 1, image, page.get_text(sort=True)
    else:
        with Image.open(path) as image:
            # Phone camera JPEG EXIF orientation is applied before OCR, hashing,
            # rendering to a page PNG, or passing the pixels to a VLM.
            yield 1, ImageOps.exif_transpose(image).convert("RGB"), ""


def extract_ocr_words(image: Image.Image, languages: str = "eng+hin") -> tuple[str, list[dict]]:
    """Return OCR text and page-pixel word regions from one Tesseract call.

    Line grouping preserves mixed Hindi/English and prevents cross-line matches.
    OCR may still mis-transcribe text; boxes only prove where OCR read it.
    """
    try:
        import pytesseract
        data = pytesseract.image_to_data(image, lang=languages, output_type=pytesseract.Output.DICT)
    except Exception as exc:  # Includes absent OCR engine and TesseractError
        raise RuntimeError(f"Tesseract OCR unavailable for {languages}: {exc}") from exc
    words = []
    lines: dict[tuple[int, int, int], list[str]] = {}
    for i, raw in enumerate(data["text"]):
        text = (raw or "").strip()
        if not text:
            continue
        try:
            confidence = float(data["conf"][i])
        except (TypeError, ValueError):
            confidence = -1
        if not math.isfinite(confidence) or confidence < 0:
            continue
        x, y = int(data["left"][i]), int(data["top"][i])
        right, bottom = x + int(data["width"][i]), y + int(data["height"][i])
        if right <= x or bottom <= y:
            continue
        # Some OCR engines return slight overhangs at image borders.
        x, y = max(0, x), max(0, y)
        right, bottom = min(image.width, right), min(image.height, bottom)
        if right <= x or bottom <= y:
            continue
        line = (int(data["block_num"][i]), int(data["par_num"][i]), int(data["line_num"][i]))
        lines.setdefault(line, []).append(text)
        words.append({"text": text, "bbox": [x, y, right, bottom],
                      "line_id": ":".join(map(str, line)), "origin": "tesseract_ocr", "ocr_confidence": confidence})
    return "\n".join(" ".join(items) for items in lines.values()), words


def extract_ocr(image: Image.Image, languages: str = "eng+hin") -> str:
    return extract_ocr_words(image, languages=languages)[0]



def _pdf_pixel_words(path: Path, page_number: int, image_size: tuple[int, int]) -> list[dict]:
    """Map embedded-PDF word coordinates into rendered image pixels.

    PyMuPDF supplies words in unrotated PDF coordinates; rotating the rectangles
    into page.rect space allows matching highlights on rotated PDFs.
    """
    import fitz
    out = []
    with fitz.open(path) as pdf:
        page = pdf[page_number - 1]
        width, height = image_size
        rx, ry = page.rect.width, page.rect.height
        if rx <= 0 or ry <= 0:
            return []
        for w in page.get_text("words", sort=True):
            x0, y0, x1, y1, txt, block, line, _ = w[:8]
            rect = fitz.Rect(x0, y0, x1, y1) * page.rotation_matrix
            box = [max(0., rect.x0 * width / rx), max(0., rect.y0 * height / ry),
                   min(float(width), rect.x1 * width / rx), min(float(height), rect.y1 * height / ry)]
            if box[2] > box[0] and box[3] > box[1]:
                out.append({"text": txt, "bbox": [round(float(v), 2) for v in box],
                            "line_id": f"{block}:{line}", "origin": "pdf_text_layer"})
    return out


def _atomic_json_write(path: Path, data: object) -> None:
    """Avoid corrupting a pre-existing corpus if serialization/write fails."""
    path.parent.mkdir(parents=True, exist_ok=True)
    name = None
    try:
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent,
                                         prefix=f".{path.name}.", suffix=".tmp",
                                         delete=False) as stream:
            name = stream.name
            json.dump(data, stream, ensure_ascii=False, indent=2)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, path)
    finally:
        if name and os.path.exists(name):
            os.unlink(name)


def load_ingestion_manifest(directory: str | Path) -> dict:
    """Read persisted coverage for detecting partial PDF ingestion later."""
    path = Path(directory) / MANIFEST_FILE
    if not path.exists():
        return {"sources": {}}
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or not isinstance(value.get("sources", {}), dict):
        raise ValueError(f"Malformed ingestion manifest: {path}")
    return value


def ingestion_coverage_notice(directory: str | Path) -> str | None:
    incomplete = [entry for entry in load_ingestion_manifest(directory)["sources"].values()
                  if entry.get("truncated_pages", 0) > 0]
    if not incomplete:
        return None
    details = ", ".join(
        f"{item['source']} ({item['truncated_pages']} omitted pages)"
        for item in incomplete
    )
    return (f"INCOMPLETE CORPUS: {details}. Search results and 'Not visible' "
            "answers cannot exclude content on missing pages. Re-ingest the "
            "affected PDF(s) with --max-pages 0.")


def _merge_record(existing: dict, incoming: dict) -> dict:
    """Merge identical page hashes without discarding richer prior OCR data."""
    if existing["page_id"] != incoming["page_id"]:
        raise ValueError("Cannot merge different document IDs")
    merged = {**existing, **incoming}
    for key in ("text", "words"):
        if not incoming.get(key) and existing.get(key):
            merged[key] = existing[key]
    merged["source"] = existing.get("source", incoming["source"])
    merged["page"] = existing.get("page", incoming["page"])
    sources = list(existing.get("sources", [existing.get("source", "")]))
    for other in incoming.get("sources", [incoming["source"]]):
        if other and other not in sources:
            sources.append(other)
    merged["sources"] = sources
    return merged


def ingest(paths: list[str | Path], directory: str | Path, *,
           max_pages_per_file: int = 30, ocr: bool = False,
           ocr_languages: str = "eng+hin") -> list[dict]:
    """Non-destructive append/merge corpus ingestion with coverage metadata.

    Duplicate page IDs are content hashes. Re-ingestion can enrich earlier
    records with OCR, but does not erase useful text/boxes when OCR is off.
    A zero page limit indexes all pages. The default 30-page cap is warned
    about and persistently recorded for downstream search/answer users.
    """
    if not isinstance(max_pages_per_file, int) or max_pages_per_file < 0:
        raise ValueError("max_pages_per_file must be >= 0 (0 means unlimited)")
    directory = Path(directory)
    pages_dir = directory / "pages"
    pages_dir.mkdir(parents=True, exist_ok=True)
    corpus_file = directory / "corpus.json"
    original = load_corpus(directory) if corpus_file.exists() else []
    if not isinstance(original, list):
        raise ValueError("Existing corpus.json must contain a JSON list")
    by_id: dict[str, dict] = {}
    for row in original:
        if not isinstance(row, dict) or not isinstance(row.get("page_id"), str):
            raise ValueError("Existing corpus contains a malformed page record")
        if row["page_id"] in by_id:
            raise ValueError(f"Duplicate page_id already exists in corpus: {row['page_id']}")
        by_id[row["page_id"]] = row

    manifest = load_ingestion_manifest(directory)
    for raw in paths:
        path = Path(raw)
        if not path.is_file() or path.suffix.lower() not in SUPPORTED:
            raise ValueError(f"Unsupported document: {path}")
        n_pages = 1
        if path.suffix.lower() == ".pdf":
            import fitz
            with fitz.open(path) as pdf:
                n_pages = len(pdf)
        rendered = 0
        for page_no, image, embedded_text in render_document(path, max_pages=max_pages_per_file):
            rendered += 1
            buf = io.BytesIO()
            image.save(buf, format="PNG")
            blob = buf.getvalue()
            page_id = hashlib.sha256(blob).hexdigest()[:20]
            image_path = pages_dir / f"{page_id}.png"
            # Stable content-addressed destination shared by duplicate inputs.
            if not image_path.exists():
                image_path.write_bytes(blob)
            text = embedded_text
            words = _pdf_pixel_words(path, page_no, image.size) if path.suffix.lower() == ".pdf" else []
            if ocr and not words:
                text, words = extract_ocr_words(image, languages=ocr_languages)
            record = {"page_id": page_id, "source": path.name,
                      "page": page_no, "image": f"pages/{page_id}.png", "text": text,
                      "words": words, "image_size": list(image.size)}
            by_id[page_id] = (_merge_record(by_id[page_id], record)
                             if page_id in by_id else record)
        manifest["sources"][str(path.resolve())] = {
            "source": path.name, "total_pages": n_pages, "rendered_pages": rendered,
            "truncated_pages": max(0, n_pages-rendered),
            "max_pages_per_file": max_pages_per_file,
        }
    # Write pages first, then coverage metadata. If the second write fails,
    # the old manifest retains a conservative incomplete-coverage warning;
    # it must never claim full coverage for pages not yet committed.
    records = list(by_id.values())
    _atomic_json_write(corpus_file, records)
    _atomic_json_write(directory / MANIFEST_FILE, manifest)
    return records


def load_corpus(directory: str | Path) -> list[dict]:
    return json.loads((Path(directory) / "corpus.json").read_text(encoding="utf-8"))
