"""Conservative, auditable *textual* evidence localization on document pixels.

Word coordinates MUST originate from PDF text-layer geometry or OCR bounding
boxes; neither a MaxSim score nor a VLM response gives a trustworthy bbox.
Even successful OCR/PDF string matching is not visual fact verification.
"""
from __future__ import annotations

from collections import Counter
import unicodedata

ABSTAIN = "Not visible in the document."


def tokens(text: str) -> list[str]:
    """Unicode-aware lexical tokens without discarding Indic combining marks."""
    if not isinstance(text, str):
        raise TypeError("Text must be Unicode str")
    normalized = unicodedata.normalize("NFKC", text).casefold()
    groups: list[str] = []
    current: list[str] = []
    for ch in normalized:
        category = unicodedata.category(ch)
        if category[0] in "LN" or category in {"Mn", "Mc", "Me"}:
            current.append(ch)
        elif current:
            groups.append("".join(current))
            current = []
    if current:
        groups.append("".join(current))
    return groups


def _validate_word(item: dict, image_size: tuple[int, int]) -> bool:
    width, height = image_size
    if not isinstance(item, dict) or not item.get("text"):
        return False
    box = item.get("bbox")
    if not isinstance(box, (tuple, list)) or len(box) != 4:
        return False
    try:
        x0, y0, x1, y1 = map(float, box)
    except (ValueError, TypeError):
        return False
    return 0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height


def locate_exact_span(answer: str, words: list[dict], image_size: tuple[int, int],
                      max_tokens: int = 30) -> dict | None:
    """Return image-pixel region ONLY for a unique contiguous exact text span.

    Reject ambiguous occurrences and multiline spans. No visual token-to-patch
    saliency, score threshold, or guessed coordinates. Text match alone does not
    establish semantic correctness or that OCR transcribed the pixels correctly.
    """
    target = tokens(answer)
    if not target or len(target) > max_tokens or answer.strip().upper() in {"NOT_FOUND", "NOT FOUND"}:
        return None
    width, height = image_size
    if width <= 0 or height <= 0:
        raise ValueError("Invalid image dimensions")
    candidates: list[dict] = []
    valid = [w for w in words if _validate_word(w, image_size)]
    for start in range(len(valid)):
        observed: list[str] = []
        anchor = valid[start].get("line_id")
        for finish in range(start, min(len(valid), start + max_tokens)):
            word = valid[finish]
            if word.get("line_id") != anchor:
                break
            observed.extend(tokens(word["text"]))
            if len(observed) >= len(target):
                if observed == target:
                    region = valid[start:finish + 1]
                    x0 = min(float(w["bbox"][0]) for w in region)
                    y0 = min(float(w["bbox"][1]) for w in region)
                    x1 = max(float(w["bbox"][2]) for w in region)
                    y1 = max(float(w["bbox"][3]) for w in region)
                    candidates.append({
                        "bbox_pixels": [round(x0, 2), round(y0, 2), round(x1, 2), round(y1, 2)],
                        "bbox_normalized": [round(x0/width, 6), round(y0/height, 6),
                                            round(x1/width, 6), round(y1/height, 6)],
                        "matched_text": " ".join(str(w["text"]) for w in region),
                        "evidence_origin": region[0].get("origin", "unknown"),
                    })
                break
    return candidates[0] if len(candidates) == 1 else None


def ground_proposal(proposal: str, page: dict, image_size: tuple[int, int]) -> dict:
    """Conservatively expose a proposed answer iff literal span has one bbox.

    Exposes `verification_status=text_span_supported`, never `fact_verified`.
    Explicit abstention when absent, ambiguous, OCR-only without coordinates,
    or VLM response is paraphrased/non-literal.
    """
    raw = (proposal or "").strip()
    if not raw or raw.upper().startswith("NOT_FOUND") or raw == ABSTAIN:
        reason = "model_abstained"
        located = None
    else:
        located = locate_exact_span(raw, page.get("words", []), image_size)
        reason = "unique_literal_match" if located else "no_unique_located_text_span"
    return {
        "answer": raw if located else ABSTAIN,
        "abstained": located is None,
        "verification_status": "text_span_supported" if located else "abstained_unverified",
        "evidence": located,
        "reason": reason,
        "note": "OCR/PDF span support is not visual or semantic fact verification; review source pixels.",
    }
