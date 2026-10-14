"""Conservative OCR-text key/value extraction with explicit non-verification warning.

Only extract explicit spans. Incomplete or unreadable documents must yield null
rather than guessing. Not intended for automated financial or legal decisions.
"""
from __future__ import annotations
import re
import unicodedata

from .evidence import tokens as evidence_tokens

FIELD_PATTERNS = {
    "invoice_number": [r"(?im)^\s*(?:invoice\s*(?:no\.?|number|#))\s*:\s*([^\n]+)$"],
    "total_payable": [r"(?im)^\s*(?:total\s*(?:payable|amount|due))\s*:\s*([^\n]+)$"],
    "application_deadline": [
        r"(?im)^\s*(?:application\s*deadline|due\s*date)\s*:\s*([^\n]+)$",
        r"(?m)आवेदन[^\n]{0,55}अंतिम\s*तिथि\s*:\s*([^\n]+)$",
    ],
    "meeting_date": [
        r"(?im)^\s*(?:review\s*meeting|meeting\s*date)\s*:\s*([^\n]+)$",
        r"(?m)समीक्षा\s*बैठक\s*:\s*([^\n]+)$",
    ],
}


def extract_fields(ocr_text: str) -> dict:
    """Return fields supported by an explicit matching line or null."""
    if not isinstance(ocr_text, str):
        raise TypeError("ocr_text must be a string")
    fields = {}
    for key, patterns in FIELD_PATTERNS.items():
        match = next((m for pattern in patterns if (m := re.search(pattern, ocr_text))), None)
        fields[key] = match.group(1).strip() if match else None
    return {"fields": fields, "evidence_type": "regex matching on OCR/PDF text",
            "warning": "Not independently verified against original pixels; manual review required."}


def comparison_tokens(value: str) -> list[str]:
    """Normalize fields consistently for *comparative* evaluation.

    NFKC + Unicode tokenization strips currency symbols, surrounding signs,
    brackets and other punctuation on BOTH sides. Thus `($500)`, `-500`,
    `$500` and `500` intentionally compare equal, as required by the audit.
    This is SIGN-INSENSITIVE textual comparison, NOT a financially safe
    amount validator (a real negative balance differs from a positive one).

    A thousands-grouping comma is removed before tokenization; Unicode Indic
    digits are converted to ASCII digits for cross-script number comparison.
    Never use a substring test for amounts (`500` must not match `5000`).
    """
    if not isinstance(value, str):
        raise TypeError("value must be a Unicode string")
    normalized = unicodedata.normalize("NFKC", value).casefold()
    normalized = "".join(
        str(unicodedata.decimal(ch)) if ch.isdecimal() else ch
        for ch in normalized
    )
    # Only delete commas forming valid 3-digit number groups; ordinary commas
    # remain delimiters. E.g. 1,250.00 -> 1250.00, but 1,25 stays separate.
    normalized = re.sub(r"(?<=\d),(?=\d{3}(?:\D|$))", "", normalized)
    # Normalize a trailing .0 / .00 on number values, without changing a
    # nonzero fractional value. Keep punctuation handling symmetric.
    normalized = re.sub(r"(?<![\w.])(\d+)\.0+(?!\d)", r"\1", normalized)
    return evidence_tokens(normalized)


def value_appears_in_text(value: str, ocr_text: str) -> bool:
    """Check whole normalized token spans, never raw substrings or facts.

    By design this normalization does *not* preserve monetary polarity.
    Applications making financial decisions must compare parsed signed values
    separately, using the original extracted text.
    """
    if not isinstance(ocr_text, str):
        raise TypeError("ocr_text must be a string")
    needle = comparison_tokens(value) if isinstance(value, str) else []
    haystack = comparison_tokens(ocr_text)
    return bool(needle) and any(
        haystack[i:i + len(needle)] == needle
        for i in range(len(haystack) - len(needle) + 1)
    )
