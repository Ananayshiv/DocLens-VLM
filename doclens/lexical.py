"""Unicode-aware BM25, a strong OCR/text-page baseline for visual retrieval."""
from __future__ import annotations
from collections import Counter
import math
import re
import unicodedata


def tokenize(text: str) -> list[str]:
    # Preserve Indic combining marks as part of the preceding grapheme/token.
    text = unicodedata.normalize("NFKC", text).casefold()
    tokens, current = [], []
    for char in text:
        cat = unicodedata.category(char)
        if cat[0] in "LN" or cat in {"Mn", "Mc", "Me"}:
            current.append(char)
        elif current:
            tokens.append("".join(current))
            current = []
    if current:
        tokens.append("".join(current))
    return tokens


class BM25:
    def __init__(self, documents: list[str], k1: float = 1.5, b: float = 0.75):
        self.tf = [Counter(tokenize(d)) for d in documents]
        self.n = len(self.tf)
        if not self.n or not any(self.tf):
            raise ValueError("Empty BM25 text index: no searchable OCR/PDF text. "
                             "Index documents with --ocr or supply text-bearing PDFs.")
        self.k1, self.b = k1, b
        self.avgdl = sum(sum(c.values()) for c in self.tf) / self.n if self.n else 0.
        self.df = Counter()
        for counts in self.tf:
            self.df.update(counts.keys())

    def scores(self, query: str) -> list[float]:
        if not tokenize(query):
            raise ValueError("Empty BM25 query: provide searchable Unicode text")
        out = []
        for counts in self.tf:
            dl = sum(counts.values())
            score = 0.
            for term in set(tokenize(query)):
                if not counts[term]:
                    continue
                idf = math.log(1 + (self.n - self.df[term] + .5) / (self.df[term] + .5))
                den = counts[term] + self.k1 * (1 - self.b + self.b * dl / max(self.avgdl, 1e-8))
                score += idf * counts[term] * (self.k1 + 1) / den
            out.append(score)
        return out
