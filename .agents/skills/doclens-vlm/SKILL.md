---
name: doclens-vlm
# Loaded by Codex when implementing or evaluating DocLens-VLM.
description: Maintain an auditable Hindi/English multimodal document retriever with ColPali-style MaxSim, T4-safe memory use, real-region grounding, conservative abstention and reproducible evaluation.
---

# SKILL: DocLens-VLM Multimodal Document Assistant & Evaluation Engine

## Scope and context

DocLens-VLM is a PyTorch-based document intelligence prototype inspired by
ColPali (ICLR 2025). It retrieves page images via token-level *MaxSim* late
interaction and optionally uses a pretrained SmolVLM model to propose answers.
The application targets multilingual Indian invoices, forms, circulars,
reports, Hindi-English mixed text, and complex layouts. It is **not** a
verified financial/compliance decision system.

## Workflow for code edits

1. Read `AGENTS.md`, `doclens/late_interaction.py`, `doclens/evidence.py`,
   `doclens/generator.py`, and affected tests first.
2. Preserve `[Q,D]` × `[P,D]` tensor dimensions; normalize via `F.normalize`
   and mask padded tokens. Use document-token chunks (`chunk_size=512`
   by default), and stream page embeddings during scoring. Do not silently
   replace cosine MaxSim with dot product or global-vector scoring.
3. Handle Unicode NFKC consistently. Check combining marks, numerals and
   mixed scripts; include tests involving Devanagari (U+0900–U+097F).
4. Treat VLM answer text as a hypothesis. PDF text-layer word boxes or OCR
   `image_to_data` boxes are the ONLY accepted coordinate sources here.
   Localize a unique, exact, same-line span. On any missing or ambiguous
   span, **abstain** with `Not visible in the document.`. Surface the source,
   1-indexed page and [left,top,right,bottom] pixel rectangle.
5. Treat `text_span_supported` as weaker than verified visual/semantic
   grounding. OCR hallucinations and wrong-question matched strings remain
   possible. Do not use attention heatmaps as pixel-level evidence.
6. Run the full CPU tests before making empirical claims. Train/infer on the
   real ViDoRe/Indic datasets only when actually available; record both
   successful and failed runs in `EXPERIMENT_LOG.md`.

## Evaluation protocol

- Compare OCR+BM25, visual MaxSim, and any hybrid system on the **same**
  document pool and question set.
- Report MRR, Recall@k, nDCG@k, paired bootstrap confidence intervals and
  per-language/per-scan-quality subgroups where labels permit.
- Use disjoint templates and held-out test questions for learned adaptations.
- Evaluate answer correctness, supporting-page precision, correctly localized
  box rate, false unsupported answers, abstention rate, and false abstention.
- Record OCR layout fragmentation, text-layer inconsistencies, duplicate
  spans, visually hidden spans, VLM paraphrase mismatch, script-token issues,
  and MaxSim visual-token confusion as failure modes.
- Clearly distinguish custom ViDoRe subsets from official full benchmark
  scores. Never fabricate metrics or describe a smoke test as SOTA.

## Minimal verification

```bash
python -m pytest -q
python -m compileall -q doclens app.py
```

GPU evaluation, download-based tests, and adapter training are **pending**
until they have actually been executed. Keep references and claims honest.
