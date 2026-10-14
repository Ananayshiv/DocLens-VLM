# DocLens-VLM — Agent instructions (root)

Read `.agents/skills/doclens-vlm/SKILL.md` before changing code, evaluations,
README claims, prompts, or Colab notebooks. Treat it as the project engineering
contract. This root file applies to the entire repository, including agents
that do not load project-specific skills automatically.

## Non-negotiable constraints

1. **PyTorch & tensor integrity.** Preserve `[query_tokens, dim]` and
   `[page_tokens, dim]` invariants. Default MaxSim is normalized cosine using
   `torch.nn.functional.normalize`, masked padding, and chunked query×page
   similarity to prevent T4 out-of-memory failures. For mathematical changes,
   test scores and gradients against a dense implementation.
2. **Indic Unicode robustness.** Preserve Hindi, Indic combining characters,
   native numerals, Latin/Indic mixed-script lines, and UTF-8 serialization.
   Never strip non-ASCII script tokens or assume English-only extraction.
3. **Evaluation integrity.** Never report ViDoRe / Indic-language gains,
   production performance, or fine-tuning improvements without successful
   measured runs. Do not tune on held-out test questions; always state
   model IDs, seed, dataset split and candidate pool. Log confounders and
   failure cases in `EXPERIMENT_LOG.md`.
4. **Evidence and abstention.** MaxSim is a retrieval score, not localization.
   VLM proposals are not facts. Emit a non-abstained answer only if there is
   a **unique, exact, contiguous** OCR/PDF-text evidence span with real
   image-pixel coordinates. Return exactly `Not visible in the document.`
   when evidence is absent, ambiguous, occluded, unparsable, or unverifiable.
   Include source, page number, bounding box, and provenance for any supported
   answer. A text match is *not* proof the underlying pixels or semantics are
   correct; preserve the `text_span_supported` label, not `fact_verified`.

## Required pre-merge checks

```bash
python -m pytest -q
python -m compileall -q doclens app.py
```

If a step requires downloaded pretrained weights or labeled external data,
report that step as **not run** rather than silently using synthetic results.
Never commit sensitive documents or large checkpoints.

## Corpus safety invariants (audit 2026-10-14)

- Keep `ingest` idempotent and non-destructive for pre-existing corpus IDs.
- Retain persistent PDF coverage warnings until all omitted pages are indexed;
  absence assertions must never silently ignore unindexed pages.
- Apply EXIF auto-orientation before OCR/VLM and deriving pixel coordinates.
- Refuse empty lexical BM25 indexes; do not rank empty text as meaningful.
- Sign/currency-stripped value comparisons are only for normalized **textual**
  scorekeeping, not semantic equality for debits, credits, or financial values.
