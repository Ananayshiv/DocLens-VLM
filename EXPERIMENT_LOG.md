# Experimental record (2026-10-14)

- Research basis: ColPali, Faysse et al., ICLR 2025; released model `vidore/colSmol-500M`.
- Intended dataset: actual held-out `vidore/docvqa_test_subsampled` (custom 50-question candidate pool).
- **What was executed here:** unit tests, rendering and OCR on three generated fictional demo images, simple BM25 search.
- **What was NOT executed:** ViDoRe download and visual-model inference; no nDCG/recall from a real benchmark; no pretrained model tuning; no Hindi/Bengali/Tamil evaluation.
- Why: source runner has CPU only and no pretrained models pre-downloaded.
- Hypotheses H1–H3 are pending; there is no positive or negative experimental finding yet.

## 2026-10-14 — Evidence-gated response and T4 memory pass (local CPU)

**Executed:** Replaced full Q×P matrix scoring with bounded document-token
chunks (default 512), including exact numeric/gradient comparison with a
dense normalized masked MaxSim reference; visual index search now reads one
stored embedding at a time. No GPU memory-performance claims are made.

Added PDF coordinate extraction from text-layer words, and Tesseract
`image_to_data` line/word bounding boxes for scanned pages. The answer
path abstains unless a unique, same-line, exact Unicode-normalized span
has real image-pixel coordinates; the UI highlights this text span. This
is **text-span support, not semantic or pixel-level verification**.

**Local outcome:** 37/37 unit tests passed, including rotated PDF
coordinates, Hindi combining marks, missing/duplicated spans, multi-line
rejection, wrong VLM proposals, masked MaxSim, gradients, and larger
chunked-document smoke scoring. Checked the fictional invoice and
bilingual notice with local Tesseract: detected unique "25 अक्टूबर 2026"
and "25 October 2026" regions on the notice, and correctly abstained on
unseen amount and duplicated strings. These data are synthetic examples,
not an external research benchmark.

**Outstanding:** GPU loading of pretrained ColSmol/SmolVLM; ViDoRe
retrieval comparison; real Hindi/Indic document and answer-level evaluation;
OCR script errors; visual vs text retrieval failures; wrong-question
literal span matches; calibration, privacy review, and real bbox annotations.
No empirical superiority, answer reliability, or zero-hallucination claim.

## 2026-10-14 — Audit hardening (items 2, 8, 9, 10, 12)

Implemented symmetrical sign/currency normalization for *comparison only*;
logged the loss-of-sign limitation explicitly because `$500` / `-500` / `500`
may not represent economically equivalent amounts. Added 30-page PDF truncation
warnings with persistent ingest coverage metadata and an unlimited-pages re-index
option. Made ingestion append/merge by stable content-hash ID, with atomic JSON
replacement and preservation of prior OCR text. Applied EXIF orientation before
OCR and visual embedding. Rejected empty BM25 text indexes. Protected ViDoRe
import from overwriting unrelated evaluation corpora.

All changes are implementation-level controls. No pretrained GPU inference or
external benchmark evaluation was conducted for this patch.

**Verification of this patch:** 56/56 local unit tests passed with `python -m
pytest -q`; compileall passed. Tests included a real locally generated 31-page
PDF and a synthetic JPEG carrying EXIF orientation 6. No external OCR/VLM
accuracy measurements were run.

## 2026-11-23 — Hybrid retrieval, one-command benchmark, HTTP API

**Implemented.** `doclens/hybrid.py`: Reciprocal Rank Fusion (k=60, equal
weights, fixed a priori; not tuned on any evaluation question) of BM25 and
visual MaxSim over the same page pool. Zero-score BM25 pages are excluded from
the lexical ranking; an empty OCR index degrades to visual-only with an explicit
`fusion_warning`. `doclens benchmark` evaluates bm25/visual/hybrid on identical
queries with one ColSmol load and writes `summary.json`, `per_query.json` and a
markdown table with paired-bootstrap 95% CIs (resampling whole queries) against
the first-listed mode. `doclens/api.py` exposes `/search` and `/ask` via FastAPI
with the unchanged abstention contract. The Colab notebook now runs the 3-way
benchmark on a 100-query ViDoRe DocVQA pool (was 50) for tighter intervals.

**Verified (CPU, no GPU, no network):** new `tests/test_hybrid_benchmark.py`
(RRF formula, agreement ordering, duplicate handling, weights, determinism,
Devanagari IDs, paired deltas, mismatched-query rejection) plus the existing
lexical/evaluation tests: 11/11 passed; `compileall` passed for `doclens`,
`app.py`, `scripts`. The environment used for this change had no PyTorch
installed, so the remaining suites (MaxSim, evidence, corpus, adapter) were
**not run** here and must be re-run with `python -m pytest -q` before merging.
`doclens/api.py` was not executed (FastAPI not installed).

**Not run:** ColSmol/SmolVLM inference, ViDoRe benchmark, hybrid vs single-mode
comparison (H4). No retrieval numbers are claimed. README table cells are
explicitly `pending`.

## 2026-11-23 — Measured: BM25 vs visual vs hybrid on ViDoRe DocVQA (CPU)

**Run:** GitHub Actions run 45085271677, `ubuntu-latest` CPU runner. Dataset
`vidore/docvqa_test_subsampled` (test), 100 questions sampled with seed 42,
100 candidate pages (one gold page per question). Retriever
`vidore/colSmol-500M` (fp32, CPU), Tesseract `eng` OCR for BM25 (2/100 pages
produced no OCR text), k=5, paired bootstrap 2,000 resamples over questions,
seed 42. Library constraint: `transformers<4.53` (newer releases crash while
listing Hub chat templates for the ColSmol processor; `doclens.vision` now
falls back to a local snapshot). Visual indexing took ~40 min on CPU.

| Mode | MRR | Recall@5 | nDCG@5 |
|---|---:|---:|---:|
| bm25 | 0.3773 | 0.47 | 0.4002 |
| visual | 0.6625 | 0.80 | 0.6972 |
| hybrid (RRF k=60, 1:1) | 0.5740 | 0.68 | 0.6004 |

visual − bm25 nDCG@5: +0.2970 [+0.2216, +0.3783]. hybrid − visual nDCG@5:
−0.0968 [−0.1683, −0.0300]; MRR −0.0885 [−0.1602, −0.0200]; Recall@5 −0.12
[−0.20, −0.04]. Complementarity check (Recall@5): BM25-only hits 1, visual-only
hits 34. **Conclusion:** H1 supported on this subset; H4 rejected. No fusion
weights were tuned on these questions. Failure analysis per question is in
`runs/benchmark/per_query.json`. Not run: GPU timing, Indic documents,
answer-level grounding evaluation.
