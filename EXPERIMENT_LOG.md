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
