# DocLens-VLM: GitHub release notes

**Stage:** Research/engineering prototype; CPU unit tests and fictional-data smoke tests exist. The pretrained visual-retriever and answer-model inference path, T4 memory usage, and ViDoRe results still require a live GPU run. No comparison against OCR retrieval is claimed to be superior at this stage.

## What is included

- Hand-implemented, normalized, chunked MaxSim scoring with unit tests
- PyTorch low-rank retrieval adapter experiment and training entrypoint
- OCR/PDF-text BM25 baseline, PDF/photo ingestion, EXIF correction and idempotent corpus updates
- Evidence-rectangle localization from PDF word spans / Tesseract OCR, explicit abstention for unsupported outputs
- Basic GUI, command-line utilities, end-to-end Colab experiment notebook, and CI test workflow
- Indic-script examples, failure-mode documentation, and audit regression tests

## What is not included

- Trained ColPali/ColSmol weights (downloaded from upstream)
- New fine-tuned model weights, production accuracy validation, full benchmark results
- Sensitive/private documents or customer data

## Minimal reproducibility checks

```bash
python -m pip install -e '.[tests]'
python -m pytest -q
python scripts/make_demo.py
python -m doclens.cli ingest assets/*.png --out runs/demo --ocr --ocr-languages eng+hin
python -m doclens.cli search 'When is the application deadline?' --corpus runs/demo --mode bm25
```

For visual evaluation, use `notebooks/DocLens_VLM_T4.ipynb` on a Colab T4 and preserve the generated JSON metrics. The notebook's ViDoRe subset is **not** an official full benchmark result.

**Model/data licensing:** This repository's MIT license covers its original source code, not the separately licensed external pretrained models or third-party datasets.
