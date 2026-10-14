import json
from pathlib import Path
base = Path(__file__).resolve().parents[1]
notebook = {"cells": [], "metadata": {"colab": {"name": "DocLens_VLM_T4.ipynb", "provenance": []},
                                      "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                      "accelerator": "GPU"}, "nbformat": 4, "nbformat_minor": 5}


def cell(kind, text):
    obj = {"cell_type": kind, "id": f"doclens-{len(notebook['cells']):03d}", "metadata": {}, "source": [ln + '\n' for ln in text.strip().splitlines()]}
    if kind == "code": obj.update(execution_count=None, outputs=[])
    notebook["cells"].append(obj)

cell("markdown", """# DocLens-VLM: T4 experiment notebook

**Research:** ColPali (ICLR 2025), token-level MaxSim. **Backbone:** pretrained ColSmol-500M document visual retriever; SmolVLM-500M answer generator.

**Important:** This notebook uses public pretrained model weights; it does **not** replicate paper-scale training, and the small ViDoRe subsample is **not** the official full benchmark. Never report retrieval metrics before running the cells. Enable **Runtime → Change runtime type → T4 GPU**.
""")
cell("markdown", "## 1. Setup / load source\nUpload `DocLens-VLM_GitHub_Ready.zip` to Colab's Files pane first (left sidebar). Run below. Alternatively clone the GitHub repository once published.")
cell("code", """import glob, zipfile
matches = sorted(glob.glob('/content/DocLens-VLM_GitHub_Ready.zip'))
assert matches, 'Upload DocLens-VLM_GitHub_Ready.zip via the Colab Files sidebar'
print('Using project ZIP:', matches[-1])
with zipfile.ZipFile(matches[-1]) as z:
    z.extractall('/content')
%cd /content/DocLens-VLM
!pip -q install -e '.[vision,evaluation,tests]'
!python -m pytest -q
""")
cell("markdown", "## 2. Inspect GPU memory")
cell("code", """import torch
print('CUDA:', torch.cuda.is_available())
print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')
!nvidia-smi
""")
cell("markdown", "## 3. Make a SMALL, genuinely held-out retrieval benchmark\nWe use the real `vidore/docvqa_test_subsampled` dataset. This custom 50-query retrieval pool is not comparable with full ViDoRe leaderboard scores.")
cell("code", """!python scripts/import_vidore.py --n 50 --seed 42 --out runs/vidore
import json
from pathlib import Path
print(json.loads(Path('runs/vidore/metadata.json').read_text()))
""")
cell("markdown", "## 4. BM25 baseline (requires OCR for image-only pages)\nThe ViDoRe subset images have no text field. Tesseract OCR is installed in Colab below. OCR every page so BM25 receives the same document pool as visual retrieval.")
cell("code", """!apt-get -qq update >/dev/null
!apt-get -qq install -y tesseract-ocr tesseract-ocr-hin >/dev/null
import json
from pathlib import Path
from PIL import Image
from doclens.corpus import extract_ocr_words
folder = Path('runs/vidore')
records = json.loads((folder/'corpus.json').read_text())
for i,r in enumerate(records):
    with Image.open(folder/r['image']) as im:
        r['text'], r['words'] = extract_ocr_words(im, 'eng')
        r['image_size'] = list(im.size)
    if i%10==0: print('OCR',i+1,'/',len(records))
(folder/'corpus.json').write_text(json.dumps(records,ensure_ascii=False,indent=2))
!python -m doclens.cli evaluate --queries runs/vidore/queries.json --corpus runs/vidore --mode bm25 -k 5 --out runs/vidore/bm25_results.json
""")
cell("markdown", "## 5. ColSmol visual embeddings (500M weights)\nFirst run downloads model. If T4 CUDA OOM: reduce sample count to 20 in step 3, restart runtime and rerun; redoing step 3 invalidates any previous metric files.")
cell("code", """!python -m doclens.cli index-visual --corpus runs/vidore
!python -m doclens.cli evaluate --queries runs/vidore/queries.json --corpus runs/vidore --mode visual -k 5 --out runs/vidore/visual_results.json
""")
cell("markdown", "## 6. Compare metrics on EXACTLY the same questions\nHere the metrics are actual measured numbers generated only after evaluation has succeeded.")
cell("code", """from doclens.evaluate import paired_bootstrap_ci
b=json.loads(Path('runs/vidore/bm25_results.json').read_text())
v=json.loads(Path('runs/vidore/visual_results.json').read_text())
print('BM25:',b['metrics'])
print('Visual MaxSim:',v['metrics'])
base=[x['ndcg@5'] for x in b['per_query']]
visual=[x['ndcg@5'] for x in v['per_query']]
print('Visual-minus-BM25 nDCG@5 95% paired bootstrap CI:', paired_bootstrap_ci(base,visual,iterations=2000,seed=42))
""")
cell("markdown", "## 7. Evidence-gated document QA demonstration\nThe VLM proposes an answer; the system shows it **only** if OCR/PDF yields a unique literal line span with an actual pixel box. Otherwise it returns `Not visible in the document.`. OCR-only alignment is not semantic/visual fact verification. If this ViDoRe question cannot be matched literally, abstention is expected. Human verification remains required.")
cell("code", """import gc
from doclens.vision import ColSmolRetriever, retrieve_visual
from doclens.generator import SmolVLMAnswerer, answer_on_page
query = 'What is the dividend payout in 2012?'
retriever = ColSmolRetriever()
hits = retrieve_visual(query,'runs/vidore', retriever, k=3)
print([(x['page_id'],x['score']) for x in hits])
del retriever
gc.collect()
if torch.cuda.is_available(): torch.cuda.empty_cache()
from IPython.display import display
display(Image.open(Path('runs/vidore')/hits[0]['image']))
answerer = SmolVLMAnswerer()
answer = answer_on_page(query,hits[0],'runs/vidore',answerer)
print(answer)
if answer['evidence'] is not None:
    from PIL import ImageDraw
    evidence_im = Image.open(Path('runs/vidore')/hits[0]['image']).convert('RGB')
    ImageDraw.Draw(evidence_im).rectangle(answer['evidence']['bbox_pixels'], outline='red', width=4)
    display(evidence_im)
""")
cell("markdown", "## 8. Launch the interactive interface\nRun one Colab cell at a time. The app's file upload supports user-provided PDFs and images; do not upload confidential documents to a public share URL.")
cell("code", """# Only use a temporary share link if you understand its public-access implications.
# For local notebooks use: !python app.py
print('Application entrypoint: python app.py (uses share=False by default)')
""")
cell("markdown", "## 9. Interpret carefully\nThe main experimental comparison is page retrieval on a small, sampled test pool. It is NOT official benchmark evaluation or accuracy on Indic languages. OCR/PDF word alignment **is not factual verification**; an answer can match a literal region and still be the wrong answer. To test Hindi/Bengali/Tamil, construct a separately labeled dataset with disjoint held-out test pages, query-answer evidence coordinates and per-language error metrics. Document null evidence, duplicate answers and OCR failures. No retrieval score alone demonstrates answer factuality.")
(base/'notebooks/DocLens_VLM_T4.ipynb').write_text(json.dumps(notebook,ensure_ascii=False,indent=2))
