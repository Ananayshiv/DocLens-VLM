"""Gradio app for local PDF/image visual retrieval with visible source-page evidence.

Run: python app.py. Defaults to BM25 (CPU) for zero-download usability; visual
mode loads pretrained VLM weights and needs network on first run.
"""
from __future__ import annotations
import tempfile
from pathlib import Path
from PIL import Image, ImageDraw
import gradio as gr
from doclens.corpus import ingest, ingestion_coverage_notice
from doclens.search import lexical_search

SESSION_DIR = Path(tempfile.mkdtemp(prefix="doclens_"))
_VISUAL = None
_ANSWER = None


def import_files(files, run_ocr, ocr_lang, index_all_pages=False):
    if not files:
        return "Upload a PDF or image first.", ""
    try:
        paths = [Path(f.name if hasattr(f, "name") else f) for f in files]
        rows = ingest(paths, SESSION_DIR, ocr=run_ocr, ocr_languages=ocr_lang,
                      max_pages_per_file=0 if index_all_pages else 30)
        notice = ingestion_coverage_notice(SESSION_DIR)
        return (f"Corpus contains {len(rows)} distinct pages. Stored only in this session's "
                f"local temporary directory." + (f"\n\n⚠️ {notice}" if notice else "")), str(SESSION_DIR)
    except Exception as e:
        return f"Indexing failed: {e}", ""


def search_and_answer(question, mode, generate):
    global _VISUAL, _ANSWER
    if not question.strip():
        return "Enter a question.", None, "", {}
    if not (SESSION_DIR / "corpus.json").exists():
        return "Upload documents and click Index first.", None, "", {}
    try:
        if mode == "Visual VLM (ColSmol)":
            from doclens.vision import ColSmolRetriever, build_index, retrieve_visual
            if _VISUAL is None:
                _VISUAL = ColSmolRetriever()
            build_index(SESSION_DIR, _VISUAL)
            results = retrieve_visual(question, SESSION_DIR, _VISUAL, k=3)
        else:
            results = lexical_search(question, SESSION_DIR, k=3)
        if not results:
            return "No pages found.", None, "", {}
        best = results[0]
        coverage_notice = best.get("coverage_warning", "")
        img = Image.open(SESSION_DIR / best["image"]).convert("RGB")
        sources = "\n".join(f"{i}. {r['source']} — page {r['page']} (retrieval score {r['score']:.3f})"
                            for i, r in enumerate(results, 1))
        if generate:
            # Deallocate large retriever GPU memory before loading the answer VLM.
            if _VISUAL is not None:
                del _VISUAL
                _VISUAL = None
                import gc, torch
                gc.collect()
                if torch.cuda.is_available(): torch.cuda.empty_cache()
            from doclens.generator import SmolVLMAnswerer, answer_on_page
            if _ANSWER is None: _ANSWER = SmolVLMAnswerer()
            grounded = answer_on_page(question, best, SESSION_DIR, _ANSWER)
            evidence = grounded["evidence"]
            if evidence is not None:
                img = img.copy()
                pen = ImageDraw.Draw(img)
                pen.rectangle(evidence["bbox_pixels"], outline="red", width=4)
                answer = (f"**Text-span-supported proposed answer:** {grounded['answer']}\n\n"
                          f"Document: **{best['source']}**, page **{best['page']}**; "
                          f"source: `{evidence['evidence_origin']}`; "
                          f"pixel box: `{evidence['bbox_pixels']}`. "
                          "OCR/text-layer agreement does **not** prove the underlying pixels or semantics; check the highlighted region.")
            else:
                answer = ("**Not visible in the document.**\n\nNo unique supporting text span "
                          "with a genuine page-region coordinate was found. "
                          "For image-only documents, enable OCR during indexing. "
                          "A confident-sounding VLM response is not evidence.")
        else:
            answer = "Retrieved the source page. Enable 'VLM answer' to generate a short answer from the page image."
        if coverage_notice:
            answer = "⚠️ **INCOMPLETE CORPUS:** " + coverage_notice + "\n\n" + answer
        from doclens.fields import extract_fields
        return answer, img, sources, extract_fields(best.get("text", ""))
    except Exception as e:
        return f"Execution error: {type(e).__name__}: {e}", None, "", {}


with gr.Blocks(title="DocLens-VLM") as demo:
    gr.Markdown("# DocLens-VLM\n**Paper-inspired visual retrieval + VLM page answers** · English / Hindi document search · Local files · Inspect every cited page")
    gr.Markdown("**Warning:** Strict literal evidence policy: no localized matching text span → abstention. OCR/PDF coordinate matches are not independent pixel verification. Do not upload confidential records to a hosted demo.")
    files = gr.File(label="PDF or scan images", file_count="multiple", file_types=[".pdf", ".png", ".jpg", ".jpeg", ".webp"])
    with gr.Row():
        ocr = gr.Checkbox(value=True, label="Tesseract OCR for image-only pages (install language packs; required for region-grounded answers)")
        lang = gr.Textbox(value="eng+hin", label="Tesseract languages")
        all_pages = gr.Checkbox(value=False, label="Index ALL PDF pages (slower; avoids the default 30-page cap)")
    index_button = gr.Button("Index documents", variant="primary")
    index_status = gr.Markdown()
    session = gr.Textbox(visible=False)
    index_button.click(import_files, inputs=[files, ocr, lang, all_pages], outputs=[index_status, session])
    question = gr.Textbox(label="Ask a question", placeholder="इस दस्तावेज़ में अंतिम तारीख क्या है? / What is the due date?")
    mode = gr.Radio(choices=["OCR/text BM25", "Visual VLM (ColSmol)"], value="OCR/text BM25", label="Retrieval method")
    answer_toggle = gr.Checkbox(value=False, label="Generate answer with SmolVLM-500M (downloads pretrained model)")
    search_button = gr.Button("Find evidence and answer")
    with gr.Row():
        response = gr.Markdown()
        source_image = gr.Image(label="Evidence page (red box = OCR/PDF-located answer)", type="pil")
    sources = gr.Textbox(label="Top retrieved pages", lines=4)
    extracted = gr.JSON(label="Conservative OCR/PDF field extraction (best-matched page)")
    search_button.click(search_and_answer, inputs=[question, mode, answer_toggle], outputs=[response, source_image, sources, extracted])

if __name__ == "__main__":
    demo.launch(share=False)
