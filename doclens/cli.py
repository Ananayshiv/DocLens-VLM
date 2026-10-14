from __future__ import annotations
import argparse
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description="DocLens-VLM: page-level visual document QA")
    sub = p.add_subparsers(dest="task", required=True)
    ingest_p = sub.add_parser("ingest", help="Render input PDFs/images into a searchable corpus")
    ingest_p.add_argument("paths", nargs="+", help="PDFs and document images")
    ingest_p.add_argument("--out", default="runs/corpus")
    ingest_p.add_argument("--ocr", action="store_true", help="Tesseract for scanned pages")
    ingest_p.add_argument("--ocr-languages", default="eng+hin")
    ingest_p.add_argument("--max-pages", type=int, default=30,
                          help="PDF page cap per input (default 30). Use 0 to index ALL pages; capped PDFs emit a persistent coverage warning.")
    visual_p = sub.add_parser("index-visual", help="Download pretrained ColSmol and create visual embeddings")
    visual_p.add_argument("--corpus", default="runs/corpus")
    search_p = sub.add_parser("search", help="Retrieve pages")
    search_p.add_argument("question")
    search_p.add_argument("--corpus", default="runs/corpus")
    search_p.add_argument("--mode", choices=["bm25", "visual"], default="bm25")
    search_p.add_argument("-k", type=int, default=3)
    ask_p = sub.add_parser("ask", help="Answer a question on the top-ranked page with SmolVLM")
    ask_p.add_argument("question")
    ask_p.add_argument("--corpus", default="runs/corpus")
    ask_p.add_argument("--mode", choices=["bm25", "visual"], default="visual")
    eval_p = sub.add_parser("evaluate", help="Evaluate against page-level relevance judgments")
    eval_p.add_argument("--queries", required=True)
    eval_p.add_argument("--corpus", default="runs/corpus")
    eval_p.add_argument("--mode", choices=["bm25", "visual"], default="bm25")
    eval_p.add_argument("-k", type=int, default=5)
    eval_p.add_argument("--out", default="evaluation.json")
    args = p.parse_args()
    if args.task == "ingest":
        from .corpus import ingest
        result = ingest(args.paths, args.out, ocr=args.ocr, ocr_languages=args.ocr_languages,
                        max_pages_per_file=args.max_pages)
        print(f"Corpus now contains {len(result)} unique pages in {args.out} (existing pages preserved)")
        from .corpus import ingestion_coverage_notice
        notice = ingestion_coverage_notice(args.out)
        if notice:
            print(notice)
    elif args.task == "index-visual":
        from .vision import ColSmolRetriever, build_index
        print(json.dumps(build_index(args.corpus, ColSmolRetriever()), indent=2))
    elif args.task in ("search", "ask"):
        if args.mode == "bm25":
            from .search import lexical_search
            result = lexical_search(args.question, args.corpus, k=3 if args.task == "ask" else args.k)
        else:
            from .vision import ColSmolRetriever, retrieve_visual
            result = retrieve_visual(args.question, args.corpus, ColSmolRetriever(),
                                     k=3 if args.task == "ask" else args.k)
        if args.task == "search":
            print(json.dumps(result, ensure_ascii=False, indent=2))
        elif result:
            from .generator import SmolVLMAnswerer, answer_on_page
            print(json.dumps(answer_on_page(args.question, result[0], args.corpus,
                                           SmolVLMAnswerer()), ensure_ascii=False, indent=2))
        else:
            print("No pages indexed")
    elif args.task == "evaluate":
        from .evaluate import run_evaluation
        result = run_evaluation(args.queries, args.corpus, args.mode, args.k, args.out)
        print(json.dumps(result["metrics"], indent=2))


if __name__ == "__main__":
    main()
