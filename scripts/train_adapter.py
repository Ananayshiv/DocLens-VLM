"""Optional small supervised domain adaptation of frozen ColSmol token embeddings.

This does NOT train a vision-language model end to end. A shared low-rank
residual token transform is learned from explicitly supplied TRAIN annotations.
Do not use `vidore/docvqa_test_subsampled` queries to train, tune or select it.

Example after obtaining separate annotated training/validation corpora:
  python scripts/train_adapter.py --corpus runs/your_train_corpus \
    --train runs/train_queries.json --valid runs/valid_queries.json
"""
from __future__ import annotations
import argparse
import json
import random
from pathlib import Path
import numpy as np
import torch
from doclens.adapter import LowRankTokenAdapter, inbatch_adapter_step, adapter_scores
from doclens.corpus import load_corpus
from doclens.evaluate import evaluate_queries
from doclens.late_interaction import maxsim


def load_pairs(filename):
    pairs = json.loads(Path(filename).read_text(encoding="utf-8"))
    if not isinstance(pairs, list) or len(pairs) < 2:
        raise ValueError("At least two training or validation queries required")
    return pairs


def validate_split(train, valid):
    tq = {x["query"].strip().casefold() for x in train}
    vq = {x["query"].strip().casefold() for x in valid}
    if tq & vq:
        raise ValueError("Query overlap across train and validation")
    train_pages = {p for x in train for p in x["relevant_page_ids"]}
    valid_pages = {p for x in valid for p in x["relevant_page_ids"]}
    if train_pages & valid_pages:
        raise ValueError("Positive page overlap across train and validation; split by document")


def prepare_embeddings(corpus_dir, samples, model):
    from doclens.vision import ColSmolRetriever
    backend = ColSmolRetriever(model_id=model)
    query_vectors = {}
    for i, row in enumerate(samples):
        q = row["query"]
        if q not in query_vectors:
            query_vectors[q] = backend.encode_query(q)
    # Explicit model unload between embedding prep and adapter training.
    del backend
    if torch.cuda.is_available():
        import gc
        gc.collect()
        torch.cuda.empty_cache()
    folder = Path(corpus_dir) / "visual"
    page_embeddings = {}
    for row in load_corpus(corpus_dir):
        src = folder / f"{row['page_id']}.npz"
        if not src.exists():
            raise FileNotFoundError(f"Visual index missing {src}; first run: doclens index-visual")
        with np.load(src, allow_pickle=False) as f:
            page_embeddings[row["page_id"]] = torch.tensor(f["embeddings"].astype("float32"))
    return query_vectors, page_embeddings


def retrieval_accuracy(rows, qemb, pemb, adapter=None, k=1):
    items = sorted(pemb)
    def search(q, topk):
        q_vec = qemb[q]
        if adapter is not None:
            q_vec = adapter(q_vec)
        scores = {}
        for page_id in items:
            page = pemb[page_id]
            if adapter is not None:
                page = adapter(page)
            scores[page_id] = float(maxsim(q_vec, page).item())
        return [{"page_id": key} for key in sorted(scores, key=scores.get, reverse=True)[:topk]]
    with torch.no_grad():
        return evaluate_queries(rows, search, k=k)["metrics"][f"recall@{k}"]


def train(corpus, train_file, valid_file, steps=120, rank=8, seed=42,
          model="vidore/colSmol-500M", out="runs/adapter.pt"):
    train_rows = load_pairs(train_file)
    valid_rows = load_pairs(valid_file)
    validate_split(train_rows, valid_rows)
    random.seed(seed)
    torch.manual_seed(seed)
    qemb, pemb = prepare_embeddings(corpus, train_rows + valid_rows, model)
    for row in train_rows + valid_rows:
        unknown = set(row["relevant_page_ids"]) - set(pemb)
        if unknown: raise ValueError(f"Relevance judgments contain missing page IDs: {unknown}")
    dim = next(iter(pemb.values())).shape[-1]
    adapter = LowRankTokenAdapter(dim=dim, rank=rank)
    optimizer = torch.optim.AdamW(adapter.parameters(), lr=1e-3, weight_decay=.01)
    baseline_val = retrieval_accuracy(valid_rows, qemb, pemb)
    losses = []
    # In-batch negatives and positives. Only one positive page per query used here.
    for step in range(steps):
        sample = random.sample(train_rows, min(4, len(train_rows)))
        # repeated positives in same batch break the diagonal-negative assumption
        positive = [r['relevant_page_ids'][0] for r in sample]
        if len(set(positive)) != len(positive):
            continue
        loss = inbatch_adapter_step(adapter, optimizer,
                                    [qemb[r['query']] for r in sample],
                                    [pemb[p] for p in positive])
        losses.append(loss)
    trained_val = retrieval_accuracy(valid_rows, qemb, pemb, adapter)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": adapter.state_dict(), "dim": dim, "rank": rank,
                "pretrained_model_id": model}, out)
    result = {"train_pairs": len(train_rows), "validation_pairs": len(valid_rows),
              "requested_steps": steps, "executed_steps": len(losses),
              "last_train_loss": losses[-1] if losses else None,
              "validation_recall_at_1_before": baseline_val,
              "validation_recall_at_1_after": trained_val,
              "warning": "Not a VLM fine-tune. Frozen embeddings and learned low-rank projection. No test-set optimization."}
    Path(out + '.json').write_text(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--corpus", required=True)
    p.add_argument("--train", required=True)
    p.add_argument("--valid", required=True)
    p.add_argument("--steps", type=int, default=120)
    p.add_argument("--rank", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--model", default="vidore/colSmol-500M")
    p.add_argument("--out", default="runs/adapter.pt")
    a = p.parse_args()
    print(json.dumps(train(a.corpus, a.train, a.valid, a.steps, a.rank, a.seed, a.model, a.out), indent=2))
