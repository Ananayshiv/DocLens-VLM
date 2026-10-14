# Selection rationale (2026-10-14)

| Research direction | Why interesting | Why not the starting point |
|---|---|---|
| **ColPali (ICLR 2025): visual page retrieval** | Late-interaction learned document/image search, simple token-level objective, ViDoRe benchmark, immediate practical application | **Selected**. Use pretrained compact ColSmol 500M on T4 and reimplement scoring; original ColPali 3B reproduction is not claimed |
| SmolVLM2 (2025) | Small image/video generation and general VQA | Good answerer, weaker end-to-end research question if used only as a chatbot |
| ColQwen2.5 multilingual 3B | Better potential multilingual visual retriever | More memory and slower on T4; option for a later controlled model comparison after lightweight baseline is stable |
| Large generative VLM LoRA | Direct model parameter fine-tuning | Risk of spending all the available time on training with no benchmark/evaluation, especially on 16GB T4 |

## Core math

For image page tokens `P ∈ R^{M×d}` and query tokens `Q ∈ R^{N×d}`, normalize feature vectors then score `s(Q,P) = Σ_i max_j Q_i · P_j` (ColBERT-style late interaction). Implemented in `doclens/late_interaction.py`. Max-over-patches preserves token-to-patch alignment more than averaging document tokens into one vector.

`doclens/adapter.py` is a NEW, optional low-rank shared residual transform trained via symmetric in-batch contrastive loss. If evaluated, use a separate training set, validation set and untouched test set, with no document template leakage.

## Result claims

- Executed: simple 3-page synthetic bilingual document corpus; five manual test questions; lexical baseline recalled correct page at rank 1 for all five. **This is a smoke test, not research evidence.**
- Pending: real benchmark ViDoRe page retrieval; ColSmol actual GPU inference; Indic-language test-set construction; VLM answer evaluation; low-rank adapter learning effectiveness.

Sources: Faysse et al., ICLR 2025; Hugging Face `vidore/colSmol-500M`; Hugging Face `HuggingFaceTB/SmolVLM-500M-Instruct`; Sarvam Vision 2.1 blog (public technical context only).
