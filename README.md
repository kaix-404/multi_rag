# Multimodal Hybrid RAG Engine

**Dense-Sparse Hybrid Retrieval with Vision-Language Reranking for Heterogeneous Financial & Technical Documents**

Implementation of the capstone specification in
[`Multimodal Hybrid RAG Capstone Specification.pdf`](./Multimodal%20Hybrid%20RAG%20Capstone%20Specification.pdf).

Conventional RAG pipelines (text chunking + pure dense cosine search) collapse on
real enterprise documents — 10-K balance sheets, nested tables, multi-column
layouts, diagrams and time-series charts get truncated, OCR-garbled or stripped
of context. This project implements an end-to-end **multimodal hybrid retrieval
engine** that parses text, tables *and* figures, indexes them in a **dual
(BM25 + BGE-M3) index**, fuses the two rankings with **Reciprocal Rank Fusion
(k = 60)**, reranks the fused candidate set with a **cross-encoder**, and
answers questions with a grounded (optionally vision-capable) LLM — with an
interactive dual-pane demo and a RAGAS / Hit-Rate@5 benchmark against a naive
text-only dense baseline.

## Architecture

```
Input PDF ──► Dual-Stream Parser ──► Hybrid Index Store ──► Fusion & Reranking ──► Multimodal Synthesis
             (PyMuPDF+pdfplumber)    (BM25 sparse + dense)   (RRF k=60 + cross-enc)   (grounded answer + bbox)
```

| System layer | Component | Role |
|---|---|---|
| Ingestion & parsing | PyMuPDF + pdfplumber | text blocks with bboxes, tables → Markdown grids, figure crops → PNG |
| Visual encoding | GPT-4o-mini / fallback | synthetic descriptor per figure so charts are indexable |
| Sparse vectorizer | BM25 (rank_bm25) | exact-match recall for tickers, dates, account figures |
| Dense vectorizer | BAAI/bge-m3 (1024-d) | semantic recall; hashing-TF-IDF fallback when models unavailable |
| Hybrid store | Qdrant / local numpy | unified dense search (docker-compose provided) |
| Reranker | BAAI/bge-reranker-large | cross-encoder scoring over the fused top-N; heuristic blend fallback |
| Evaluation | RAGAS-style + Hit Rate@5 | context precision/recall, faithfulness, answer relevance |
| Frontend | FastAPI + Streamlit | dual-pane query/evidence workspace with bbox overlays |

Every indexed chunk stores provenance `{doc_id, page_number, chunk_type,
bounding_box, section}` enabling zero-latency highlighting in the demo viewer.

**Offline honesty:** the pipeline runs end-to-end with *zero* model downloads
and *zero* API keys (hashing embedder, heuristic reranker, extractive answer
fallback, caption-anchored figure descriptors). The active backend for every
run is reported in telemetry and in `data/results/benchmark.md`, so numbers are
never mislabeled as BGE-M3/RAGAS results when the fallbacks were active.

## Quickstart

```bash
poetry install --extras "models eval"     # or: python -m venv .venv && pip install -e .

# 1) corpus: 20 Form 10-K annual reports + 5 arXiv papers
python scripts/download_corpus.py

# 2) ingest: dual-stream parse -> chunks (512-token windows, 10% overlap)
mmrag ingest                     # or: python -m mmrag.cli ingest

# 3) index: embed + BM25 + vector store
mmrag index

# 4) ask
mmrag query "What was the percentage change in R&D expenditure between FY2022 and FY2023?"

# 5) benchmark (40 questions: 15 factual / 15 tabular / 10 visual)
mmrag eval
```

### Demo portal

```bash
docker compose up -d qdrant       # optional: Qdrant (auto-fallback to local store)
mmrag serve                       # FastAPI on :8000
mmrag ui                          # Streamlit dual-pane portal
```

Left pane: query + streamed answer + ranked passages with RRF/confidence
telemetry. Right pane: original page render with precise bounding-box overlays
for the paragraph/table/figure actually used.

The portal has two tabs: **Query & Synthesis** (above) and **Add documents** —
upload a PDF (≤ 30 MB / ≤ 80 pages) and the backend parses, chunks and embeds
just that document (`POST /documents/upload`), then hot-swaps it into the live
retriever so it is searchable immediately. Only the new file's chunks are
embedded; re-uploading the same file replaces it.

### Configuration

Copy `.env.example` → `.env`. Key knobs:

| Variable | Purpose |
|---|---|
| `QDRANT_URL` | vector store; empty ⇒ local numpy store |
| `MMRAG_DENSE_BACKEND` | `auto` (BGE-M3 if installed) or `hash` |
| `MMRAG_DENSE_MODEL` / `MMRAG_RERANKER_MODEL` | model ids; a vendored copy under `models/<name>` is preferred automatically |
| `MMRAG_EMBED_BATCH` | encode batch size (default 1 for low-RAM hosts; raise on bigger machines) |
| `MMRAG_ENABLE_RERANKER` | `0` disables the cross-encoder |
| `OPENAI_API_KEY` / `OPENAI_BASE_URL` / `LLM_MODEL` | generative + VLM backend; defaults to Puter AI's OpenAI-compatible endpoint (also works with OpenAI/Ollama/vLLM); empty key ⇒ offline extractive mode |

## Repository layout

```
src/mmrag/
├── config.py           # spec constants: RRF k=60, 512-token/10% windows, 1024-dim
├── ingestion/          # parser (PyMuPDF+pdfplumber), semantic window chunker
├── index/              # BM25, dense embedders (BGE-M3 / hashing), vector stores
├── retrieval/          # RRF fusion, cross-encoder reranker, hybrid retriever
├── rag/                # LLM/VLM client, figure descriptors, grounded synthesis
├── eval/               # 40-question dataset, metrics, naive baseline, harness
├── api/                # FastAPI backend
├── ui/                 # Streamlit dual-pane portal
├── pipeline.py         # ingest + index orchestration
└── cli.py              # mmrag ingest|index|query|eval|serve|ui
scripts/download_corpus.py   # 20 × 10-K + 5 × arXiv papers
docker-compose.yml           # Qdrant
notebooks/evaluation.ipynb   # reproduces benchmark tables + hypothesis check
data/results/benchmark.md    # evaluation output
```

## Evaluation

40 challenge questions (15 factual text, 15 tabular numerical, 10 visual/chart)
generated deterministically from the ingested corpus with gold evidence, so
**Hit Rate @5 / Context Precision / Context Recall** are measurable without an
LLM judge. Faithfulness and Answer Relevance use RAGAS when installed with an
LLM configured, otherwise a prompted judge, otherwise they are reported `n/a`.

The harness also runs the **naive baseline** (structure-blind 512-token
word-window chunking over plain text, pure dense cosine — no BM25/RRF/reranker/
figures) and checks the spec hypothesis:

> The hybrid multimodal model will achieve over 30% improvement in Context
> Recall on tabular/chart questions vs. the naive-chunking baseline.

Results land in `data/results/benchmark.md`, `metrics.json`, `per_question.jsonl`.

## Academic foundations

Lewis et al. (2020, NeurIPS) RAG · Robertson et al. (2009) BM25 ·
Faysse et al. (2024) ColPali (arXiv:2407.01449) · Es et al. (2023, EACL 2024)
RAGAS · Xiao et al. (2023) BGE-M3 · Li et al. (2023) bge-reranker.

## Deliverables checklist

- [x] Modular Python repository (Poetry-packaged `pyproject.toml`)
- [x] Dual-stream ingestion: text / Markdown tables / figure crops with bboxes
- [x] Dual index: BM25 sparse + dense vectors, Qdrant via docker-compose
- [x] RRF fusion (k=60) + cross-encoder reranking
- [x] VLM figure descriptors + grounded streaming answer synthesis
- [x] Evaluation notebook (`notebooks/evaluation.ipynb`) reproducing benchmark tables
- [x] Streamlit + FastAPI demo with dual-pane evidence inspection
- [ ] Academic report + slides (see `report/`)
