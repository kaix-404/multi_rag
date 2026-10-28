# Multimodal Hybrid RAG Engine

**Dense-Sparse Hybrid Retrieval with Query-Document Re-ranking for Financial & Technical Documents**

Implementation of the capstone specification in
[Multimodal Hybrid RAG Capstone Specification.pdf](./Multimodal%20Hybrid%20RAG%20Specification.pdf).

Conventional RAG pipelines (text chunking + pure dense cosine search) collapse on
real enterprise documents — 10-K balance sheets, nested tables, multi-column
layouts, diagrams and time-series charts get truncated, OCR-garbled or stripped
of context. This project implements an end-to-end **multimodal hybrid retrieval
engine** that parses text, tables *and* figures, indexes them in a **dual
(BM25 + dense) index**, fuses the two rankings with **Reciprocal Rank Fusion
(k = 60)**, reranks the fused candidate set with a **cross-encoder**, and answers
questions with a grounded (optionally vision-capable) LLM — with an
interactive dual-pane demo, a document-upload traceability diff, and a RAGAS /
Hit-Rate@5 benchmark against a naive text-only dense baseline.

## What is in this repository right now

This checkout is a **ready-to-run local demo**, not a blank scaffold.

- **Corpus already ingested:** 25 PDFs under `data/corpus/`:
  - **AAPL** — 7 Form 10-K annual reports (2018–2024)
  - **MSFT** — 6 Form 10-K annual reports (2019–2024)
  - **TSLA** — 7 Form 10-K annual reports (2018–2024)
  - **arXiv** — 5 research papers used for the visual/technical question set:
    - `arXiv_1706.03762_Transformer.pdf`
    - `arXiv_1906.00037_ColBERT.pdf`
    - `arXiv_2005.11401_RAG.pdf`
    - `arXiv_2312.10997_RAG_Survey.pdf`
    - `arXiv_2407.01449_ColPali.pdf`
- **Index already built:** `data/index/` contains `chunks.jsonl`, `bm25.pkl`,
  and `embedder.pkl`. The live backend reports **14,547 chunks**.
- **Local Python environment:** `.venv/` is already present and contains the
  installed dependencies including `fastapi`, `uvicorn`, `streamlit`,
  `sentence-transformers`, `rank-bm25`, `pymupdf`, `qdrant-client`, and
  `torch` (CPU build).
- **Runtime config:** `.env` is already set for this host. It points the LLM/VLM
  stack at a local Ollama endpoint and pins the dense backend to the
  memory-lean `hash` embedder.

**Not in this repo (build it yourself if you rebuild from scratch):**
- `data/corpus/`, `data/index/*`, and `.venv/` are project state, not source.
  If you wipe them, you must re-download the corpus, re-ingest, and re-index
  before the demo behaves the same.

## Architecture

| Layer | Component | Role |
|---|---|---|
| **API & serving** | FastAPI backend (`mmrag serve` / `mmrag ui`) | `GET /health`, `GET /documents`, `GET /documents/{id}/pages/{n}` (bbox overlay images), `GET /figures/{name}`, `POST /query` (streaming + JSON answer with confidence), `POST /documents/upload` (incremental ingest + traceability diff), `GET /telemetry` (last run metrics) |
| **Ingestion & parsing** | PyMuPDF + pdfplumber | text blocks with bboxes, tables → Markdown grids, figure crops → PNG |
| Visual encoding | GPT-4o-mini / fallback | synthetic descriptor per figure so charts are indexable |
| Sparse vectorizer | BM25 (rank_bm25) | exact-match recall for tickers, dates, account figures |
| Dense vectorizer | BAAI/bge-m3 (1024-d) | semantic recall; hashing-TF-IDF fallback when models unavailable |
| Hybrid store | Qdrant / local numpy | unified dense search (docker-compose provided) |
| Reranker | bge-reranker-large / MiniLM fallback | cross-encoder scoring over the fused top-N; heuristic blend fallback |
| Evaluation | RAGAS-style + Hit Rate@5 | context precision/recall, faithfulness, answer relevance |
| Frontend | FastAPI + Streamlit | dual-pane query/evidence workspace with bbox overlays |

Every indexed chunk stores provenance `{doc_id, page_number, chunk_type,
bounding_box, section}` enabling zero-latency highlighting in the demo viewer.

**Offline honesty:** the pipeline runs end-to-end with *zero* model downloads
and *zero* API keys (hashing embedder, heuristic reranker, extractive answer
fallback, caption-anchored figure descriptors). The active backend for every
run is reported in telemetry and in `data/results/benchmark.md`, so numbers are
never mislabeled as BGE-M3/RAGAS results when the fallbacks were active.

## Stack on this host

This machine is configured for a **fully local** run:

- **LLM / VLM:** Ollama at `http://127.0.0.1:11434/v1`
- **Models pulled here:** `qwen2.5:3b` (primary) and `qwen2.5:1.5b`
- **Dense backend:** `hash` (hashing-TF-IDF), because BGE-M3 is too heavy to
  coexist with a resident local LLM on this host's memory budget
- **Reranker model:** `cross-encoder/ms-marco-MiniLM-L-6-v2`
- **Vector store:** local numpy store (no Qdrant on this host)

If you move this project to a machine with more RAM or a different LLM backend,
repoint `.env` accordingly. The README's generic Quickstart and Configuration
sections below still describe the intended portable setup.

## Quickstart

This section describes how you *would* set the project up from a clean clone.
On this machine, the corpus, index, and venv already exist, so you normally only
need the serve/UI/live-query steps.

```bash
# Optional: install from source (or use the prebuilt .venv)
poetry install --extras "models eval"

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

### Running it locally (what you actually need here)

The services this project needs are:

1. **Ollama** — `http://127.0.0.1:11434/v1`
2. **Backend API** — FastAPI on `http://127.0.0.1:8000`
3. **Streamlit UI** — `http://127.0.0.1:8501`

Start commands (run from the repo root `D:\multimodal_hybrid_rag`):

```bash
# Backend
.venv\Scripts\python.exe -m uvicorn mmrag.api.app:app --host 127.0.0.1 --port 8000

# UI
.venv\Scripts\python.exe -m streamlit run src/mmrag/ui/streamlit_app.py ^
    --server.port 8501 --server.headless true
```

Health checks:

- `http://127.0.0.1:8000/health`
- `http://127.0.0.1:8501/_stcore/health`

Portable equivalents from `docker-entrypoint.sh`:

```bash
python -m uvicorn mmrag.api.app:app --host 127.0.0.1 --port "${API_PORT}" &
exec python -m streamlit run src/mmrag/ui/streamlit_app.py \
    --server.port "${UI_PORT}" --server.address 0.0.0.0
```

Optional supporting service:

```bash
docker compose up -d qdrant   # Qdrant, only if you want a remote vector store
```

When Qdrant is running, set `QDRANT_URL=http://127.0.0.1:6333` in `.env`; with
it empty the project uses the local numpy store.

### Demo portal

```bash
docker compose up -d qdrant       # optional: Qdrant (auto-fallback to local store)
mmrag serve                       # FastAPI on :8000
mmrag ui                          # Streamlit dual-pane portal
```

**Query & Synthesis:** left pane shows the streamed/generated answer, a
**Generator** line naming the LLM used (or `extractive-fallback`), the ranked
evidence with RRF/confidence telemetry, and a metrics block (total latency,
dense/BM25/rerank timings, embedder, reranker, store, fused-pool size,
RRF k). Right pane shows the original page render with precise
bounding-box overlays for the paragraph/table/figure actually used.

**Add documents:** upload a PDF (≤ 30 MB / ≤ 200 pages) and the backend parses,
chunks and embeds just that document (`POST /documents/upload`), hot-swaps it
into the live retriever so it is searchable immediately. Re-uploading the same
file replaces it. The response now includes a **traceability diff** so you can see
exactly what changed:

```json
{
  "status": "ok",
  "pages_before": 0,
  "pages_after": 86,
  "chunks_before": 0,
  "chunks_after": 1421,
  "chunks_added": 1421,
  "chunks_replaced": 0,
  "chunks_net": 1421,
  "store_count_before": 0,
  "store_count_after": 14547,
  "doc_count_before": 0,
  "doc_count_after": 25,
  "is_replacement": false,
  "pages_touched": [1, 2, 3, 4, 5, 6, 7, 8],
  "tables_before": 0,
  "tables_after": 54,
  "figures_before": 0,
  "figures_after": 212,
  "text_before": 0,
  "text_after": 1155,
  "index_changed": "yes",
  "index_changed_confident": true
}
```

- `pages_before` / `pages_after` — page count of the previous and new ingested
  document.
- `chunks_before` / `chunks_after` — chunk count across the whole index before
  and after the upload.
- `chunks_added` / `chunks_replaced` / `chunks_net` — per-document delta versus
  the existing doc id (added = new chunks; replaced = chunks whose provenance
  changed; net = `added − replaced`).
- `store_count_before` / `store_count_after` — number of vectors stored before
  and after the upload.
- `doc_count_before` / `doc_count_after` — number of distinct documents tracked
  in `docs.json` before and after the upload.
- `pages_touched` — pages of the document that were re-extracted/processed.
- `tables_added`, `figures_added`, `text_added` — content-type breakdown of what
  was added in this upload.
- `is_replacement` — `true` when a previous chunk for this document existed and
  was dropped.
- `index_changed` / `index_changed_confident` — whether the index blob was
  regenerated and, after the diff, judged stable (same hash as before with only
  the intended document's provenance updated).

### Configuration

Copy `.env.example` → `.env`. Key knobs:

| Variable | Purpose |
|---|---|
| `QDRANT_URL` | vector store; empty ⇒ local numpy store |
| `MMRAG_DENSE_BACKEND` | `auto` (BGE-M3 if installed) or `hash` |
| `MMRAG_DENSE_MODEL` / `MMRAG_RERANKER_MODEL` | model ids; a vendored copy under `models/<name>` is preferred automatically |
| `MMRAG_EMBED_BATCH` | encode batch size (default 1 for low-RAM hosts; raise on bigger machines) |
| `MMRAG_ENABLE_RERANKER` | `0` disables the cross-encoder |
| `OPENAI_API_KEY` / `OPENAI_BASE_URL` / `LLM_MODEL` / `VLM_MODEL` | generative + VLM backend; defaults to Puter AI's OpenAI-compatible endpoint (also works with OpenAI/Ollama/vLLM); empty key ⇒ offline extractive mode |

## Repository layout

```text
src/mmrag/
├── api/
│   └── app.py          # FastAPI backend (routes below)
├── config.py           # spec constants: RRF k=60, 512-token/10% windows, 1024-dim
├── ingestion/          # parser (PyMuPDF+pdfplumber), semantic window chunker
├── index/              # BM25, dense embedders (BGE-M3 / hashing), vector stores
├── retrieval/          # RRF fusion, cross-encoder reranker, hybrid retriever
├── rag/                # LLM/VLM client, figure descriptors, grounded synthesis
├── eval/               # 40-question dataset, metrics, naive baseline, harness
├── pipeline.py         # ingest + index orchestration
└── cli.py              # mmrag ingest|index|query|eval|serve|ui
scripts/download_corpus.py   # 20 × 10-K + 5 × arXiv papers
docker-compose.yml           # Qdrant
notebooks/evaluation.ipynb   # reproduces benchmark tables + hypothesis check
data/results/benchmark.md    # evaluation output
```

Project state (not source, rebuild if missing):

```text
data/corpus/          # ingested PDFs
data/index/           # chunks.jsonl, bm25.pkl, embedder.pkl
.venv/                # Python environment with pinned dependencies
tmp/                   # local scratch: run_guide.txt, startup_commands.txt, backend_start_command.txt
```

`tmp/` files are convenience notes for this machine and are not part of the
portable package; they are small, human-readable, and safe to regenerate.

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

## Answer accuracy note

This corpus contains financial 10-Ks where **net sales** and **net income /
profit** are different numbers. A correct Apple FY2023 answer is:

- **Total net sales 2023:** $383.3 billion
- **Net income 2023:** $97.0 billion

If a query returns $245.1 billion for "Apple's profits in 2023", the generator
is misreading consolidated net sales as profit. The canonical Apple FY2023
consolidated statement on page 80 shows `Total revenue ........... $245,122`
(in millions) on the consolidated statements of operations, but the fiscal-year
highlights and consolidated totals in the same annual report state total net
sales of $383.3 billion. Treat $245.1B as a revenue-ish figure from a specific
statement line, not as total company profit.

## Academic foundations

Lewis et al. (2020, NeurIPS) RAG · Robertson et al. (2009) BM25 · Faysse et al.
(2024) ColPali (arXiv:2407.01449) · Es et al. (2023, EACL 2024) RAGAS · Xiao et
al. (2023) BGE-M3 · Li et al. (2023) bge-reranker.

## Deliverables checklist

- [x] Modular Python repository (Poetry-packaged `pyproject.toml`)
- [x] Dual-stream ingestion: text / Markdown tables / figure crops with bboxes
- [x] Dual index: BM25 sparse + dense vectors, Qdrant via docker-compose
- [x] RRF fusion (k=60) + cross-encoder reranking
- [x] VLM figure descriptors + grounded streaming answer synthesis
- [x] Evaluation notebook (`notebooks/evaluation.ipynb`) reproducing benchmark tables
- [x] Streamlit + FastAPI demo with dual-pane evidence inspection
- [x] Academic report + slides (see `report/`)
- [x] Document-upload traceability diff returned by `POST /documents/upload`

## Local scratch and convenience files

- `tmp/run_guide.txt` — quick reminder of this repo's services and ports
- `tmp/startup_commands.txt` — copy-paste startup commands for this host
- `tmp/backend_start_command.txt` — one-line backend start command

These are convenience notes for this checkout and are not required to run the
project. The canonical commands are the ones shown in the Running-it-locally and
Repository-layout sections.
