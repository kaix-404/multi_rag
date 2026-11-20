# Multimodal Hybrid RAG Engine — Project Summary

## Overview
A complete end-to-end retrieval-augmented generation system for financial and technical documents, implementing **dense-sparse hybrid retrieval with query-document re-ranking and multimodal figure grounding**. The system processes 25 PDFs (20 Form 10-K annual reports + 5 arXiv papers), indexes them in a dual BM25+dense vector store, and supports question answering with grounded citations.

## Architecture (3-Tier Pipeline)

### 1. Ingestion & Parsing (`src/mmrag/ingestion/`)
- **Dual-stream PDF parsing**:
  - **PyMuPDF stream**: Extracts text blocks with bounding boxes, raster figure crops, and vector chart clustering
  - **pdfplumber stream**: Table detection via pdfplumber → Markdown grid conversion with data-coverage validation
- **Provenance tracking**: Every extracted unit carries `{doc_id, page_number, chunk_type, bounding_box, section}`
- **Figure descriptors**: VLM-generated (GPT-4o-mini/llava) or deterministic caption-anchored fallback

### 2. Chunking (`src/mmrag/ingestion/chunker.py`)
- **Semantic sentence-window chunking**: 512-token windows with 10% overlap for prose
- **Atomic preservation**: Tables and figures stay intact (never split across chunks)
- **Document-contextualized headers**: `[LABEL fiscal year YYYY annual report | doc_id]` prepended to every chunk
- **Provenance**: Each chunk stores `{doc_id, page_number, chunk_type, bounding_box, section}`

### 3. Indexing & Retrieval (`src/mmrag/index/`, `src/mmrag/retrieval/`)
- **Sparse index**: BM25 (`rank_bm25`) with financial-token-aware tokenizer (keeps tickers, numbers, percentages)
- **Dense index**: BGE-M3 (1024-dim) or hashing TF-IDF fallback (same dimension, zero model dependencies)
- **Vector store**: Local numpy store (primary) or Qdrant (docker-compose optional)
- **RRF fusion**: Reciprocal Rank Fusion with k=60 over dense + sparse rankings
- **Cross-encoder reranking**: 
  - Primary: `cross-encoder/ms-marco-MiniLM-L-6-v2` (90 MB, memory-safe)
  - Fallback: Heuristic blend of min-max normalized dense + BM25 scores

### 4. Answer Synthesis (`src/mmrag/rag/`)
- **LLM path**: Strict "answer only from numbered contexts [n]" prompting with financial number rules
- **Extractive fallback**: When no LLM API key, quotes top-ranked sentences with `[n]` provenance
- **Figure grounding**: Figure captions/descriptors prepended to chunk text for lexical retrievability

### 5. Evaluation (`src/mmrag/eval/`)
- **40-question suite**: 15 factual text + 15 tabular numerical + 10 visual/chart
- **Gold evidence**: Every question has ground-truth chunk ids from the indexed corpus
- **Metrics**: Hit Rate@5, Context Recall, Context Precision, Faithfulness, Answer Relevance
- **Spec hypothesis**: >30% Context Recall gain on tabular/chart questions vs naive-dense baseline
- **Baseline**: Structure-blind 512-word windows over plain text, pure dense cosine (no BM25/RRF/reranker/figures)

## Key Configuration (`src/mmrag/config.py`)

| Variable | Purpose | Default |
|---|---|---|
| `rrf_k` | RRF constant | 60 |
| `chunk_tokens` | Semantic window size | 512 |
| `chunk_overlap_frac` | Overlap fraction | 0.10 |
| `dense_backend` | `auto` or `hash` | hash (memory-lean) |
| `dense_model` | Embedder model | BAAI/bge-m3 |
| `reranker_model` | Reranker model | cross-encoder/ms-marco-MiniLM-L-6-v2 |
| `enable_reranker` | Enable cross-encoder | True |
| `sparse_top_k` | BM25 candidate pool | 250 |
| `dense_top_k` | Dense candidate pool | 100 |
| `rerank_pool` | Candidates routed to reranker | 60 |
| `answer_top_k` | Contexts handed to generator | 8 |

## Runtime Configuration (`.env`)

- **Dense embedder**: `hashing TF-IDF` (BGE-M3 excluded due to ~2 GB RAM on this host)
- **Reranker**: `cross-encoder/ms-marco-MiniLM-L-6-v2` (small, reliable fallback)
- **LLM/VLM**: Ollama `qwen2.5:3b` (primary) / `qwen2.5:1.5b` (verified financial QA)
- **LLM base**: `http://127.0.0.1:11434/v1` (Ollama)
- **Vector store**: Local numpy (Qdrant optional via `QDRANT_URL`)
- **APIs**: FastAPI on `:8000`, Streamlit on `:8501`