"""Central configuration.

All knobs from the capstone spec live here: RRF constant k=60, 512-token
semantic windows with 10% overlap, 1024-dim dense vectors, model names, etc.
Values are read from environment variables / .env so the same code runs in
offline (fallback) mode and in full-model mode without edits.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
CORPUS_DIR = DATA_DIR / "corpus"
ASSETS_DIR = DATA_DIR / "assets"       # cropped figure PNGs
PAGES_DIR = DATA_DIR / "pages"         # rendered full page PNGs (for bbox overlays)
INDEX_DIR = DATA_DIR / "index"         # chunks.jsonl, bm25.pkl, vectors, store
RESULTS_DIR = DATA_DIR / "results"     # evaluation outputs


def _load_dotenv() -> None:
    """Minimal .env loader (no external dependency)."""
    env_path = PROJECT_ROOT / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default)


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() not in {"0", "false", "no", "off", ""}


@dataclass
class Settings:
    # --- retrieval -----------------------------------------------------------
    rrf_k: int = 60                      # spec: RRF constant k = 60
    chunk_tokens: int = 512              # spec: 512-token semantic windows
    chunk_overlap_frac: float = 0.10     # spec: 10% overlap
    dense_top_k: int = 100               # candidates pulled from each retriever
    sparse_top_k: int = 250              # widened: the consolidated-total line can sit
                                         # deep in BM25 (observed rank ~206) and must
                                         # still reach the reranker/generator
    rerank_pool: int = 60                # fused candidates routed to cross-encoder.
                                         # Post-enrichment the consolidated income-statement
                                         # chunk sits at fused rank ~41 (measured); reranking
                                         # 160 instead floods the saturated-score tie band
                                         # with prose and pushes it out of the top-8.
    answer_top_k: int = 8                # contexts handed to the generator.
                                         # Measured: the consolidated income-statement
                                         # chunk reranks at #6 (scores 2-8 are a saturated
                                         # 0.9997 tie), so top-5 cut excluded the one
                                         # context that labels the total explicitly.

    # --- models ---------------------------------------------------------------
    dense_backend: str = "auto"          # auto | hash
    dense_model: str = "BAAI/bge-m3"
    dense_dim: int = 1024                # spec: 1024-dimensional representations
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    enable_reranker: bool = True

    # --- stores ----------------------------------------------------------------
    qdrant_url: str = ""                 # empty = local numpy store
    qdrant_collection: str = "mmrag_chunks"

    # --- LLM / VLM (OpenAI chat-completions protocol; empty key => offline) ----
    # Default backend is Puter AI, whose REST endpoint speaks the OpenAI protocol.
    llm_base_url: str = "https://api.puter.com/puterai/openai/v1"
    llm_api_key: str = ""
    llm_model: str = "gpt-4o-mini"
    vlm_model: str = "gpt-4o-mini"

    # --- serving ----------------------------------------------------------------
    api_url: str = "http://127.0.0.1:8000"

    # --- paths (derived) ---------------------------------------------------------
    root: Path = field(default_factory=lambda: PROJECT_ROOT)
    data_dir: Path = field(default_factory=lambda: DATA_DIR)
    corpus_dir: Path = field(default_factory=lambda: CORPUS_DIR)
    assets_dir: Path = field(default_factory=lambda: ASSETS_DIR)
    pages_dir: Path = field(default_factory=lambda: PAGES_DIR)
    index_dir: Path = field(default_factory=lambda: INDEX_DIR)
    results_dir: Path = field(default_factory=lambda: RESULTS_DIR)

    @property
    def chunks_path(self) -> Path:
        return self.index_dir / "chunks.jsonl"

    @property
    def bm25_path(self) -> Path:
        return self.index_dir / "bm25.pkl"

    @property
    def llm_available(self) -> bool:
        return bool(self.llm_api_key)


def _model_ref(env_key: str, hub_id: str, folder: str) -> str:
    """Prefer a vendored copy under ``<root>/models/<folder>`` when present
    (hosts with flaky or blocked HuggingFace access), else the hub id."""
    override = _env(env_key, "")
    if override:
        return override
    local = PROJECT_ROOT / "models" / folder
    return str(local) if (local / "config.json").exists() else hub_id


def load_settings() -> Settings:
    _load_dotenv()
    return Settings(
        dense_backend=_env("MMRAG_DENSE_BACKEND", "auto"),
        dense_model=_model_ref("MMRAG_DENSE_MODEL", "BAAI/bge-m3", "bge-m3"),
        reranker_model=_model_ref("MMRAG_RERANKER_MODEL", "BAAI/bge-reranker-large",
                                  "bge-reranker-large"),
        enable_reranker=_env_bool("MMRAG_ENABLE_RERANKER", True),
        qdrant_url=_env("QDRANT_URL", ""),
        qdrant_collection=_env("QDRANT_COLLECTION", "mmrag_chunks"),
        llm_base_url=_env("OPENAI_BASE_URL", "https://api.puter.com/puterai/openai/v1"),
        llm_api_key=_env("OPENAI_API_KEY", ""),
        llm_model=_env("LLM_MODEL", "gpt-4o-mini"),
        vlm_model=_env("VLM_MODEL", "gpt-4o-mini"),
        api_url=_env("MMRAG_API_URL", "http://127.0.0.1:8000"),
    )


settings = load_settings()
