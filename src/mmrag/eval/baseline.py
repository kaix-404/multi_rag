"""Baseline system for comparison (spec: "LangChain naive-chunking baseline").

Text-only pipeline: plain text extraction (tables flattened to raw words,
figures dropped), fixed-size word windows (structure-blind chunking, the way
RecursiveCharacterTextSplitter behaves on plain text), pure dense cosine
retrieval. No BM25, no RRF, no cross-encoder, no VLM — exactly the naive
architecture the hybrid system is hypothesised to beat.
"""

from __future__ import annotations

import numpy as np

from ..config import Settings
from ..index.dense import Embedder, get_embedder
from ..ingestion.parser import ParsedItem


def naive_chunks(items: list[ParsedItem], window_words: int = 400,
                 overlap_frac: float = 0.10) -> list[dict]:
    """Structure-blind fixed word windows over plain text only."""
    texts = [it.text for it in items if it.chunk_type == "text"]
    flat = " ".join(texts)
    words = flat.split()
    if not words:
        return []
    step = int(window_words * (1 - overlap_frac))
    out, i, n = [], 0, len(words)
    while i < n:
        window = words[i:i + window_words]
        out.append({
            "chunk_id": f"naive:{len(out):05d}",
            "text": " ".join(window),
        })
        if i + window_words >= n:
            break
        i += step
    return out


class BaselineDenseRetriever:
    """Pure dense cosine top-k over naive chunks."""

    name = "baseline-dense-naive"

    def __init__(self, chunks: list[dict], embedder: Embedder):
        self.chunks = {c["chunk_id"]: c for c in chunks}
        self.embedder = embedder
        texts = [c["text"] for c in chunks]
        if hasattr(embedder, "fit"):
            embedder.fit(texts)  # hashing TF-IDF needs corpus statistics
        self.ids = [c["chunk_id"] for c in chunks]
        self.matrix = embedder.embed(texts)

    @classmethod
    def build(cls, items: list[ParsedItem], settings: Settings) -> "BaselineDenseRetriever":
        embedder = get_embedder(settings)
        return cls(naive_chunks(items), embedder)

    def retrieve_ids(self, query: str, top_k: int = 5) -> list[str]:
        q = self.embedder.embed_one(query)
        sims = self.matrix @ q
        k = min(top_k, len(self.ids))
        if k == 0:
            return []
        top = np.argpartition(-sims, k - 1)[:k]
        top = top[np.argsort(-sims[top])]
        return [self.ids[i] for i in top]
