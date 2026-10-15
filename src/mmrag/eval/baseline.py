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
    """Structure-blind fixed word windows over plain text only.

    Streams through the token buffer so the whole corpus is never duplicated
    in memory (a 20-document 10-K corpus is enough to blow up a naive
    ``" ".join(...).split()`` implementation).
    """
    step = max(1, int(window_words * (1 - overlap_frac)))
    out: list[dict] = []
    buf: list[str] = []
    for it in items:
        if it.chunk_type != "text":
            continue
        buf.extend(it.text.split())
        while len(buf) >= window_words:
            out.append({
                "chunk_id": f"naive:{len(out):05d}",
                "text": " ".join(buf[:window_words]),
            })
            buf = buf[step:]
    if 50 <= len(buf):
        out.append({"chunk_id": f"naive:{len(out):05d}", "text": " ".join(buf)})
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
    def build(cls, items: list[ParsedItem], settings: Settings,
              embedder: Embedder | None = None) -> "BaselineDenseRetriever":
        # ``embedder`` may be the hybrid retriever's instance: sharing one
        # BGE-M3 keeps a second 2.3 GB copy out of memory (segfaults on
        # low-commit hosts) and makes the comparison strictly like-for-like.
        # Fit-based embedders (hashing TF-IDF) must NOT be shared: baseline
        # ``fit()`` would corrupt the hybrid store's corpus statistics.
        if embedder is None or hasattr(embedder, "fit"):
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
