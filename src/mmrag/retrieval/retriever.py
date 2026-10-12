"""Hybrid retriever: BM25 ∥ BGE-M3 → RRF (k=60) → cross-encoder rerank."""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from ..config import Settings, load_settings
from ..index import BM25Index, Embedder, VectorStore, get_embedder, open_store
from ..ingestion.chunker import Chunk, load_chunks
from .fusion import rrf_fuse
from .rerank import Reranker, get_reranker


@dataclass
class RetrievalResult:
    chunk: Chunk
    dense_rank: int | None
    dense_score: float | None
    sparse_rank: int | None
    sparse_score: float | None
    rrf_score: float
    rerank_confidence: float
    final_score: float
    meta: dict = field(default_factory=dict)


@dataclass
class QueryTelemetry:
    latency_ms: float
    dense_ms: float
    sparse_ms: float
    fuse_ms: float
    rerank_ms: float
    embedder: str
    reranker: str
    store: str
    fused_pool: int


class HybridRetriever:
    def __init__(self, chunks: list[Chunk], embedder: Embedder, bm25: BM25Index,
                 store: VectorStore, reranker: Reranker, settings: Settings):
        self.chunks = {c.chunk_id: c for c in chunks}
        self.embedder = embedder
        self.bm25 = bm25
        self.store = store
        self.reranker = reranker
        self.settings = settings

    @classmethod
    def load(cls, settings: Settings | None = None) -> "HybridRetriever":
        settings = settings or load_settings()
        chunks = load_chunks(settings.chunks_path)
        bm25 = BM25Index.load(settings.bm25_path)
        embedder = get_embedder(settings)
        store = open_store(settings, embedder.dim)
        reranker = get_reranker(settings)
        return cls(chunks, embedder, bm25, store, reranker, settings)

    def retrieve(self, query: str, top_k: int = 5) -> tuple[list[RetrievalResult], QueryTelemetry]:
        s = self.settings
        t0 = time.perf_counter()

        qv = self.embedder.embed_one(query)
        t1 = time.perf_counter()
        dense_hits = self.store.search(qv, s.dense_top_k)
        t2 = time.perf_counter()
        sparse_hits = self.bm25.search(query, s.sparse_top_k)
        t3 = time.perf_counter()

        dense_rank = {cid: r for r, (cid, _) in enumerate(dense_hits, start=1)}
        dense_score = {cid: sc for cid, sc in dense_hits}
        sparse_rank = {cid: r for r, (cid, _) in enumerate(sparse_hits, start=1)}
        sparse_score = {cid: sc for cid, sc in sparse_hits}

        fused = rrf_fuse([list(dense_rank), list(sparse_rank)], k=s.rrf_k)
        ordered = sorted(fused.items(), key=lambda kv: kv[1], reverse=True)
        pool = ordered[: s.rerank_pool]
        t4 = time.perf_counter()

        pool_ids = [cid for cid, _ in pool]
        pool_texts = [self.chunks[cid].text if cid in self.chunks else "" for cid in pool_ids]
        aux = [{
            "dense": dense_score.get(cid, 0.0),
            "sparse": sparse_score.get(cid, 0.0),
        } for cid in pool_ids]
        confidences = self.reranker.rerank(query, pool_texts, aux)
        t5 = time.perf_counter()

        ranked = sorted(
            zip(pool_ids, pool, confidences),
            key=lambda x: x[2], reverse=True,
        )
        results: list[RetrievalResult] = []
        for cid, (_, rrf_score), conf in ranked[:top_k]:
            if cid not in self.chunks:
                continue
            results.append(RetrievalResult(
                chunk=self.chunks[cid],
                dense_rank=dense_rank.get(cid),
                dense_score=dense_score.get(cid),
                sparse_rank=sparse_rank.get(cid),
                sparse_score=sparse_score.get(cid),
                rrf_score=rrf_score,
                rerank_confidence=conf,
                final_score=conf,
            ))

        telemetry = QueryTelemetry(
            latency_ms=(t5 - t0) * 1000,
            dense_ms=(t2 - t1) * 1000,
            sparse_ms=(t3 - t2) * 1000,
            fuse_ms=(t4 - t3) * 1000,
            rerank_ms=(t5 - t4) * 1000,
            embedder=self.embedder.name,
            reranker=self.reranker.name,
            store=self.store.name,
            fused_pool=len(pool),
        )
        return results, telemetry

    def retrieve_ids(self, query: str, top_k: int = 5) -> list[str]:
        results, _ = self.retrieve(query, top_k)
        return [r.chunk.chunk_id for r in results]
