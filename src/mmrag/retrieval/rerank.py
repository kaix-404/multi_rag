"""Reranking engine.

Spec: ``BAAI/bge-reranker-large`` cross-encoder scoring full query-document
interactions across the fused candidate set. When the model stack is not
installed (no torch), a transparent heuristic reranker that combines the two
retrieval signals keeps the pipeline runnable; ``Reranker.name`` always
reports which one is active so benchmark numbers are never mislabeled.
"""

from __future__ import annotations

import math

from ..config import Settings


class Reranker:
    name = "base"
    kind = "none"

    def rerank(self, query: str, texts: list[str],
               aux: list[dict] | None = None) -> list[float]:
        """Return a confidence score per text (higher = more relevant)."""
        raise NotImplementedError


class HeuristicReranker(Reranker):
    """Min-max blended dense-similarity + BM25 score (offline fallback)."""

    name = "heuristic-rrf-blend"
    kind = "heuristic"

    def rerank(self, query: str, texts: list[str],
               aux: list[dict] | None = None) -> list[float]:
        if not texts:
            return []
        aux = aux or [{} for _ in texts]
        dense = [a.get("dense", 0.0) for a in aux]
        sparse = [a.get("sparse", 0.0) for a in aux]

        def norm(vals: list[float]) -> list[float]:
            lo, hi = min(vals), max(vals)
            if hi - lo < 1e-12:
                return [1.0] * len(vals)
            return [(v - lo) / (hi - lo) for v in vals]

        nd, ns = norm(dense), norm(sparse)
        return [0.5 * d + 0.5 * s for d, s in zip(nd, ns)]


class CrossEncoderReranker(Reranker):
    name = "cross-encoder"
    kind = "cross-encoder"

    def __init__(self, model_name: str):
        from sentence_transformers import CrossEncoder  # lazy: needs torch

        self.model = CrossEncoder(model_name, device="cpu")
        self.name = model_name

    def rerank(self, query: str, texts: list[str],
               aux: list[dict] | None = None) -> list[float]:
        if not texts:
            return []
        logits = self.model.predict([(query, t) for t in texts], show_progress_bar=False)
        return [1.0 / (1.0 + math.exp(-float(z))) for z in logits]


def get_reranker(settings: Settings) -> Reranker:
    if settings.enable_reranker:
        try:
            return CrossEncoderReranker(settings.reranker_model)
        except Exception as exc:
            print(f"[rerank] cross-encoder unavailable ({exc.__class__.__name__}: {exc}); "
                  f"using heuristic score-blend reranker")
    return HeuristicReranker()
