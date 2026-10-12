"""Reciprocal Rank Fusion (RRF).

Spec formula:  RRF_Score(d) = Σ_m 1 / (k + rank_m(d)),  k = 60

Rank-based fusion is scale-free, so BM25 term-frequency scores and dense
cosine similarities can be combined without calibration.
"""

from __future__ import annotations

DEFAULT_K = 60


def rrf_fuse(rankings: list[list[str]], k: int = DEFAULT_K) -> dict[str, float]:
    """Fuse several rankings (best-first lists of chunk ids) with RRF.

    Returns {chunk_id: rrf_score} sorted best-first when consumed by callers.
    """
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, cid in enumerate(ranking, start=1):  # 1-based ranks
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank)
    return scores


def rrf_explain(sparse_rank: dict[str, int], dense_rank: dict[str, int],
                k: int = DEFAULT_K) -> dict[str, dict]:
    """Per-document fusion audit trail: contributions of each retriever."""
    ids = set(sparse_rank) | set(dense_rank)
    out = {}
    for cid in ids:
        s = sparse_rank.get(cid)
        d = dense_rank.get(cid)
        out[cid] = {
            "sparse_rank": s,
            "dense_rank": d,
            "sparse_contrib": 0.0 if s is None else 1.0 / (k + s),
            "dense_contrib": 0.0 if d is None else 1.0 / (k + d),
            "rrf_score": (0.0 if s is None else 1.0 / (k + s))
                         + (0.0 if d is None else 1.0 / (k + d)),
        }
    return out
