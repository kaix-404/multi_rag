"""BM25 sparse inverted index (lexical exact-match backbone).

Uses rank_bm25's BM25Okapi implementation; the fitted index is pickled so the
sparse retriever and the dense store always score the *same* chunk corpus.
"""

from __future__ import annotations

import pickle
import re
from pathlib import Path

from rank_bm25 import BM25Okapi

_TOKEN_RE = re.compile(r"[A-Za-z0-9]+(?:[.,%$/-][A-Za-z0-9]+)*|\$?\d+(?:[.,]\d+)*%?")
_STOP = {
    "the", "a", "an", "and", "or", "of", "to", "in", "for", "on", "by", "with",
    "is", "are", "was", "were", "be", "been", "as", "at", "it", "its", "that",
    "this", "these", "those", "from", "we", "our", "their", "they", "you", "your",
}


def tokenize(text: str) -> list[str]:
    """Lowercased tokenizer that keeps financial identifiers intact
    (tickers, numbers, percentages, dates like 2023-12-31)."""
    toks = []
    for m in _TOKEN_RE.findall(text.lower()):
        if m in _STOP:
            continue
        toks.append(m)
        # split hyphen/slash compounds so 'long-term' matches 'long' + 'term'
        if any(c in m for c in "-/"):
            toks.extend(p for p in re.split(r"[-/]", m) if p and p not in _STOP)
    return toks


class BM25Index:
    def __init__(self, chunk_ids: list[str], bm25: BM25Okapi, tokenized_corpus: list[list[str]]):
        self.chunk_ids = chunk_ids
        self.bm25 = bm25
        self.tokenized_corpus = tokenized_corpus

    @classmethod
    def build(cls, chunk_ids: list[str], texts: list[str]) -> "BM25Index":
        tokenized = [tokenize(t) for t in texts]
        tokenized = [t if t else ["<empty>"] for t in tokenized]
        return cls(chunk_ids, BM25Okapi(tokenized), tokenized)

    def search(self, query: str, top_k: int) -> list[tuple[str, float]]:
        scores = self.bm25.get_scores(tokenize(query))
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        out = []
        for i in ranked[:top_k]:
            if scores[i] > 0:
                out.append((self.chunk_ids[i], float(scores[i])))
        return out

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as fh:
            pickle.dump(self, fh)

    @classmethod
    def load(cls, path: Path) -> "BM25Index":
        with Path(path).open("rb") as fh:
            obj = pickle.load(fh)
        if not isinstance(obj, cls):
            raise RuntimeError(f"BM25Index.load: expected BM25Index, got {type(obj).__name__}")
        return obj
