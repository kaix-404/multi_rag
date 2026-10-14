"""Dense vectorizer.

Preferred backend: BAAI/bge-m3 via sentence-transformers (1024-dim, spec).
Fallback backend: deterministic hashing TF-IDF embedder with the same output
dimension — lets the full pipeline run on a machine with no model downloads
while keeping the hybrid architecture intact. The active backend is chosen
automatically and reported in telemetry so results are never mislabeled.
"""

from __future__ import annotations

import hashlib
import math
import os
import re
from pathlib import Path

import numpy as np

from ..config import Settings

_WORD_RE = re.compile(r"[A-Za-z0-9]+|[^\w\s]", re.UNICODE)


class Embedder:
    name: str = "base"
    dim: int = 0

    def embed(self, texts: list[str]) -> np.ndarray:  # (n, dim) float32, L2-normalized
        raise NotImplementedError

    def embed_one(self, text: str) -> np.ndarray:
        return self.embed([text])[0]


class HashingTfidfEmbedder(Embedder):
    """Offline fallback: hashed word 1-2 gram TF-IDF, L2-normalized."""

    def __init__(self, dim: int = 1024):
        self.dim = dim
        self.name = "hashing-tfidf"
        self._idf: dict[str, float] = {}
        self._fitted = False

    @staticmethod
    def _feats(text: str) -> list[str]:
        words = [w.lower() for w in _WORD_RE.findall(text)]
        bigrams = [f"{a}__{b}" for a, b in zip(words, words[1:]) if a.isalnum() or b.isalnum()]
        return words + bigrams

    @staticmethod
    def _bucket(feat: str) -> int:
        h = hashlib.md5(feat.encode("utf-8")).digest()
        return int.from_bytes(h[:8], "little")

    def fit(self, texts: list[str]) -> "HashingTfidfEmbedder":
        df: dict[str, int] = {}
        n = 0
        for t in texts:
            n += 1
            for f in set(self._feats(t)):
                df[f] = df.get(f, 0) + 1
        self._idf = {f: math.log((n + 1) / (c + 1)) + 1.0 for f, c in df.items()}
        self._fitted = True
        return self

    def embed(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, t in enumerate(texts):
            counts: dict[str, int] = {}
            for f in self._feats(t):
                counts[f] = counts.get(f, 0) + 1
            for f, c in counts.items():
                idx = self._bucket(f) % self.dim
                sign = 1.0 if (self._bucket(f) >> 60) & 1 else -1.0
                idf = self._idf.get(f, 1.6) if self._fitted else 1.0
                out[i, idx] += sign * (1.0 + math.log(c)) * idf
            norm = float(np.linalg.norm(out[i]))
            if norm > 0:
                out[i] /= norm
        return out


class BgeM3Embedder(Embedder):
    def __init__(self, model_name: str, dim: int = 1024):
        from sentence_transformers import SentenceTransformer  # lazy import

        self.model = SentenceTransformer(model_name, device="cpu")
        local = Path(model_name)
        self.name = local.name if local.is_dir() else model_name
        self.dim = dim

    def embed(self, texts: list[str]) -> np.ndarray:
        # batch_size kept small: the 2.3 GB fp32 model plus larger-batch SDPA
        # activations exhausted the commit limit on this 16 GB host (and
        # batches >1 have also segfaulted under that pressure), so batch=1
        # by default; raise via MMRAG_EMBED_BATCH on high-RAM hosts.
        batch = int(os.environ.get("MMRAG_EMBED_BATCH", "1"))
        step = 1000
        parts: list[np.ndarray] = []
        for i in range(0, len(texts), step):
            parts.append(self.model.encode(
                texts[i:i + step], batch_size=batch, convert_to_numpy=True,
                normalize_embeddings=True, show_progress_bar=False,
            ))
            print(f"[dense] encoded {min(i + step, len(texts))}/{len(texts)} "
                  f"chunks (batch={batch})", flush=True)
        vecs = np.vstack(parts) if parts else np.empty((0, self.dim), np.float32)
        return np.asarray(vecs, dtype=np.float32)


def _load_hash_state(emb: HashingTfidfEmbedder, settings: Settings) -> None:
    """Restore TF-IDF statistics saved at index time so queries embed consistently."""
    import pickle

    path = settings.index_dir / "embedder.pkl"
    if not path.exists():
        return
    try:
        with Path(path).open("rb") as fh:
            state = pickle.load(fh)
        if state.get("name") == emb.name:
            emb._idf = state.get("idf", {})
            emb._fitted = bool(state.get("fitted"))
    except Exception:
        pass


def get_embedder(settings: Settings) -> Embedder:
    """Pick BGE-M3 when available, otherwise the hashing fallback."""
    if settings.dense_backend != "hash":
        try:
            return BgeM3Embedder(settings.dense_model, settings.dense_dim)
        except Exception as exc:  # missing package, no network, bad model id...
            print(f"[dense] BGE-M3 unavailable ({exc.__class__.__name__}: {exc}); "
                  f"falling back to hashing TF-IDF embedder")
    emb = HashingTfidfEmbedder(settings.dense_dim)
    _load_hash_state(emb, settings)
    return emb


def cosine(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Row-wise cosine similarity of (n,d) against (d,) or (m,d)."""
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    an = a / (np.linalg.norm(a, axis=-1, keepdims=True) + 1e-12)
    bn = b / (np.linalg.norm(b, axis=-1, keepdims=True) + 1e-12)
    return an @ bn.T if b.ndim == 2 else an @ bn
