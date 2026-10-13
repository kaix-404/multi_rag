"""Hybrid vector store.

Two interchangeable backends:
- ``QdrantVectorStore``  — Qdrant server (docker-compose) or embedded local mode.
- ``LocalVectorStore``   — numpy cosine store persisted to disk (zero deps).

Both store dense vectors keyed by chunk_id; chunk payloads themselves live in
``chunks.jsonl`` as the single source of truth.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ..config import Settings


class VectorStore:
    name = "base"

    def upsert(self, chunk_ids: list[str], vectors: np.ndarray) -> None:
        raise NotImplementedError

    def reset(self) -> None:
        """Drop all vectors (called before a full re-index)."""
        raise NotImplementedError

    def search(self, vector: np.ndarray, top_k: int) -> list[tuple[str, float]]:
        raise NotImplementedError

    def count(self) -> int:
        raise NotImplementedError

    def close(self) -> None:
        pass


class LocalVectorStore(VectorStore):
    name = "local-numpy"

    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.mkdir(parents=True, exist_ok=True)
        self.ids: list[str] = []
        self.vectors: np.ndarray | None = None
        self._load()

    def _vec_path(self) -> Path:
        return self.path / "vectors.npy"

    def _ids_path(self) -> Path:
        return self.path / "vector_ids.json"

    def _load(self) -> None:
        if self._vec_path().exists() and self._ids_path().exists():
            self.vectors = np.load(self._vec_path())
            self.ids = json.loads(self._ids_path().read_text(encoding="utf-8"))
            if len(self.ids) != self.vectors.shape[0]:
                self.ids, self.vectors = [], None

    def upsert(self, chunk_ids: list[str], vectors: np.ndarray) -> None:
        vectors = np.asarray(vectors, dtype=np.float32)
        replace = {cid: i for i, cid in enumerate(self.ids)}
        idx_of, new_rows = [], []
        for cid, vec in zip(chunk_ids, vectors):
            if cid in replace:
                idx_of.append(replace[cid])
                self.vectors[replace[cid]] = vec
            else:
                replace[cid] = len(self.ids)
                idx_of.append(len(self.ids))
                self.ids.append(cid)
                new_rows.append(vec)
        if new_rows:
            block = np.vstack(new_rows)
            self.vectors = block if self.vectors is None else np.vstack([self.vectors, block])
        np.save(self._vec_path(), self.vectors)
        self._ids_path().write_text(json.dumps(self.ids), encoding="utf-8")

    def reset(self) -> None:
        self.ids: list[str] = []
        self.vectors: np.ndarray | None = None
        self._vec_path().unlink(missing_ok=True)
        self._ids_path().unlink(missing_ok=True)

    def search(self, vector: np.ndarray, top_k: int) -> list[tuple[str, float]]:
        if self.vectors is None or len(self.ids) == 0:
            return []
        v = np.asarray(vector, dtype=np.float32)
        vn = v / (np.linalg.norm(v) + 1e-12)
        M = self.vectors / (np.linalg.norm(self.vectors, axis=1, keepdims=True) + 1e-12)
        sims = M @ vn
        k = min(top_k, len(self.ids))
        top = np.argpartition(-sims, k - 1)[:k]
        top = top[np.argsort(-sims[top])]
        return [(self.ids[i], float(sims[i])) for i in top]

    def count(self) -> int:
        return len(self.ids)


class QdrantVectorStore(VectorStore):
    name = "qdrant"

    def __init__(self, url: str, collection: str, dim: int):
        from qdrant_client import QdrantClient, models

        self.client = QdrantClient(url=url, timeout=5)
        self.collection = collection
        self.dim = dim
        if not self.client.collection_exists(collection):
            self.client.create_collection(
                collection_name=collection,
                vectors_config=models.VectorParams(size=dim, distance=models.Distance.COSINE),
            )
        self._ids: list[str] = []

    def upsert(self, chunk_ids: list[str], vectors: np.ndarray) -> None:
        from qdrant_client import QdrantClient, models
        import hashlib

        vectors = np.asarray(vectors, dtype=np.float32)
        points = [
            # deterministic uint64 id so re-ingesting updates instead of duplicating
            models.PointStruct(
                id=int.from_bytes(hashlib.sha1(cid.encode()).digest()[:8], "little"),
                vector=vec.tolist(),
                payload={"chunk_id": cid},
            )
            for cid, vec in zip(chunk_ids, vectors)
        ]
        self.client.upsert(collection_name=self.collection, points=points)
        self._ids = list(chunk_ids)
        self._save_id_map()

    def _map_path(self) -> Path:
        return Path(self.collection + "_point_ids.json")

    def _save_id_map(self) -> None:
        self._map_path().write_text(json.dumps(self._ids), encoding="utf-8")

    def reset(self) -> None:
        from qdrant_client import models

        self.client.delete_collection(self.collection)
        self.client.create_collection(
            collection_name=self.collection,
            vectors_config=models.VectorParams(size=self.dim,
                                               distance=models.Distance.COSINE),
        )

    def search(self, vector: np.ndarray, top_k: int) -> list[tuple[str, float]]:
        res = self.client.query_points(
            collection_name=self.collection,
            query=np.asarray(vector, dtype=np.float32).tolist(),
            limit=top_k, with_payload=True,
        )
        out = []
        for p in res.points:
            cid = (p.payload or {}).get("chunk_id")
            if cid:
                out.append((cid, float(p.score)))
        return out

    def count(self) -> int:
        return self.client.get_collection(self.collection).points_count or 0


def open_store(settings: Settings, dim: int) -> VectorStore:
    """Prefer Qdrant when QDRANT_URL is reachable; fall back to local numpy."""
    if settings.qdrant_url:
        try:
            store = QdrantVectorStore(settings.qdrant_url, settings.qdrant_collection, dim)
            store.count()  # health check
            print(f"[store] using Qdrant at {settings.qdrant_url}")
            return store
        except Exception as exc:
            print(f"[store] Qdrant unavailable ({exc.__class__.__name__}: {exc}); "
                  f"falling back to local numpy store")
    return LocalVectorStore(settings.index_dir / "local_store")
