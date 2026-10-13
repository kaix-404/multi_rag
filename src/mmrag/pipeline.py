"""Pipeline orchestration: ingest (parse+chunk+VLM) and index (embed+store)."""

from __future__ import annotations

import json
import pickle
import time
from pathlib import Path

from .config import Settings, load_settings
from .eval.harness_io import save_parsed_items
from .index import BM25Index, get_embedder, open_store
from .ingestion import chunk_documents, parse_corpus
from .ingestion.chunker import load_chunks, save_chunks
from .rag import LLMClient, describe_figures


def cmd_ingest(settings: Settings | None = None, limit: int | None = None,
               render_dpi: int = 144, from_items: bool = False) -> dict:
    settings = settings or load_settings()
    t0 = time.perf_counter()

    if from_items and (settings.index_dir / "parsed_items.jsonl").exists():
        print("[ingest] re-chunking from saved parsed items (--from-items)")
        from .eval.harness_io import load_docs_meta, load_parsed_items
        metas = load_docs_meta(settings)
        items = load_parsed_items(settings)
    else:
        pdfs = sorted(settings.corpus_dir.glob("*.pdf"))
        if limit:
            pdfs = pdfs[:limit]
        if not pdfs:
            raise SystemExit(f"No PDFs found in {settings.corpus_dir}. "
                             f"Run `python scripts/download_corpus.py` first.")

        metas, items = [], []
        from .ingestion.parser import parse_pdf
        for pdf in pdfs:
            print(f"[ingest] parsing {pdf.name} ...")
            meta, it = parse_pdf(pdf, settings.assets_dir, settings.pages_dir,
                                 render_dpi=render_dpi)
            metas.append(meta)
            items.extend(it)

        settings.index_dir.mkdir(parents=True, exist_ok=True)
        (settings.index_dir / "docs.json").write_text(
            json.dumps(metas, indent=2, ensure_ascii=False), encoding="utf-8")
        save_parsed_items(items, settings.index_dir / "parsed_items.jsonl")

    chunks = chunk_documents(items, settings)
    llm = LLMClient(settings)
    vlm_stats = describe_figures(chunks, llm)
    save_chunks(chunks, settings.chunks_path)

    stats = {
        "documents": len(metas),
        "pages": sum(m["num_pages"] for m in metas),
        "parsed_items": len(items),
        "chunks": len(chunks),
        "tables": sum(1 for c in chunks if c.chunk_type == "table"),
        "figures": sum(1 for c in chunks if c.chunk_type == "figure"),
        "text": sum(1 for c in chunks if c.chunk_type == "text"),
        "vlm": vlm_stats,
        "seconds": round(time.perf_counter() - t0, 1),
    }
    (settings.index_dir / "ingest_stats.json").write_text(
        json.dumps(stats, indent=2), encoding="utf-8")
    print(f"[ingest] done in {stats['seconds']}s: {stats}")
    return stats


def cmd_index(settings: Settings | None = None) -> dict:
    settings = settings or load_settings()
    t0 = time.perf_counter()
    chunks = load_chunks(settings.chunks_path)
    if not chunks:
        raise SystemExit("No chunks found — run `mmrag ingest` first.")

    embedder = get_embedder(settings)
    texts = [c.text for c in chunks]

    if hasattr(embedder, "fit"):
        embedder.fit(texts)
        # persist TF-IDF statistics so query-time embedding matches index-time
        with (settings.index_dir / "embedder.pkl").open("wb") as fh:
            pickle.dump({"name": embedder.name, "idf": embedder._idf,
                         "fitted": embedder._fitted, "dim": embedder.dim}, fh)

    print(f"[index] embedding {len(texts)} chunks with {embedder.name} ...")
    vectors = embedder.embed(texts)

    store = open_store(settings, embedder.dim)
    store.reset()  # drop vectors from any previous chunking
    store.upsert([c.chunk_id for c in chunks], vectors)

    bm25 = BM25Index.build([c.chunk_id for c in chunks], texts)
    bm25.save(settings.bm25_path)

    stats = {"chunks": len(chunks), "embedder": embedder.name,
             "dim": int(embedder.dim), "store": store.name,
             "store_count": store.count(),
             "seconds": round(time.perf_counter() - t0, 1)}
    (settings.index_dir / "index_stats.json").write_text(
        json.dumps(stats, indent=2), encoding="utf-8")
    print(f"[index] done in {stats['seconds']}s: {stats}")
    return stats
