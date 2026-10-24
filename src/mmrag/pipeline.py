"""Pipeline orchestration: ingest (parse+chunk+VLM) and index (embed+store)."""

from __future__ import annotations

import json
import pickle
import time
from pathlib import Path

from .config import Settings, load_settings
from .eval.harness_io import save_parsed_items
from .index import BM25Index, VectorStore, get_embedder, open_store
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


def ingest_single_pdf(pdf_path: Path, settings: Settings | None = None, *,
                      embedder=None, store: VectorStore | None = None,
                      render_dpi: int = 144,
                      before: dict | None = None) -> dict:
    """Incrementally ingest ONE PDF: parse -> chunk -> figure descriptors ->
    embed only its chunks -> upsert into the existing store -> rebuild BM25.

    Pass the live ``embedder``/``store`` (the API passes its resident
    retriever's) so no model reload happens and existing vectors are never
    re-embedded. Re-ingesting a file whose doc_id already exists replaces that
    document's chunks and vectors. Returns per-document stats.

    The ``before`` snapshot should be a dict with the keys ``chunks``,
    ``store_count``, ``doc_count``, ``pages`` (the document's previous page
    count), ``tables``, ``figures`` and ``text`` (this document's previous
    chunk counts), captured before this PDF was parsed and keyed by the same
    doc_id the parser will produce. It is used only to produce a concise
    ``diff`` summary for the upload response; if omitted, the diff fields are
    omitted too.
    """
    settings = settings or load_settings()
    t0 = time.perf_counter()
    pdf_path = Path(pdf_path)

    from .ingestion.parser import parse_pdf
    print(f"[ingest] parsing {pdf_path.name} ...")
    meta, items = parse_pdf(pdf_path, settings.assets_dir, settings.pages_dir,
                            render_dpi=render_dpi)
    new_chunks = chunk_documents(items, settings)
    if not new_chunks:
        raise ValueError(f"{pdf_path.name}: no text/table/figure items parsed")
    vlm_stats = describe_figures(new_chunks, LLMClient(settings))

    existing = (load_chunks(settings.chunks_path)
                if settings.chunks_path.exists() else [])
    stale_ids = [c.chunk_id for c in existing if c.doc_id == meta["doc_id"]]
    merged = [c for c in existing if c.doc_id != meta["doc_id"]] + new_chunks

    if embedder is None:
        embedder = get_embedder(settings)
    if store is None:
        store = open_store(settings, embedder.dim)

    if stale_ids:
        print(f"[ingest] replacing {meta['doc_id']} "
              f"({len(stale_ids)} stale chunks dropped)")
        store.delete(stale_ids)

    texts = [c.text for c in new_chunks]
    print(f"[ingest] embedding {len(texts)} new chunks with {embedder.name} ...")
    store.upsert([c.chunk_id for c in new_chunks], embedder.embed(texts))

    # BM25 is rebuilt over the merged corpus (pure-Python, seconds); chunks.jsonl
    # is written after the store/bm25 so a crash mid-ingest never yields chunks
    # whose vectors are missing from the on-disk store.
    bm25 = BM25Index.build([c.chunk_id for c in merged], [c.text for c in merged])
    bm25.save(settings.bm25_path)
    save_chunks(merged, settings.chunks_path)

    docs_path = settings.index_dir / "docs.json"
    docs = (json.loads(docs_path.read_text(encoding="utf-8"))
            if docs_path.exists() else [])
    docs = [d for d in docs if d.get("doc_id") != meta["doc_id"]] + [meta]
    settings.index_dir.mkdir(parents=True, exist_ok=True)
    docs_path.write_text(json.dumps(docs, indent=2, ensure_ascii=False),
                         encoding="utf-8")

    stats = {
        "doc_id": meta["doc_id"],
        "filename": meta["filename"],
        "pages": meta["num_pages"],
        "chunks_added": len(new_chunks),
        "chunks_replaced": len(stale_ids),
        "tables": sum(1 for c in new_chunks if c.chunk_type == "table"),
        "figures": sum(1 for c in new_chunks if c.chunk_type == "figure"),
        "text": sum(1 for c in new_chunks if c.chunk_type == "text"),
        "chunks_total": len(merged),
        "store_count": store.count(),
        "embedder": embedder.name,
        "vlm": vlm_stats,
        "seconds": round(time.perf_counter() - t0, 1),
    }

    diff = None
    if before is not None:
        diff = {
            "chunks_before": before["chunks"],
            "chunks_after": len(merged),
            "chunks_added": stats["chunks_added"],
            "chunks_replaced": stats["chunks_replaced"],
            "chunks_net": stats["chunks_added"] - stats["chunks_replaced"],
            "store_count_before": before["store_count"],
            "store_count_after": store.count(),
            "doc_count_before": before["doc_count"],
            "doc_count_after": before["doc_count"] + (0 if stale_ids else 1) \
                if before.get("doc_count") is not None else None,
            "pages_before": before.get("pages"),
            "pages_after": meta["num_pages"],
            "is_replacement": bool(stale_ids),
            "pages_touched": sorted({c.page_number for c in new_chunks}),
            "tables_before": before.get("tables"),
            "tables_after": (
                (before.get("tables") or 0)
                - sum(1 for c in existing if c.doc_id == meta["doc_id"] and c.chunk_type == "table")
                + stats["tables"]
            ) if before.get("tables") is not None else None,
            "figures_before": before.get("figures"),
            "figures_after": (
                (before.get("figures") or 0)
                - sum(1 for c in existing if c.doc_id == meta["doc_id"] and c.chunk_type == "figure")
                + stats["figures"]
            ) if before.get("figures") is not None else None,
            "text_before": before.get("text"),
            "text_after": (
                (before.get("text") or 0)
                - sum(1 for c in existing if c.doc_id == meta["doc_id"] and c.chunk_type == "text")
                + stats["text"]
            ) if before.get("text") is not None else None,
        }

    result = {**stats}
    if diff:
        result["diff"] = diff
    print(f"[ingest] done in {stats['seconds']}s: {stats}")
    return result
