"""FastAPI backend for the demonstration portal.

Endpoints
- ``GET  /health``                       backends + index stats
- ``GET  /documents``                    ingested document metadata
- ``GET  /documents/{id}/pages/{n}``     rendered page PNG (for bbox overlays)
- ``GET  /figures/{name}``               cropped figure PNG asset
- ``POST /query``                        hybrid retrieval + grounded answer
      body: {question, top_k?, stream?}
      JSON mode returns {answer, generator, contexts[], telemetry}
      stream=true returns SSE events: `delta` → `done`
- ``POST /documents/upload``             ingest+index one PDF (multipart file),
      embeds only the new chunks and hot-swaps the retriever
"""

from __future__ import annotations

import json
import re
import threading
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from starlette.requests import Request as StarletteRequest

from ..config import load_settings
from ..rag.generate import _ctx_dict, build_prompt, stream_answer
from ..rag.llm import LLMClient
from ..retrieval.retriever import HybridRetriever

settings = load_settings()
app = FastAPI(title="Multimodal Hybrid RAG Engine", version="0.1.0")

_lock = threading.Lock()
_ingest_lock = threading.Lock()
_retriever: HybridRetriever | None = None
_llm: LLMClient | None = None
_last_telemetry: dict | None = None

MAX_UPLOAD_MB = 30
MAX_UPLOAD_PAGES = 200


def get_retriever() -> HybridRetriever:
    global _retriever
    if _retriever is None:
        with _lock:
            if _retriever is None:
                _retriever = HybridRetriever.load(settings)
    return _retriever


def get_llm() -> LLMClient:
    global _llm
    if _llm is None:
        _llm = LLMClient(settings)
    return _llm


def _safe_under(root: Path, *parts: str) -> Path:
    p = (root.joinpath(*parts)).resolve()
    if not str(p).startswith(str(root.resolve())):
        raise HTTPException(status_code=404, detail="not found")
    return p


@app.get("/health")
def health() -> dict:
    try:
        r = get_retriever()
        return {
            "status": "ok",
            "chunks": len(r.chunks),
            "embedder": r.embedder.name,
            "reranker": r.reranker.name,
            "store": r.store.name,
            "store_count": r.store.count(),
            "llm": get_llm().model if get_llm().available else None,
            "rrf_k": settings.rrf_k,
        }
    except Exception as exc:
        return JSONResponse(status_code=503,
                            content={"status": "degraded", "error": str(exc)})


def _safe_pdf_name(name: str | None) -> str:
    """Sanitize an uploaded filename to a flat, traversal-safe .pdf name."""
    base = Path(name or "").name
    base = re.sub(r"[^A-Za-z0-9._-]", "_", base).strip("._")
    if not base.lower().endswith(".pdf"):
        base = f"{base or 'upload'}.pdf"
    return base


@app.post("/documents/upload")
def upload_document(file: UploadFile = File(...)) -> dict:
    """Accept one PDF, parse+chunk it, embed only its new chunks, and hot-swap
    the retriever so the document is searchable immediately. Multipart body
    with a single ``file`` field. Returns per-document ingest stats."""
    raw = file.file.read()
    if not raw.startswith(b"%PDF"):
        raise HTTPException(status_code=422, detail="uploaded file is not a PDF")
    if len(raw) > MAX_UPLOAD_MB * 1024 * 1024:
        raise HTTPException(status_code=413,
                            detail=f"PDF larger than {MAX_UPLOAD_MB} MB")

    import fitz  # pymupdf: cheap page-count check before the heavy parse
    try:
        with fitz.open(stream=raw, filetype="pdf") as doc:
            pages = doc.page_count
    except Exception:
        raise HTTPException(status_code=422, detail="could not open PDF")
    if pages == 0:
        raise HTTPException(status_code=422, detail="PDF has no pages")
    if pages > MAX_UPLOAD_PAGES:
        raise HTTPException(status_code=413,
                            detail=f"{pages} pages exceeds limit {MAX_UPLOAD_PAGES}")

    pdf_path = settings.corpus_dir / _safe_pdf_name(file.filename)
    settings.corpus_dir.mkdir(parents=True, exist_ok=True)
    pdf_path.write_bytes(raw)

    from ..index import BM25Index
    from ..ingestion.chunker import load_chunks
    from ..pipeline import ingest_single_pdf

    retriever = get_retriever()  # ensures the embedder is resident; reuse it
    with _ingest_lock:
        # Resolve the doc_id this upload must be compared against BEFORE the
        # "before" snapshot is taken, otherwise the per-document counts are
        # keyed by the wrong id and a re-upload reports zeros.
        old_chunks = load_chunks(settings.chunks_path)
        # name+size mirrors the parser's ``_doc_id()``, but the parser hashes
        # the full filename while this helper hashes the stem, so when a
        # docs.json entry already exists for this file (a previous ingest)
        # that recorded doc_id is authoritative.
        doc_id = _doc_id_for_pdf(pdf_path, raw)
        docs_path = settings.index_dir / "docs.json"
        existing_docs: list[dict] = []
        if docs_path.exists():
            try:
                existing_docs = json.loads(docs_path.read_text(encoding="utf-8"))
            except Exception:
                existing_docs = []
        match = next(
            (d["doc_id"] for d in existing_docs
             if d.get("filename") == pdf_path.name
             or d.get("path", "").endswith(pdf_path.name)
             or d.get("doc_id", "").split("-", 1)[0] == pdf_path.stem),
            None,
        )
        prev_pages = 0
        if match:
            doc_id = match
            prev_doc = next((d for d in existing_docs if d.get("doc_id") == match), {})
            prev_pages = int(prev_doc.get("num_pages") or 0)

        # Snapshot the pre-ingest state of this document and of the whole
        # index, so the upload response can say exactly what changed.
        before = {
            "chunks": len(old_chunks),
            "store_count": retriever.store.count(),
            "doc_count": len({c.doc_id for c in old_chunks}),
            "pages": prev_pages,
            "tables": sum(1 for c in old_chunks if c.doc_id == doc_id and c.chunk_type == "table"),
            "figures": sum(1 for c in old_chunks if c.doc_id == doc_id and c.chunk_type == "figure"),
            "text": sum(1 for c in old_chunks if c.doc_id == doc_id and c.chunk_type == "text"),
        }
        try:
            stats = ingest_single_pdf(pdf_path, settings,
                                      embedder=retriever.embedder,
                                      store=retriever.store,
                                      before=before)
        except Exception as exc:
            pdf_path.unlink(missing_ok=True)
            raise HTTPException(status_code=500, detail=f"ingest failed: {exc}")
        chunks = load_chunks(settings.chunks_path)
        bm25 = BM25Index.load(settings.bm25_path)
        # Hot-swap: in-flight queries keep the old retriever object (safe);
        # new requests see updated chunks/BM25 and the same embedder/store.
        global _retriever
        _retriever = HybridRetriever(chunks, retriever.embedder, bm25,
                                     retriever.store, retriever.reranker,
                                     settings)
    return {"status": "ok", **stats, "chunks": len(chunks)}


def _doc_id_for_pdf(pdf_path: Path, raw: bytes) -> str:
    """Mirror the package's existing doc_id derivation so the before/after
    snapshot keys line up with what ``ingest_single_pdf`` will produce."""
    import hashlib
    h = hashlib.sha1(f"{pdf_path.stem[:48]}:{pdf_path.stat().st_size}".encode()).hexdigest()[:10]
    return f"{pdf_path.stem[:48]}-{h}"


@app.get("/documents")
def documents() -> dict:
    path = settings.index_dir / "docs.json"
    if not path.exists():
        return {"documents": []}
    return {"documents": json.loads(path.read_text(encoding="utf-8"))}


@app.get("/documents/{doc_id}/pages/{page_number}")
def page_image(doc_id: str, page_number: int) -> FileResponse:
    path = _safe_under(settings.pages_dir, f"{doc_id}_p{page_number}.png")
    if not path.exists():
        raise HTTPException(status_code=404, detail="page render not found")
    return FileResponse(path, media_type="image/png")


@app.get("/figures/{name}")
def figure_asset(name: str) -> FileResponse:
    path = _safe_under(settings.assets_dir, name)
    if not path.exists():
        raise HTTPException(status_code=404, detail="asset not found")
    return FileResponse(path, media_type="image/png")


@app.post("/query")
async def query(request: StarletteRequest):
    body = await request.json()
    question = (body.get("question") or "").strip()
    if not question:
        raise HTTPException(status_code=422, detail="question is required")
    top_k = int(body.get("top_k", settings.answer_top_k))
    stream = bool(body.get("stream", False))

    if stream:
        retriever = get_retriever()
        llm = get_llm()
        results, _telemetry = retriever.retrieve(question, top_k=top_k)
        tel = _telemetry.__dict__.copy()
        global _last_telemetry
        _last_telemetry = tel
        contexts = [_ctx_dict(i, r) for i, r in enumerate(results, 1)]

        def event_stream():
            for delta in stream_answer(question, results, llm):
                yield f"data: {json.dumps(dict(type='delta', text=delta), ensure_ascii=True)}\n\n"
            yield ("data: " + json.dumps(dict(
                type='done',
                generator=f"llm:{llm.model}" if llm.available
                          else "extractive-fallback",
                contexts=contexts,
                telemetry=tel,
            ), ensure_ascii=True) + "\n\n")

        return StreamingResponse(event_stream(), media_type="text/event-stream; charset=utf-8", headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no", "Access-Control-Allow-Origin": "*"})

    retriever = get_retriever()
    llm = get_llm()
    results, _telemetry = retriever.retrieve(question, top_k=top_k)
    tel = _telemetry.__dict__.copy()
    _last_telemetry = tel

    contexts = [_ctx_dict(i, r) for i, r in enumerate(results, 1)]

    from ..rag.generate import answer_question
    out = answer_question(question, results, llm, settings)
    out["telemetry"] = tel

    ctxs = contexts if isinstance(contexts, list) else []
    if ctxs and isinstance(ctxs[0], dict) and ctxs[0].get("rrf_score"):
        rank1 = ctxs[0]["rank"]
        rrf1 = float(ctxs[0]["rrf_score"])
        rrf2 = float(ctxs[1]["rrf_score"]) if len(ctxs) > 1 and isinstance(ctxs[1], dict) else 0.0
        answer_length = len(out.get("answer", ""))
        delta = rrf1 - rrf2
        if delta > 0.015 and rank1 == 1:
            label, reason = "High", (
                "The model grounded this answer in its top-ranked evidence with a clear winner "
                "between the best and second-best passage."
            )
        elif delta > 0.003 and rank1 <= 2:
            label, reason = "Medium", (
                "The answer is supported by ranked evidence, but several passages scored similarly "
                "— treat the key numbers as worth double-checking."
            )
        else:
            label, reason = "Low", (
                "This answer is tentative. The strongest evidence still carried the result, but the "
                "model had little to work with — verify the key point against the document itself."
            )
        out["confidence_score"] = round((rrf1 / (rrf1 + rrf2)) if (rrf1 + rrf2) > 0 else 0.0, 3)
        out["confidence_label"] = label
        out["confidence_breakdown"] = {
            "top_context_rank": rank1,
            "top_rrf": round(rrf1, 5),
            "second_rrf": round(rrf2, 5),
            "rrf_gap": round(delta, 5),
            "answer_length_chars": answer_length,
            "reason": reason,
        }
    else:
        out["confidence_score"] = 0.0
        out["confidence_label"] = "Low"
        out["confidence_breakdown"] = {
            "top_context_rank": None,
            "top_rrf": None,
            "second_rrf": None,
            "rrf_gap": None,
            "answer_length_chars": len(out.get("answer", "")),
            "reason": (
                "The query returned no scorable evidence for a reliable confidence estimate. "
                "Treat the answer as tentative and verify the key point against the document."
            ),
        }

    return out


@app.get("/telemetry")
def telemetry() -> dict:
    return {"last": _last_telemetry}
