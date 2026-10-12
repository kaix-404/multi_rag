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
"""

from __future__ import annotations

import json
import threading
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse

from ..config import load_settings
from ..rag.generate import _ctx_dict, build_prompt, stream_answer
from ..rag.llm import LLMClient
from ..retrieval.retriever import HybridRetriever

settings = load_settings()
app = FastAPI(title="Multimodal Hybrid RAG Engine", version="0.1.0")

_lock = threading.Lock()
_retriever: HybridRetriever | None = None
_llm: LLMClient | None = None
_last_telemetry: dict | None = None


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
async def query(request: Request):
    body = await request.json()
    question = (body.get("question") or "").strip()
    if not question:
        raise HTTPException(status_code=422, detail="question is required")
    top_k = int(body.get("top_k", settings.answer_top_k))
    stream = bool(body.get("stream", False))

    retriever = get_retriever()
    llm = get_llm()
    results, telemetry = retriever.retrieve(question, top_k=top_k)
    tel = telemetry.__dict__.copy()
    global _last_telemetry
    _last_telemetry = tel

    contexts = [_ctx_dict(i, r) for i, r in enumerate(results, 1)]

    if stream:
        def event_stream():
            for delta in stream_answer(question, results, llm):
                yield f"data: {json.dumps({'type': 'delta', 'text': delta})}\n\n"
            yield ("data: " + json.dumps({
                "type": "done",
                "generator": f"llm:{llm.model}" if llm.available
                             else "extractive-fallback",
                "contexts": contexts,
                "telemetry": tel,
            }) + "\n\n")
        return StreamingResponse(event_stream(), media_type="text/event-stream")

    from ..rag.generate import answer_question
    out = answer_question(question, results, llm, settings)
    out["telemetry"] = tel
    return out


@app.get("/telemetry")
def telemetry() -> dict:
    return {"last": _last_telemetry}
