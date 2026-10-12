"""Grounded answer synthesis.

With an LLM: strict "answer only from the numbered contexts, cite [n]"
prompting, streamed. Without one: an extractive fallback that quotes the
best-ranked sentences with the same [n] provenance markers, and the response
``generator`` field always says which mode produced it.
"""

from __future__ import annotations

import re
from typing import Iterator

from ..config import Settings
from ..retrieval.retriever import RetrievalResult
from .llm import LLMClient

_SYSTEM = (
    "You are a careful financial/technical research assistant. Answer ONLY from "
    "the supplied contexts. Cite supporting contexts inline as [1], [2], ... "
    "If the contexts do not contain the answer, say so explicitly. Quote numbers "
    "and periods exactly. Be concise (<= 180 words)."
)


def _context_block(results: list[RetrievalResult]) -> str:
    parts = []
    for i, r in enumerate(results, start=1):
        c = r.chunk
        src = (f"{c.section or 'document'} | page {c.page_number} | "
               f"{c.chunk_type} | {c.chunk_id}")
        parts.append(f"[{i}] ({src})\n{c.text}")
    return "\n\n".join(parts)


def build_prompt(question: str, results: list[RetrievalResult]) -> list[dict]:
    user = f"Contexts:\n{_context_block(results)}\n\nQuestion: {question}"
    return [
        {"role": "system", "content": _SYSTEM},
        {"role": "user", "content": user},
    ]


def _extractive_answer(results: list[RetrievalResult], question: str) -> str:
    """No-LLM fallback: quote the top-ranked evidence sentences."""
    q_terms = set(re.findall(r"[a-z0-9]+", question.lower()))
    picks: list[str] = []
    for i, r in enumerate(results[:3], start=1):
        best, best_score = None, 0.0
        for sent in re.split(r"(?<=[.!?])\s+", r.chunk.text):
            terms = set(re.findall(r"[a-z0-9]+", sent.lower()))
            score = len(q_terms & terms) / (len(q_terms) or 1)
            if score > best_score and 30 <= len(sent) <= 400:
                best, best_score = sent, score
        if best:
            picks.append(f"{best.strip()} [{i}]")
    if not picks:
        picks = [f"{results[0].chunk.text[:300]} [1]"] if results else []
    return ("Extractive answer (no LLM configured — enable OPENAI_API_KEY for "
            "generative synthesis):\n" + "\n".join(f"• {p}" for p in picks))


def answer_question(question: str, results: list[RetrievalResult], llm: LLMClient,
                    settings: Settings) -> dict:
    """Return {answer, generator, contexts}. Non-streaming convenience wrapper."""
    if llm.available:
        answer = llm.chat(build_prompt(question, results))
        generator = f"llm:{llm.model}"
    else:
        answer = _extractive_answer(results, question)
        generator = "extractive-fallback"
    return {"answer": answer, "generator": generator,
            "contexts": [_ctx_dict(i, r) for i, r in enumerate(results, 1)]}


def stream_answer(question: str, results: list[RetrievalResult], llm: LLMClient):
    """Yield answer text chunks (LLM streaming, or one-shot fallback)."""
    if llm.available:
        yield from llm.chat_stream(build_prompt(question, results))
    else:
        yield _extractive_answer(results, question)


def _ctx_dict(i: int, r: RetrievalResult) -> dict:
    c = r.chunk
    return {
        "rank": i,
        "chunk_id": c.chunk_id,
        "doc_id": c.doc_id,
        "page_number": c.page_number,
        "chunk_type": c.chunk_type,
        "section": c.section,
        "bbox": list(c.bbox),
        "page_size": list(c.page_size),
        "asset_path": c.asset_path,
        "text": c.text,
        "dense_rank": r.dense_rank,
        "dense_score": r.dense_score,
        "sparse_rank": r.sparse_rank,
        "sparse_score": r.sparse_score,
        "rrf_score": r.rrf_score,
        "rerank_confidence": r.rerank_confidence,
    }
