"""Vision-language descriptors for cropped figures/charts.

Spec: "An automated Vision Model inspects the image and generates a synthetic
descriptor text block paired with the image reference" so figures become
directly indexable by the lexical + dense retrievers.

With a configured VLM (GPT-4o-mini, or LLaVA via Ollama's OpenAI endpoint)
the real vision model is used. Without one, a deterministic caption-anchored
descriptor is built from the page's own text so figures remain retrievable —
the descriptor records which mode produced it.
"""

from __future__ import annotations

from ..ingestion.chunker import Chunk
from .llm import LLMClient

_VLM_PROMPT = (
    "You are indexing a financial/technical document for a retrieval system. "
    "Describe this figure precisely and factually in 2-4 sentences: chart type, "
    "axes/units if visible, series and notable trends or values, and any legend. "
    "Only state what is visible. Respond with the description only."
)


def _fallback_descriptor(chunk: Chunk, nearby_text: str) -> str:
    bits = [
        f"[figure, page {chunk.page_number}]",
        f"Section: {chunk.section}." if chunk.section else "",
        f"Caption: {chunk.caption}" if chunk.caption else "",
        f"Context: {nearby_text[:400]}" if nearby_text else "",
    ]
    return " ".join(b for b in bits if b).strip()


def _set_figure_text(chunk: Chunk) -> None:
    """Figure chunks index caption + descriptor as one lexical unit."""
    joined = " ".join(x for x in [chunk.caption, chunk.descriptor] if x).strip()
    chunk.text = joined or f"[figure, page {chunk.page_number}] {chunk.section}".strip()


def describe_figures(chunks: list[Chunk], llm: LLMClient,
                     verbose: bool = True) -> dict[str, str]:
    """Fill ``chunk.descriptor`` for every figure chunk. Returns mode stats."""
    figures = [c for c in chunks if c.chunk_type == "figure"]
    stats = {"figures": len(figures), "vlm": 0, "fallback": 0}
    if not figures:
        return stats

    # nearby prose for fallback descriptors (same page, non-figure text)
    page_text: dict[tuple[str, int], list[str]] = {}
    for c in chunks:
        if c.chunk_type == "text":
            page_text.setdefault((c.doc_id, c.page_number), []).append(c.text)

    for c in figures:
        nearby = " ".join(page_text.get((c.doc_id, c.page_number), []))[:600]
        if llm.available and c.asset_path:
            try:
                desc = llm.describe_image(c.asset_path, _VLM_PROMPT).strip()
                if desc:
                    c.descriptor = desc
                    stats["vlm"] += 1
                    _set_figure_text(c)
                    continue
            except Exception as exc:
                if verbose:
                    print(f"[vlm] {c.chunk_id}: {exc.__class__.__name__}, using fallback")
        c.descriptor = _fallback_descriptor(c, nearby)
        stats["fallback"] += 1
        _set_figure_text(c)

    if verbose:
        print(f"[vlm] described {stats['figures']} figures "
              f"(vlm={stats['vlm']}, fallback={stats['fallback']})")
    return stats
