"""Semantic window chunking: 512-token sentence windows with 10% overlap.

Tables and figures stay atomic (never split) so 2-D header/column relations
and chart assets survive intact; prose is packed greedily by sentence so
windows break on natural boundaries instead of mid-token. Every chunk keeps
`{doc_id, page_number, chunk_type, bounding_box, section}` provenance.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path

from ..config import Settings
from .parser import ParsedItem

_SENT_SPLIT = re.compile(r"(?<=[.!?;:])\s+(?=[A-Z0-9\"'(])")
_TOKEN_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)


def token_est(text: str) -> int:
    """Cheap tokenizer-independent token estimate (words + punctuation)."""
    return len(_TOKEN_RE.findall(text))


def split_sentences(text: str) -> list[str]:
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []
    parts = _SENT_SPLIT.split(text)
    return [p.strip() for p in parts if p.strip()]


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    page_number: int
    chunk_type: str              # text | table | figure
    section: str
    text: str
    bbox: tuple[float, float, float, float]
    page_size: tuple[float, float]
    asset_path: str = ""
    caption: str = ""
    descriptor: str = ""         # VLM-generated figure descriptor (filled later)
    meta: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["bbox"] = list(self.bbox)
        d["page_size"] = list(self.page_size)
        return d

    @staticmethod
    def from_dict(d: dict) -> "Chunk":
        return Chunk(
            chunk_id=d["chunk_id"],
            doc_id=d["doc_id"],
            page_number=int(d["page_number"]),
            chunk_type=d["chunk_type"],
            section=d.get("section", ""),
            text=d["text"],
            bbox=tuple(d.get("bbox", (0, 0, 0, 0))),
            page_size=tuple(d.get("page_size", (0, 0, 0, 0))),
            asset_path=d.get("asset_path", ""),
            caption=d.get("caption", ""),
            descriptor=d.get("descriptor", ""),
            meta=dict(d.get("meta", {})),
        )


_DOC_LABELS = {
    "AAPL": "Apple Inc.",
    "MSFT": "Microsoft Corporation",
    "TSLA": "Tesla, Inc.",
}

# Backward-compatible alias used by any existing import of DOC_LABELS.
DOC_LABELS = _DOC_LABELS


def doc_context_header(doc_id: str) -> str:
    """Document-contextualized header prepended to every chunk of a document.

    Parses the doc_id prefix and emits a short provenance string that the
    retriever prepends to every chunk so the LLM can always see which
    document and page a chunk came from.
    """
    prefix = doc_id.split("_")[0].upper()
    label = _DOC_LABELS.get(prefix, prefix)
    m = re.match(r"^([A-Z]+)_(\d{4})", doc_id)
    if m:
        fiscal_year = m.group(2)
    else:
        fiscal_year = doc_id.split("_")[1] if "_" in doc_id else "unknown"
    return f"[{label} fiscal year {fiscal_year} annual report (Form 10-K) | {doc_id}]"


def _pack_sentences(sentences: list[str], budget: int, overlap_budget: int
                    ) -> list[tuple[int, int, int]]:
    """Greedy pack -> list of (start, end, overlap_start) sentence spans.

    Overlap windows are created by walking back from the window START
    (ov_start), not the window end. Walking back from the window end instead
    leaves ov_end modified and makes every middle window emit only its tiny
    overlap tail and silently drop its whole body.
    """
    spans: list[tuple[int, int, int]] = []
    i = 0
    n = len(sentences)
    while i < n:
        total = 0
        j = i
        while j < n:
            t = token_est(sentences[j])
            if total + t > budget and j > i:
                break
            total += t
            j += 1
        if j == i:  # single sentence longer than budget: hard-split by words
            words = sentences[i].split()
            cut, acc, k = [], 0, 0
            while k < len(words):
                acc += 1
                if acc >= budget:
                    cut.append(k)
                    acc = 0
                k += 1
            # degenerate safety: emit at least one span
            spans.append((i, i + 1, i))
            i += 1
            continue

        # overlap: extend this window backward into the previous window's tail
        # (walk back from the window START, floored at 0). The consumer joins
        # sentences[ov_start:end], so ov_start must precede i -- walking back
        # from the window end instead made every middle window emit only its
        # tiny overlap tail and silently drop its whole body.
        ov_start, ov_tokens = i, 0
        while ov_start > 0:
            t = token_est(sentences[ov_start - 1])
            if ov_tokens + t > overlap_budget:
                break
            ov_tokens += t
            ov_start -= 1

        spans.append((ov_start, j, ov_start))

        i = j

    return spans


def chunk_documents(items, settings):
    """Chunk a list of parsed items into a list of Chunk objects."""
    from .parser import ParsedItem

    by_doc: dict[str, list[ParsedItem]] = {}
    for it in items:
        by_doc.setdefault(it.doc_id, []).append(it)

    chunks: list[Chunk] = []
    doc_chunk_start: dict[str, int] = {}

    for doc_id in sorted(by_doc):
        doc_chunk_start[doc_id] = len(chunks)
        doc_items = sorted(by_doc[doc_id], key=lambda x: (x.page_number, x.order))
        pending: list[ParsedItem] = []
        for item in doc_items:
            if item.chunk_type == "text":
                if len(item.text) < 120 or item.meta.get("is_heading"):
                    pending.append(item)
                    continue
                merged = _coalesce_text(pending + [item]) if pending else item
                chunks.extend(_chunk_text_item(merged))
                pending = []
            else:
                if pending:
                    chunks.extend(_chunk_text_item(_coalesce_text(pending)))
                    pending = []
                chunks.extend(_chunk_atomic(item))
        if pending:
            chunks.extend(_chunk_text_item(_coalesce_text(pending)))

    for doc_id in sorted(by_doc):
        start = doc_chunk_start[doc_id]
        header = doc_context_header(doc_id)
        for ch in chunks[start:]:
            if ch.doc_id == doc_id:
                ch.text = f"{header} {ch.text}".strip()
            else:
                break

    for ch in chunks:
        if not ch.chunk_id:
            ch.chunk_id = f"{ch.doc_id}:{ch.page_number}:{id(ch) % 10**8}"

    return chunks


def _coalesce_text(items: list[ParsedItem]) -> ParsedItem:
    """Coalesce a list of text items into one ParsedItem."""
    if not items:
        raise ValueError("_coalesce_text called with empty list")
    first = items[0]
    kw: dict = dict(first.__dict__) if hasattr(first, "__dict__") else {}
    kw["text"] = "\n".join(it.text for it in items)
    kw["chunk_type"] = "text"
    return ParsedItem(**kw)


def _chunk_text_item(item: ParsedItem) -> list[Chunk]:
    """Chunk a single text item into sentence-window chunks."""
    sents = split_sentences(item.text)
    if not sents:
        return []
    budget = 512
    overlap_budget = max(8, int(budget * 0.10))
    spans = _pack_sentences(sents, budget, overlap_budget)
    chunks = []
    for start, end, ov_start in spans:
        body = " ".join(sents[ov_start:end])
        chunk = Chunk(
            chunk_id="",
            doc_id=item.doc_id,
            page_number=item.page_number,
            chunk_type="text",
            section=item.section,
            text=body,
            bbox=item.bbox,
            page_size=item.page_size,
        )        chunks.append(chunk)
    return chunks


def _chunk_atomic(item: ParsedItem) -> list[Chunk]:
    """Chunk a single atomic item (table/figure) into one chunk."""
    kw: dict = dict(item.__dict__) if hasattr(item, "__dict__") else {}
    kw.setdefault("chunk_type", item.chunk_type)
    return [Chunk(
        chunk_id="",
        doc_id=item.doc_id,
        page_number=item.page_number,
        chunk_type=item.chunk_type,
        section=item.section,
        text=item.text,
        bbox=item.bbox,
        page_size=item.page_size,
    )]




def load_chunks(path):
    """Load chunks from a JSONL file as Chunk objects."""
    chunks: list[Chunk] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        d = json.loads(line)
        obj = Chunk.from_dict(d)
        if not isinstance(obj, Chunk):
            raise TypeError(
                f"load_chunks: Chunk.from_dict returned {type(obj).__name__} "
                f"for chunk_id={d.get('chunk_id')!r}; expected Chunk."
            )
        chunks.append(obj)
    return chunks


def save_chunks(chunks, path):
    """Save chunks to a JSONL file."""
    lines = [json.dumps(c.to_dict(), ensure_ascii=False) for c in chunks]
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


# Re-export the names the rest of the package expects.
__all__ = [
    "Chunk", "chunk_documents", "load_chunks", "save_chunks",
    "split_sentences", "token_est", "doc_context_header",
    "_pack_sentences", "_DOC_LABELS", "DOC_LABELS",
]


def __getattr__(name):
    if name in (__all__):
        return globals()[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
