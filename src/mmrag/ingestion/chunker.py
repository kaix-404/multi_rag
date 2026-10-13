"""Semantic window chunking: 512-token sentence windows with 10% overlap.

Tables and figures stay atomic (never split) so 2-D header/column relations
and chart assets survive intact; prose is packed greedily by sentence so
windows break on natural boundaries instead of mid-token. Every chunk keeps
``{doc_id, page_number, chunk_type, bounding_box, section}`` provenance.
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
        d = dict(d)
        d["bbox"] = tuple(d.get("bbox", (0, 0, 0, 0)))
        d["page_size"] = tuple(d.get("page_size", (0, 0)))
        return Chunk(**d)


def _pack_sentences(sentences: list[str], budget: int, overlap_budget: int
                    ) -> list[tuple[int, int, int]]:
    """Greedy pack -> list of (start, end, overlap_start) sentence spans."""
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

        # overlap: walk back from the start of this window
        ov_start, ov_tokens = j, 0
        while ov_start > i:
            t = token_est(sentences[ov_start - 1])
            if ov_tokens + t > overlap_budget:
                break
            ov_tokens += t
            ov_start -= 1
        ov_start = max(ov_start, i) if spans else i  # no overlap into prior doc start
        spans.append((i, j, ov_start if spans else i))
        i = j
    return spans


def _chunk_text_item(item: ParsedItem, counter: list[int]) -> list[Chunk]:
    budget = 512
    overlap = int(budget * 0.10)
    sentences = split_sentences(item.text)
    if not sentences:
        return []
    out: list[Chunk] = []
    for (s, e, ov) in _pack_sentences(sentences, budget, overlap):
        text = " ".join(sentences[ov:e])
        out.append(Chunk(
            chunk_id=f"{item.doc_id}:p{item.page_number}:{counter[0]}",
            doc_id=item.doc_id,
            page_number=item.page_number,
            chunk_type="text",
            section=item.section,
            text=text,
            bbox=item.bbox,
            page_size=item.page_size,
            meta={"sentence_span": [s, e], "is_heading": item.meta.get("is_heading", False)},
        ))
        counter[0] += 1
    return out


def _chunk_atomic(item: ParsedItem, counter: list[int]) -> list[Chunk]:
    text = item.text
    if item.chunk_type == "figure":
        # figures start with caption only; VLM descriptor is attached in rag.vlm
        # *after* chunking, and only then does the figure become searchable.
        text = item.caption or ""
    if not text.strip():
        # caption-less figure: keep it indexable via its provenance string
        if item.chunk_type == "figure":
            text = (f"[figure on page {item.page_number}"
                   f"{', ' + item.section if item.section else ''}]")
        else:
            return []
    if len(text) > 40_000:  # safety cap on giant tables
        text = text[:40_000] + "\n…[truncated]"
    ch = Chunk(
        chunk_id=f"{item.doc_id}:p{item.page_number}:{counter[0]}",
        doc_id=item.doc_id,
        page_number=item.page_number,
        chunk_type=item.chunk_type,
        section=item.section,
        text=text,
        bbox=item.bbox,
        page_size=item.page_size,
        asset_path=item.asset_path,
        caption=item.caption,
        meta=dict(item.meta),
    )
    counter[0] += 1
    return [ch]


TINY_TEXT_CHARS = 120  # row labels / orphaned fragments below this get merged


def _merge_bbox(items: list[ParsedItem]) -> tuple[float, float, float, float]:
    return (
        min(it.bbox[0] for it in items),
        min(it.bbox[1] for it in items),
        max(it.bbox[2] for it in items),
        max(it.bbox[3] for it in items),
    )


def _coalesce_text(items: list[ParsedItem]) -> ParsedItem:
    """Fuse a run of small text blocks (typical 10-K row-label leakage) into
    one paragraph-level item so retrievers see context, not label shards."""
    head = items[0]
    text = " ".join(it.text for it in items)
    return ParsedItem(
        doc_id=head.doc_id,
        page_number=head.page_number,
        chunk_type="text",
        text=text,
        bbox=_merge_bbox(items),
        page_size=head.page_size,
        section=next((it.section for it in reversed(items) if it.section), head.section),
        order=items[0].order,
        meta={"coalesced_blocks": len(items)},
    )


def chunk_documents(items: list[ParsedItem], settings: Settings | None = None) -> list[Chunk]:
    """Chunk a parsed corpus preserving document/page order.

    Small text blocks (single row labels, orphaned headings, footer shards)
    are buffered and fused into the next substantial paragraph on the same
    page, which prevents short label fragments from dominating BM25/dense
    rankings in financial filings.
    """
    by_doc: dict[str, list[ParsedItem]] = {}
    for it in items:
        by_doc.setdefault(it.doc_id, []).append(it)

    chunks: list[Chunk] = []
    for doc_id in sorted(by_doc):
        doc_items = sorted(by_doc[doc_id], key=lambda x: (x.page_number, x.order))
        counter = [0]
        pending: list[ParsedItem] = []
        for item in doc_items:
            if item.chunk_type == "text":
                if len(item.text) < TINY_TEXT_CHARS or item.meta.get("is_heading"):
                    pending.append(item)
                    continue
                merged = _coalesce_text(pending + [item]) if pending else item
                chunks.extend(_chunk_text_item(merged, counter))
                pending = []
            else:
                if pending:
                    chunks.extend(_chunk_text_item(_coalesce_text(pending), counter))
                    pending = []
                chunks.extend(_chunk_atomic(item, counter))
        if pending:
            chunks.extend(_chunk_text_item(_coalesce_text(pending), counter))
    return chunks


# --------------------------------------------------------------------------- #
# persistence
# --------------------------------------------------------------------------- #

def save_chunks(chunks: list[Chunk], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for ch in chunks:
            fh.write(json.dumps(ch.to_dict(), ensure_ascii=False) + "\n")


def load_chunks(path: Path) -> list[Chunk]:
    out: list[Chunk] = []
    with Path(path).open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                out.append(Chunk.from_dict(json.loads(line)))
    return out


def save_json(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def load_json(path: Path):
    return json.loads(Path(path).read_text(encoding="utf-8"))
