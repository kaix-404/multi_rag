"""40-question challenge suite generated from the ingested corpus.

Per spec: 15 Factual Text + 15 Tabular Numerical + 10 Visual/Chart questions.
Questions are derived deterministically from parsed evidence so every item
carries ground truth *and* gold retrieval evidence (chunk ids), which makes
Hit-Rate@5 and context precision/recall measurable without an LLM judge.
"""

from __future__ import annotations

import json
import random
import re
from pathlib import Path

from ..ingestion.chunker import Chunk

TARGET = {"factual": 15, "tabular": 15, "visual": 10}

TICKER_COMPANY = {
    "AAPL": "Apple Inc.", "TSLA": "Tesla, Inc.", "MSFT": "Microsoft Corporation",
    "NVDA": "NVIDIA Corporation", "GOOGL": "Alphabet Inc.", "AMZN": "Amazon.com, Inc.",
    "META": "Meta Platforms, Inc.",
}
_NUM_RE = re.compile(r"^\(?\$?\s*-?[\d,]+(?:\.\d+)?\s*(?:million|billion|%|shares)?\)?$", re.I)
_YEAR_RE = re.compile(r"\b(19|20)\d{2}\b")


def doc_label(filename: str) -> str:
    """Human label for a corpus document (ticker/year or arXiv id)."""
    m = re.search(r"_([A-Z]{2,5})_(\d{4})\.pdf$", filename, re.I)
    if m:
        ticker, year = m.group(1).upper(), m.group(2)
        company = TICKER_COMPANY.get(ticker, ticker)
        return f"{company} {year} Form 10-K"
    m = re.search(r"(\d{4}\.\d{4,5})", filename)
    if m:
        return f"arXiv paper {m.group(1)}"
    return Path(filename).stem


def _clean_cell(v: str) -> str:
    return v.replace("\\|", "|").strip()


def _to_number(v: str):
    v = _clean_cell(v)
    neg = v.startswith("(") and v.endswith(")")
    v = v.strip("()").replace("$", "").replace(",", "").replace("%", "").strip()
    try:
        x = float(v)
    except ValueError:
        return None
    return -x if neg else x


def _tabular_questions(chunks: list[Chunk], labels: dict[str, str],
                       rng: random.Random, n: int) -> list[dict]:
    out: list[dict] = []
    seen_rows: set[tuple[str, str]] = set()
    candidates = [c for c in chunks if c.chunk_type == "table" and "|" in c.text]
    rng.shuffle(candidates)
    for c in candidates:
        if len(out) >= n:
            break
        lines = [l for l in c.text.splitlines() if l.strip().startswith("|")]
        if len(lines) < 3:
            continue

        def _cells(line: str) -> list[str]:
            return [_clean_cell(x) for x in line.strip().strip("|").split("|")]

        header = _cells(lines[0])
        body = [_cells(l) for l in lines[2:]]
        # year columns: stitched tables often put the "Three Years Ended …"
        # row *below* the caption, so scan header + first body rows
        year_cols: list[tuple[int, str]] = []
        seen: set[int] = set()
        for row in [header] + body[:3]:
            for i, cell in enumerate(row):
                m = _YEAR_RE.search(cell)
                if m and i not in seen:
                    year_cols.append((i, m.group(0)))
                    seen.add(i)
        rows = [r for r in body if len(r) == len(header) and r[0]]
        numeric_rows = [r for r in rows
                        if sum(1 for cell in r[1:] if _to_number(cell) is not None)
                        >= max(2, len(header) - 2)]
        if not numeric_rows:
            continue
        label = labels.get(c.doc_id, c.doc_id)
        # two year columns -> percentage-change question (spec's canonical example)
        if len(year_cols) >= 2:
            (i1, y1), (i2, y2) = year_cols[0], year_cols[1]
            row = next((r for r in numeric_rows
                        if _to_number(r[i1]) not in (None, 0.0)), None)
            if row is None:
                row = numeric_rows[0]
            metric = row[0]
            v1, v2 = _to_number(row[i1]), _to_number(row[i2])
            if v1 in (None, 0.0) or v2 is None:
                continue
            pct = (v2 - v1) / abs(v1) * 100.0
            out.append({
                "id": f"tab-{len(out)+1:02d}",
                "category": "tabular",
                "question": (f"What was the percentage change in {metric} at "
                             f"{label} between {y1} and {y2}?"),
                "ground_truth": f"{pct:.1f}%",
                "doc_id": c.doc_id, "page_number": c.page_number,
                "gold_chunk_ids": [c.chunk_id],
            })
            continue
        # single-value lookup (tables whose year header lives outside the
        # ruled grid — common in printed 10-Ks — still make answerable questions)
        for row in numeric_rows:
            if (c.doc_id, row[0]) in seen_rows:
                continue
            col_idx = next((i for i in range(1, len(header))
                            if _to_number(row[i]) is not None), None)
            if col_idx is None:
                continue
            val = _to_number(row[col_idx])
            caption = header[0].strip()
            where = (f"the '{caption}' table" if caption
                     else f"the table on page {c.page_number}")
            seen_rows.add((c.doc_id, row[0]))
            out.append({
                "id": f"tab-{len(out)+1:02d}",
                "category": "tabular",
                "question": (f"According to {label}, what value is reported for "
                             f"'{row[0]}' in {where}?"),
                "ground_truth": f"{val:g}",
                "doc_id": c.doc_id, "page_number": c.page_number,
                "gold_chunk_ids": [c.chunk_id],
            })
            break  # one question per table keeps the suite diverse
    return out


_SENT_RE = re.compile(r"(?<=[.!?])\s+")
_FACT_HINT = re.compile(
    r"(\$?\d[\d,]*\.?\d*\s*(million|billion|percent|%|shares))|"
    r"(revenue|net sales|income|expenditure|operating|margin|customers|employees)",
    re.I)


def _factual_questions(chunks: list[Chunk], labels: dict[str, str],
                       rng: random.Random, n: int) -> list[dict]:
    out: list[dict] = []
    candidates = [c for c in chunks if c.chunk_type == "text" and len(c.text) > 80]
    rng.shuffle(candidates)
    seen_prefixes: set[str] = set()
    for c in candidates:
        if len(out) >= n:
            break
        label = labels.get(c.doc_id, c.doc_id)
        sents = [s.strip() for s in _SENT_RE.split(c.text) if 60 <= len(s) <= 260]
        sents = [s for s in sents if _FACT_HINT.search(s)]
        if not sents:
            continue
        sents.sort(key=len, reverse=True)
        sent = None
        for s in sents:
            prefix = s[:60].lower()
            if prefix not in seen_prefixes:
                sent, = [s]
                seen_prefixes.add(prefix)
                break
        if not sent:
            continue
        # mask the most specific numeric/entity token -> cloze question
        tokens = re.findall(r"\$?\d[\d,]*(?:\.\d+)?\s*(?:million|billion|%)?|"
                            r"[A-Z][A-Za-z]+(?:\s[A-Z][A-Za-z]+)+", sent)
        if not tokens:
            continue
        target = tokens[0]
        masked = sent.replace(target, "_____", 1)
        if masked == sent:
            continue
        out.append({
            "id": f"fact-{len(out)+1:02d}",
            "category": "factual",
            "question": f"Fill in the blank in this statement from {label}: \"{masked}\"",
            "ground_truth": target,
            "doc_id": c.doc_id, "page_number": c.page_number,
            "gold_chunk_ids": [c.chunk_id],
        })
    return out


def _visual_questions(chunks: list[Chunk], labels: dict[str, str],
                      rng: random.Random, n: int) -> list[dict]:
    out: list[dict] = []
    candidates = [c for c in chunks if c.chunk_type == "figure"]
    rng.shuffle(candidates)
    for c in candidates:
        if len(out) >= n:
            break
        label = labels.get(c.doc_id, c.doc_id)
        caption = (c.caption or c.descriptor or "").strip()
        caption = re.sub(r"^\s*(figure|fig\.|chart|exhibit|table)\s*\d*[:.\s]*",
                         "", caption, flags=re.I).strip()
        if len(caption) < 15:
            continue
        snippet = caption[:140] + ("…" if len(caption) > 140 else "")
        ground_truth = caption or c.text
        out.append({
            "id": f"vis-{len(out)+1:02d}",
            "category": "visual",
            "question": (f"In {label}, what does the figure described as "
                         f"\"{snippet}\" illustrate?"),
            "ground_truth": ground_truth[:400],
            "doc_id": c.doc_id, "page_number": c.page_number,
            "gold_chunk_ids": [c.chunk_id],
        })
    return out


def build_dataset(chunks: list[Chunk], docs_meta: list[dict],
                  seed: int = 42) -> dict:
    rng = random.Random(seed)
    labels = {m["doc_id"]: doc_label(m["filename"]) for m in docs_meta}
    # figure descriptors must exist before visual questions are cut
    items = (_tabular_questions(chunks, labels, rng, TARGET["tabular"])
             + _factual_questions(chunks, labels, rng, TARGET["factual"])
             + _visual_questions(chunks, labels, rng, TARGET["visual"]))
    counts = {k: sum(1 for i in items if i["category"] == k) for k in TARGET}
    return {"version": 1, "seed": seed, "counts": counts,
            "target": TARGET, "items": items}


def save_dataset(ds: dict, path: Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(ds, ensure_ascii=False, indent=2), encoding="utf-8")


def load_dataset(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))
