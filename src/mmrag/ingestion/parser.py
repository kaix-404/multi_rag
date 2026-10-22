"""Dual-stream PDF parser.

Stream 1 (PyMuPDF): text blocks with bounding boxes + raster figure crops.
Stream 2 (pdfplumber): coordinate-based table detection -> Markdown grids.

Every extracted unit carries the provenance payload required by the spec:
``{doc_id, page_number, chunk_type, bounding_box}``. Rendered page PNGs are
written alongside so the frontend can draw zero-latency bounding-box overlays.
"""

from __future__ import annotations

import hashlib
import os
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable

import fitz  # PyMuPDF
import pdfplumber

FIGURE_MIN_PT = 55.0          # ignore tiny icons
FIGURE_MAX_AREA = 0.85        # ignore full-page scans/backgrounds
CAPTION_MAX_DIST = 90.0       # pt below a figure where a caption may live
CAPTION_RE = re.compile(r"^\s*(figure|fig\.|chart|exhibit|table|graph|plot)\b", re.I)
VECTOR_CLUSTER_MIN_SEGS = 15
VECTOR_MIN_AREA = 0.035
VECTOR_MAX_AREA = 0.55


@dataclass
class ParsedItem:
    doc_id: str
    page_number: int                 # 1-based
    chunk_type: str                  # "text" | "table" | "figure"
    text: str
    bbox: tuple[float, float, float, float]  # (x0, y0, x1, y1) in PDF points
    page_size: tuple[float, float]   # (width, height) in PDF points
    section: str = ""
    asset_path: str = ""             # cropped figure PNG (figures only)
    caption: str = ""
    order: int = 0
    meta: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


def _doc_id(pdf_path: Path) -> str:
    h = hashlib.sha1(f"{pdf_path.name}:{pdf_path.stat().st_size}".encode()).hexdigest()[:10]
    return f"{pdf_path.stem[:48]}-{h}"


# --------------------------------------------------------------------------- #
# text + figures (PyMuPDF)
# --------------------------------------------------------------------------- #

def _median_body_size(doc: fitz.Document) -> float:
    sizes: list[float] = []
    for pno in range(min(4, doc.page_count)):
        d = doc[pno].get_text("dict")
        for block in d.get("blocks", []):
            if block.get("type") != 0:
                continue
            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    if span.get("text", "").strip():
                        sizes.append(float(span.get("size", 10.0)))
    if not sizes:
        return 10.0
    sizes.sort()
    return sizes[len(sizes) // 2]


def _join_block_text(block: dict) -> str:
    parts: list[str] = []
    for line in block.get("lines", []):
        line_txt = "".join(span.get("text", "") for span in line.get("spans", []))
        line_txt = line_txt.replace("\u00a0", " ").strip()
        if not line_txt:
            continue
        if parts and parts[-1].endswith("-") and line_txt[:1].islower():
            parts[-1] = parts[-1][:-1] + line_txt
        else:
            parts.append(line_txt)
    return re.sub(r"\s+", " ", " ".join(parts)).strip()


def _is_heading(block: dict, body_size: float) -> bool:
    lines = block.get("lines", [])
    if len(lines) != 1:
        return False
    txt = _join_block_text(block)
    if not (3 <= len(txt) <= 110) or txt.endswith((".", ":", ";")):
        return False
    spans = [s for ln in lines for s in ln.get("spans", [])]
    if not spans:
        return False
    bold = any("bold" in s.get("font", "").lower() or "black" in s.get("font", "").lower()
               for s in spans)
    big = max(float(s.get("size", 0)) for s in spans) >= body_size * 1.12
    caps = txt.isupper() and len(txt) <= 80
    return (bold or big) and (caps or bold or big)


def _overlap_frac(a: tuple, b: tuple) -> float:
    ix0, iy0 = max(a[0], b[0]), max(a[1], b[1])
    ix1, iy1 = min(a[2], b[2]), min(a[3], b[3])
    if ix1 <= ix0 or iy1 <= iy0:
        return 0.0
    inter = (ix1 - ix0) * (iy1 - iy0)
    area_a = max((a[2] - a[0]) * (a[3] - a[1]), 1e-9)
    return inter / area_a


def _extract_text_and_raster(
    page: fitz.Page,
    doc_id: str,
    pno: int,
    page_size: tuple[float, float],
    body_size: float,
    table_bboxes: list[tuple],
    assets_dir: Path,
    section: str,
    order_start: int,
) -> tuple[list[ParsedItem], str, int]:
    items: list[ParsedItem] = []
    order = order_start
    page_dict = page.get_text("dict", sort=True)

    # -- figures: embedded raster images ------------------------------------
    for i, info in enumerate(page.get_image_info(xrefs=True)):
        bbox = tuple(info.get("bbox", ()))
        if len(bbox) != 4:
            continue
        w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
        if w < FIGURE_MIN_PT or h < FIGURE_MIN_PT:
            continue
        if (w * h) > FIGURE_MAX_AREA * page_size[0] * page_size[1]:
            continue
        if any(_overlap_frac(bbox, tb) > 0.5 for tb in table_bboxes):
            continue
        try:
            pix = page.get_pixmap(clip=fitz.Rect(*bbox), matrix=fitz.Matrix(2, 2))
            rel = assets_dir / f"{doc_id}_p{pno}_fig{i}.png"
            rel.parent.mkdir(parents=True, exist_ok=True)
            pix.save(str(rel))
        except Exception:
            continue
        items.append(ParsedItem(
            doc_id=doc_id, page_number=pno, chunk_type="figure", text="",
            bbox=bbox, page_size=page_size, section=section,
            asset_path=str(rel), order=order,
        ))
        order += 1

    # -- figures: clustered vector drawings (charts drawn as paths) ----------
    if os.environ.get("MMRAG_DETECT_VECTOR_FIGURES", "1") not in {"0", "false"}:
        for i, bbox in enumerate(_vector_chart_clusters(page, page_size, table_bboxes)):
            if any(_overlap_frac(bbox, it.bbox) > 0.3 for it in items):
                continue
            try:
                pix = page.get_pixmap(clip=fitz.Rect(*bbox), matrix=fitz.Matrix(2, 2))
                rel = assets_dir / f"{doc_id}_p{pno}_vecfig{i}.png"
                rel.parent.mkdir(parents=True, exist_ok=True)
                pix.save(str(rel))
            except Exception:
                continue
            items.append(ParsedItem(
                doc_id=doc_id, page_number=pno, chunk_type="figure", text="",
                bbox=bbox, page_size=page_size, section=section,
                asset_path=str(rel), meta={"source": "vector-drawings"}, order=order,
            ))
            order += 1

    # -- text blocks ----------------------------------------------------------
    blocks = [b for b in page_dict.get("blocks", []) if b.get("type") == 0]
    for block in blocks:
        bbox = tuple(block.get("bbox", ()))
        if len(bbox) != 4:
            continue
        if any(_overlap_frac(bbox, tb) > 0.5 for tb in table_bboxes):
            continue  # belongs to a table
        txt = _join_block_text(block)
        if len(re.findall(r"[A-Za-z0-9]", txt)) < 3:
            continue
        heading = _is_heading(block, body_size)
        if heading:
            section = txt
        items.append(ParsedItem(
            doc_id=doc_id, page_number=pno, chunk_type="text", text=txt,
            bbox=bbox, page_size=page_size, section=section, order=order,
            meta={"is_heading": heading},
        ))
        order += 1

    items.sort(key=lambda it: (it.bbox[1], it.bbox[0]))
    for i, it in enumerate(items):
        it.order = order_start + i

    # captions for figures (text block just below the image)
    for it in items:
        if it.chunk_type != "figure":
            continue
        below = [
            t for t in items
            if t.chunk_type == "text"
            and t.bbox[1] >= it.bbox[3] - 4
            and t.bbox[1] - it.bbox[3] <= CAPTION_MAX_DIST
            and min(t.bbox[2], it.bbox[2]) - max(t.bbox[0], it.bbox[0]) > 0.4 * (it.bbox[2] - it.bbox[0])
        ]
        if below:
            cap = min(below, key=lambda t: t.bbox[1]).text
            if CAPTION_RE.match(cap) or len(cap) < 200:
                it.caption = cap[:300]

    next_section = section
    return items, next_section, order


def _vector_chart_clusters(
    page: fitz.Page,
    page_size: tuple[float, float],
    table_bboxes: list[tuple],
) -> list[tuple[float, float, float, float]]:
    """Cluster vector drawing segments into chart-like regions."""
    rects = []
    for d in page.get_drawings():
        r = d.get("rect")
        if r is None or r.width < 6 or r.height < 6:
            continue
        if d.get("fill") is None and d.get("color") is None:
            continue
        rects.append((r.x0, r.y0, r.x1, r.y1))
    if len(rects) < VECTOR_CLUSTER_MIN_SEGS:
        return []

    # greedy union-find style merging of nearby rects
    clusters: list[list[tuple]] = [[r] for r in rects]
    changed = True
    while changed:
        changed = False
        merged: list[list[tuple]] = []
        for cl in clusters:
            bx0 = min(r[0] for r in cl); by0 = min(r[1] for r in cl)
            bx1 = max(r[2] for r in cl); by1 = max(r[3] for r in cl)
            for other in merged:
                ox0 = min(r[0] for r in other); oy0 = min(r[1] for r in other)
                ox1 = max(r[2] for r in other); oy1 = max(r[3] for r in other)
                if not (bx1 + 14 < ox0 or ox1 + 14 < bx0 or by1 + 14 < oy0 or oy1 + 14 < by0):
                    other.extend(cl)
                    changed = True
                    break
            else:
                merged.append(cl)
        clusters = merged

    page_area = page_size[0] * page_size[1]
    out = []
    for cl in clusters:
        if len(cl) < VECTOR_CLUSTER_MIN_SEGS:
            continue
        bx0 = min(r[0] for r in cl); by0 = min(r[1] for r in cl)
        bx1 = max(r[2] for r in cl); by1 = max(r[3] for r in cl)
        area = (bx1 - bx0) * (by1 - by0)
        frac = area / page_area
        if not (VECTOR_MIN_AREA <= frac <= VECTOR_MAX_AREA):
            continue
        aspect = (bx1 - bx0) / max(by1 - by0, 1e-9)
        if not (0.6 <= aspect <= 4.0):
            continue
        if any(_overlap_frac((bx0, by0, bx1, by1), tb) > 0.4 for tb in table_bboxes):
            continue
        out.append((bx0, by0, bx1, by1))
    return out


# --------------------------------------------------------------------------- #
# tables (pdfplumber)
# --------------------------------------------------------------------------- #

def _cell_to_md(cell) -> str:
    if cell is None:
        return ""
    txt = re.sub(r"\s+", " ", str(cell)).strip()
    return txt.replace("|", "\\|")


def _table_to_markdown(grid: list[list]) -> str:
    if not grid:
        return ""
    width = max(len(r) for r in grid)
    norm = [list(r) + [""] * (width - len(r)) for r in grid]
    header = [_cell_to_md(c) for c in norm[0]]
    body = [[_cell_to_md(c) for c in row] for row in norm[1:]]
    # drop fully empty rows
    rows = [r for r in body if any(r)] or body
    lines = ["| " + " | ".join(header) + " |",
             "| " + " | ".join(["---"] * width) + " |"]
    lines += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(lines)


def _stitch_rows(tables) -> list[dict]:
    """Reassemble vertically-split tables.

    Financial PDFs (especially HTML printed to PDF) often have horizontal
    rules but no vertical rules, so pdfplumber returns one "table" per row.
    Rows with the same horizontal span and small vertical gaps are stitched
    back into a single grid.
    """
    entries = []
    for tbl in tables:
        try:
            grid = tbl.extract()
        except Exception:
            continue
        if not grid:
            continue
        entries.append({"bbox": tuple(float(v) for v in tbl.bbox), "rows": grid})
    entries.sort(key=lambda e: (e["bbox"][1], e["bbox"][0]))

    clusters: list[list[dict]] = []
    for e in entries:
        placed = False
        for cl in clusters:
            b = cl[-1]["bbox"]
            w = min(e["bbox"][2] - e["bbox"][0], b[2] - b[0])
            h_ov = min(e["bbox"][2], b[2]) - max(e["bbox"][0], b[0])
            gap = e["bbox"][1] - b[3]
            if w > 0 and h_ov >= 0.6 * w and -4 <= gap <= 26:
                cl.append(e)
                placed = True
                break
        if not placed:
            clusters.append([e])

    out = []
    for cl in clusters:
        rows = [r for e in cl for r in e["rows"]]
        out.append({
            "bbox": (min(e["bbox"][0] for e in cl),
                     min(e["bbox"][1] for e in cl),
                     max(e["bbox"][2] for e in cl),
                     max(e["bbox"][3] for e in cl)),
            "rows": rows,
        })
    return out


def _grid_quality(rows: list[list]) -> bool:
    """Reject fragments: need >=2 rows, >=2 columns and real content."""
    if len(rows) < 2:
        return False
    width = max(len(r) for r in rows)
    if width < 2:
        return False
    filled = sum(1 for r in rows if any(_cell_to_md(c) for c in r))
    return filled >= 2


# Fraction of non-label cells that must be non-empty for a table to be trusted
# as a real extraction. HTML-printed financial PDFs (horizontal rules only, no
# vertical rules) make pdfplumber emit label-only "shells" whose data columns
# are empty; the actual numbers then only exist in the PyMuPDF text stream.
# We only "claim" a page region for a table when it genuinely captured its
# data, so the overlapping PyMuPDF text (which carries the numbers) survives.
_TABLE_DATA_COVERAGE = 0.5


def _table_data_coverage(rows: list[list]) -> float:
    """Fraction of non-first-column body cells that are non-empty.

    First column holds row labels; the remaining columns hold the data. A
    well-extracted grid fills most of them; a label-only shell leaves them empty.
    """
    if not rows:
        return 0.0
    width = max(len(r) for r in rows)
    if width < 2:
        return 0.0
    total = 0
    filled = 0
    for r in rows:
        padded = list(r) + [""] * (width - len(r))
        for c in padded[1:]:          # skip label column
            total += 1
            if _cell_to_md(c):
                filled += 1
    return filled / total if total else 0.0


def _extract_tables(page_plumber, doc_id: str, pno: int,
                    page_size: tuple[float, float], section: str,
                    order_start: int) -> tuple[list[ParsedItem], list[tuple], int]:
    items: list[ParsedItem] = []
    bboxes: list[tuple] = []
    order = order_start
    try:
        found = page_plumber.find_tables()
    except Exception:
        found = []
    for ti, tbl in enumerate(_stitch_rows(found)):
        grid = tbl["rows"]
        if not _grid_quality(grid):
            continue
        # A label-only "shell" (data columns empty) is a failed extraction on
        # rules-only financial PDFs. Do NOT emit it and do NOT let it claim the
        # region: the real numbers live in the PyMuPDF text stream and must be
        # kept. Skipping the bbox here is what preserves them.
        if _table_data_coverage(grid) < _TABLE_DATA_COVERAGE:
            continue
        md = _table_to_markdown(grid)
        if len(md) < 40:
            continue
        bbox = tbl["bbox"]
        bboxes.append(bbox)
        items.append(ParsedItem(
            doc_id=doc_id, page_number=pno, chunk_type="table", text=md,
            bbox=bbox, page_size=page_size, section=section, order=order,
            meta={"n_rows": len(grid), "n_cols": len(grid[0]) if grid else 0},
        ))
        order += 1
    return items, bboxes, order


# --------------------------------------------------------------------------- #
# public API
# --------------------------------------------------------------------------- #

def parse_pdf(pdf_path: Path, assets_dir: Path, pages_dir: Path,
              render_dpi: int = 144) -> tuple[dict, list[ParsedItem]]:
    pdf_path = Path(pdf_path)
    doc_id = _doc_id(pdf_path)
    assets_dir.mkdir(parents=True, exist_ok=True)
    pages_dir.mkdir(parents=True, exist_ok=True)

    items: list[ParsedItem] = []
    page_sizes: dict[str, list[float]] = {}
    section = ""

    with fitz.open(str(pdf_path)) as fitz_doc, pdfplumber.open(str(pdf_path)) as plumb_doc:
        body_size = _median_body_size(fitz_doc)
        zoom = render_dpi / 72.0
        for pno in range(fitz_doc.page_count):
            fpage = fitz_doc[pno]
            page_size = (float(fpage.rect.width), float(fpage.rect.height))
            page_sizes[str(pno + 1)] = [page_size[0], page_size[1]]

            # rendered page PNG for bbox-overlay frontend (skip when cached)
            page_png = pages_dir / f"{doc_id}_p{pno + 1}.png"
            if not page_png.exists():
                try:
                    pix = fpage.get_pixmap(matrix=fitz.Matrix(zoom, zoom))
                    pix.save(str(page_png))
                except Exception:
                    pass

            ppage = plumb_doc.pages[pno] if pno < len(plumb_doc.pages) else None
            if ppage is not None:
                tab_items, table_bboxes, order = _extract_tables(
                    ppage, doc_id, pno + 1, page_size, section, 0)
            else:
                tab_items, table_bboxes, order = [], [], 0

            txt_items, section, _ = _extract_text_and_raster(
                fpage, doc_id, pno + 1, page_size, body_size,
                table_bboxes, assets_dir, section, 0)

            page_items = tab_items + txt_items
            page_items.sort(key=lambda it: (it.bbox[1], it.bbox[0]))
            for i, it in enumerate(page_items):
                it.order = i
            items.extend(page_items)

    meta = {
        "doc_id": doc_id,
        "filename": pdf_path.name,
        "path": str(pdf_path),
        "title": (fitz.open(str(pdf_path)).metadata or {}).get("title") or pdf_path.stem,
        "num_pages": len(page_sizes),
        "page_sizes": page_sizes,
        "n_items": len(items),
    }
    return meta, items


def parse_corpus(corpus_dir: Path, assets_dir: Path, pages_dir: Path,
                 limit: int | None = None) -> tuple[list[dict], list[ParsedItem]]:
    pdfs = sorted(Path(corpus_dir).glob("*.pdf"))
    if limit:
        pdfs = pdfs[:limit]
    metas: list[dict] = []
    all_items: list[ParsedItem] = []
    for pdf in pdfs:
        meta, items = parse_pdf(pdf, assets_dir, pages_dir)
        metas.append(meta)
        all_items.extend(items)
    return metas, all_items
