"""Persistence helpers shared by the evaluation harness."""

from __future__ import annotations

import json
from pathlib import Path

from ..config import Settings
from ..ingestion.chunker import Chunk  # noqa: F401  (type parity)
from ..ingestion.parser import ParsedItem


def load_docs_meta(settings: Settings) -> list[dict]:
    path = settings.index_dir / "docs.json"
    if not path.exists():
        return []
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return data if isinstance(data, list) else data.get("docs", [])


def load_parsed_items(settings: Settings) -> list[ParsedItem]:
    path = settings.index_dir / "parsed_items.jsonl"
    if not path.exists():
        return []
    out: list[ParsedItem] = []
    with Path(path).open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            d["bbox"] = tuple(d.get("bbox", (0, 0, 0, 0)))
            d["page_size"] = tuple(d.get("page_size", (0, 0)))
            out.append(ParsedItem(**d))
    return out


def save_parsed_items(items: list[ParsedItem], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("w", encoding="utf-8") as fh:
        for it in items:
            fh.write(json.dumps(it.to_dict(), ensure_ascii=False) + "\n")
