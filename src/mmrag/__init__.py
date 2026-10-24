"""Multimodal Hybrid RAG Engine (mmrag)."""

__version__ = "0.1.0"

from pathlib import Path
import sys

try:
    _here = Path(__file__).resolve().parent
except Exception:
    _here = Path.cwd()


def _ensure_src_on_path():
    parent = _here.parent
    if str(parent) not in sys.path:
        sys.path.insert(0, str(parent))


_ensure_src_on_path()
