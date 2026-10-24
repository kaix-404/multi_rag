"""Allow `python -m mmrag <cmd> ...`."""

from __future__ import annotations

import sys
from pathlib import Path

# When invoked via `python -m mmrag`, ensure the package root is on
# sys.path so intra-package imports like `from .cli import main` resolve.
here = Path(__file__).resolve().parent
if str(here.parent) not in sys.path:
    sys.path.insert(0, str(here.parent))

from .cli import main

if __name__ == "__main__":
    raise SystemExit(main())
