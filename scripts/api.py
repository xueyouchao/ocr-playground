"""Compatibility entry point for `make api` and the container's CMD.

The service lives in `ocr_playground.interfaces.http`; this file only puts `src/` on
`sys.path` (running `python scripts/api.py` puts `scripts/` there instead) and delegates.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ocr_playground.interfaces.http import main  # noqa: E402 - the path above has to be set first

if __name__ == "__main__":
    main()
