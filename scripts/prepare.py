"""Entry point: `python scripts/prepare.py`, and `make pages`."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ocr_playground.interfaces.site.prepare import main  # noqa: E402

if __name__ == "__main__":
    main(ROOT)
