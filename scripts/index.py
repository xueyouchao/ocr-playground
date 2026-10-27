"""CocoIndex entry point for the corpus indexer.

    python -m venv .venv-index && .venv-index/bin/pip install -r requirements-cocoindex.txt
    .venv-index/bin/cocoindex update scripts/index.py

The app itself lives in `ocr_playground/interfaces/indexer/flow.py`; this file is the handle
CocoIndex points its update at. Environment: OCR_CORPUS, OCR_OUT, OCR_DB, OCR_MODE, OCR_MAX_PAGES —
see that module and the README.
"""
from __future__ import annotations

import sys
from pathlib import Path

import cocoindex as coco

ROOT = Path(__file__).resolve().parents[1]
# The importable tree lives under src/; the repository root holds data, site and scripts.
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ocr_playground.interfaces.indexer.flow import build_app  # noqa: E402

app = build_app()

if __name__ == "__main__":
    with coco.runtime():
        app.update_blocking(report_to_stdout=True)
