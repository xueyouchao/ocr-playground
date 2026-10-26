"""CocoIndex entry point for the corpus indexer.

    python -m venv .venv-index && .venv-index/bin/pip install -r requirements-cocoindex.txt
    .venv-index/bin/cocoindex update main.py

Environment: OCR_CORPUS, OCR_OUT, OCR_DB, OCR_MODE, OCR_MAX_PAGES — see indexer/flow.py and README.
"""
import cocoindex as coco

from indexer.flow import build_app

app = build_app()

if __name__ == "__main__":
    with coco.runtime():
        app.update_blocking(report_to_stdout=True)
