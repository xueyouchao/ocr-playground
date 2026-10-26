"""End to end: a real CocoIndex update over a scratch corpus, then an unchanged second run.

Skipped when the optional toolchain is absent — the indexer is an extra, and so are poppler and
LiteParse. When they are present, this is the test that proves memoisation actually happens.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from conftest import ROOT
from ocr_playground.infrastructure.settings import Settings

COCOINDEX = Path(sys.executable).with_name("cocoindex")
LIT = Path(Settings.from_env({}).lit_path)

pytestmark = pytest.mark.skipif(
    not (COCOINDEX.exists() and shutil.which("pdftoppm") and LIT.exists()),
    reason="needs the cocoindex CLI, poppler and the lit CLI",
)


def run_update(env: dict[str, str]) -> subprocess.CompletedProcess:
    return subprocess.run(
        [str(COCOINDEX), "update", "main.py"],
        cwd=ROOT, env=env, capture_output=True, text=True, timeout=300,
    )


def test_a_corpus_is_read_incrementally(tmp_path) -> None:
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    subprocess.run(
        ["pdftoppm", "-png", "-r", "120", "-f", "1", "-l", "1", "-singlefile",
         str(ROOT / "corpus" / "document.pdf"), str(corpus / "form")],
        check=True,
    )
    (corpus / "notes.txt").write_text("not a document\n")
    out = tmp_path / "out"
    env = {
        **os.environ,
        "OCR_CORPUS": str(corpus),
        "OCR_OUT": str(out),
        "OCR_DB": str(tmp_path / "cocoindex.db"),
        "OCR_MODE": "fast",
        "OCR_MAX_PAGES": "1",
        "TMPDIR": "/tmp",
    }
    env.pop("GLM_OCR_BASE_URL", None)

    first = run_update(env)
    assert first.returncode == 0, first.stdout + first.stderr

    summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    assert summary["documents"] == 1
    assert summary["read"] == 1
    assert summary["failed"] == 0
    assert summary["pages"] == 1
    assert summary["skipped"] == ["notes.txt"]
    assert summary["mode"] == "fast"
    assert summary["engine"].startswith("liteparse-tesseract lit=")
    assert "pipeline=" in summary["engine"]

    rows = [json.loads(line) for line in (out / "pages.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 1
    assert rows[0]["document"] == "form.png"
    assert rows[0]["engine"] == "liteparse-tesseract"
    assert rows[0]["chars"] > 0
    assert rows[0]["page"] == 1

    stored = json.loads((out / "documents" / "form.png.json").read_text(encoding="utf-8"))
    assert stored["document"]["path"] == "form.png"
    assert stored["document"]["digest"] == rows[0]["digest"]
    assert stored["document"]["engine_identity"] == summary["engine"]
    assert stored["document"]["pages"] == 1
    assert stored["pages"][0]["cells"]["liteparse-tesseract"]["text"] == rows[0]["text"]

    second = run_update(env)
    assert second.returncode == 0, second.stdout + second.stderr
    assert "unchanged" in (second.stdout + second.stderr), "an unchanged corpus must not be re-read"


def test_glm_without_an_endpoint_fails_before_reading(tmp_path) -> None:
    env = {
        **os.environ,
        "OCR_CORPUS": str(tmp_path),
        "OCR_OUT": str(tmp_path / "out"),
        "OCR_DB": str(tmp_path / "db"),
        "OCR_MODE": "glm",
        "TMPDIR": "/tmp",
    }
    env.pop("GLM_OCR_BASE_URL", None)
    result = run_update(env)

    assert result.returncode != 0
    assert "OCR_MODE=glm needs GLM_OCR_BASE_URL" in (result.stdout + result.stderr)
