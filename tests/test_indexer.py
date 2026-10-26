"""The indexer's configuration and projections.

The CocoIndex plumbing itself is exercised end to end by `test_indexer_run.py`; these tests cover the
parts that must be right before a run can start.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from fakes import FAST, GLM, FakeEngine, registry
from ocr_playground.interfaces.indexer.config import ConfigError, IndexerConfig
from ocr_playground.interfaces.indexer.projection import page_rows, payload


def test_defaults_stay_on_the_cpu_path_without_an_endpoint(tmp_path) -> None:
    config = IndexerConfig.from_env({
        "OCR_CORPUS": str(tmp_path), "OCR_OUT": "out", "OCR_DB": "cocoindex.db",
    })
    assert config.mode == "fast"
    assert config.max_pages == 0, "an index run reads every page unless told otherwise"
    assert config.corpus == tmp_path.resolve()
    assert config.out == Path("out")


def test_an_endpoint_selects_glm_but_an_explicit_mode_wins(tmp_path) -> None:
    env = {"OCR_CORPUS": str(tmp_path), "GLM_OCR_BASE_URL": "https://gpu.example/v1"}
    assert IndexerConfig.from_env(env).mode == "glm"
    assert IndexerConfig.from_env(env | {"OCR_MODE": " FAST "}).mode == "fast"


def test_an_unknown_mode_names_the_choices(tmp_path) -> None:
    config = IndexerConfig.from_env({"OCR_CORPUS": str(tmp_path), "OCR_MODE": "nope"})
    with pytest.raises(ConfigError, match="OCR_MODE='nope' is not an extraction mode; use glm or fast"):
        config.validate(registry(), {"glm": FakeEngine(GLM), "fast": FakeEngine(FAST)})


def test_glm_without_an_endpoint_is_a_configuration_error(tmp_path) -> None:
    config = IndexerConfig.from_env({"OCR_CORPUS": str(tmp_path), "OCR_MODE": "glm"})
    with pytest.raises(ConfigError, match="OCR_MODE=glm needs GLM_OCR_BASE_URL"):
        config.validate(registry(), {"glm": FakeEngine(GLM, available=False),
                                     "fast": FakeEngine(FAST)})


def test_a_missing_corpus_is_named_before_the_run(tmp_path) -> None:
    config = IndexerConfig.from_env({"OCR_CORPUS": str(tmp_path / "nope")})
    with pytest.raises(ConfigError, match="is not a directory"):
        config.validate(registry(), {"glm": FakeEngine(GLM), "fast": FakeEngine(FAST)})


def test_a_negative_page_limit_is_refused(tmp_path) -> None:
    config = IndexerConfig.from_env({"OCR_CORPUS": str(tmp_path), "OCR_MAX_PAGES": "-1"})
    with pytest.raises(ConfigError, match="OCR_MAX_PAGES must be 0"):
        config.validate(registry(), {"glm": FakeEngine(GLM), "fast": FakeEngine(FAST)})


def test_a_valid_run_passes_validation(tmp_path) -> None:
    config = IndexerConfig.from_env({"OCR_CORPUS": str(tmp_path), "OCR_MODE": "fast",
                                     "OCR_MAX_PAGES": "2"})
    config.validate(registry(), {"glm": FakeEngine(GLM), "fast": FakeEngine(FAST)})
    assert config.max_pages == 2


def _document(path: str, digest: str, *, engine: str, pages: list[tuple[int, str]]) -> dict:
    return {
        "document": {"path": path, "digest": digest, "name": path, "note": "n", "pages": len(pages)},
        "systems": [{"id": engine}],
        "pages": [
            {
                "page": number,
                "cells": {engine: {"text": text, "chars": len(text), "tokens": 1,
                                   "numeric_share": 0.5}},
            }
            for number, text in pages
        ],
    }


def test_page_rows_are_flat_sorted_and_carry_provenance() -> None:
    rows = page_rows([
        _document("b.pdf", "d2", engine="liteparse-tesseract", pages=[(1, "b1")]),
        _document("a.pdf", "d1", engine="liteparse-tesseract", pages=[(2, "a2"), (1, "a1")]),
    ])

    assert [(row["document"], row["page"]) for row in rows] == [
        ("a.pdf", 1), ("a.pdf", 2), ("b.pdf", 1),
    ]
    assert rows[0] == {
        "document": "a.pdf",
        "digest": "d1",
        "page": 1,
        "engine": "liteparse-tesseract",
        "chars": 2,
        "tokens": 1,
        "numeric_share": 0.5,
        "text": "a1",
    }


def test_payload_is_stable_json() -> None:
    assert payload({"b": 1, "a": 2}) == '{\n  "a": 2,\n  "b": 1\n}\n', (
        "sorted keys keep re-runs byte-identical"
    )
