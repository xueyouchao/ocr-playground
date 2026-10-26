"""The reading fingerprint that keeps the index honest across code changes."""
from __future__ import annotations

from pathlib import Path

from ocr_playground.infrastructure.fingerprint import reading_fingerprint


def make_tree(root: Path) -> None:
    (root / "pkg").mkdir()
    (root / "pkg" / "a.py").write_text("A = 1\n")
    (root / "pkg" / "b.py").write_text("B = 2\n")


def test_the_fingerprint_is_stable(tmp_path) -> None:
    make_tree(tmp_path)
    assert reading_fingerprint(tmp_path) == reading_fingerprint(tmp_path)
    assert len(reading_fingerprint(tmp_path)) == 12


def test_editing_a_reader_changes_the_fingerprint(tmp_path) -> None:
    make_tree(tmp_path)
    before = reading_fingerprint(tmp_path)
    (tmp_path / "pkg" / "a.py").write_text("A = 3\n")
    assert reading_fingerprint(tmp_path) != before


def test_renaming_a_reader_changes_the_fingerprint(tmp_path) -> None:
    make_tree(tmp_path)
    before = reading_fingerprint(tmp_path)
    (tmp_path / "pkg" / "a.py").rename(tmp_path / "pkg" / "c.py")
    assert reading_fingerprint(tmp_path) != before


def test_bytecode_caches_are_ignored(tmp_path) -> None:
    make_tree(tmp_path)
    before = reading_fingerprint(tmp_path)
    cache = tmp_path / "pkg" / "__pycache__"
    cache.mkdir()
    (cache / "stale.py").write_text("not source\n")
    assert reading_fingerprint(tmp_path) == before


def test_the_real_package_has_a_fingerprint() -> None:
    assert len(reading_fingerprint()) == 12
