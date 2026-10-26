"""The static build: records + roster in, one evidence file and one page out.

`collect.main` and `build.main` take the repository root as an argument, so a test builds a scratch
site instead of rewriting the real one.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from conftest import ROOT
from ocr_playground.domain.reading import ReadingStats
from ocr_playground.interfaces.site import build as site_build
from ocr_playground.interfaces.site import collect as site_collect

SYSTEMS = [
    {
        "id": "a", "name": "System A", "kind": "baseline",
        "pipeline": "does a thing", "runner": "runners/a.py", "needs": "nothing", "device": "CPU",
    },
    {
        "id": "b", "name": "System B", "kind": "vlm",
        "pipeline": "does another", "runner": "runners/b.py", "needs": "a GPU", "device": "GPU",
    },
]

TEMPLATE = "<script>const DATA = __EVIDENCE__;</script>"


def build_site(tmp_path: Path, records: dict[str, dict[str, str | None]]) -> Path:
    (tmp_path / "records").mkdir()
    (tmp_path / "systems.json").write_text(json.dumps(SYSTEMS), encoding="utf-8")
    for name, pages in records.items():
        (tmp_path / "records" / f"{name}.json").write_text(json.dumps(pages), encoding="utf-8")
    (tmp_path / "template.html").write_text(TEMPLATE, encoding="utf-8")

    site_collect.main(tmp_path)
    site_build.main(tmp_path)
    return tmp_path


def test_evidence_joins_records_against_the_roster(tmp_path) -> None:
    out = build_site(tmp_path, {
        "a": {"page-1": "one"},
        "b": {"page-1": "<b>two</b>", "page-2": "3 4"},
    })
    evidence = json.loads((out / "evidence.json").read_text(encoding="utf-8"))

    assert evidence["document"] == {
        "name": "efda8b60…a8a9.pdf",
        "pages": 2,
        "note": "One printed page, five handwritten forms, from the demo corpus.",
    }
    assert evidence["systems"] == [
        {key: system[key] for key in ("id", "name", "kind", "pipeline", "device", "needs")}
        for system in SYSTEMS
    ]
    assert [page["page"] for page in evidence["pages"]] == [1, 2]


def test_a_parser_that_did_not_run_is_null_not_zero(tmp_path) -> None:
    out = build_site(tmp_path, {"a": {"page-1": "one"}, "b": {"page-1": "two", "page-2": "three"}})
    pages = json.loads((out / "evidence.json").read_text(encoding="utf-8"))["pages"]

    assert pages[0]["cells"]["a"]["text"] == "one"
    assert pages[0]["cells"]["a"]["chars"] == 3
    assert pages[1]["cells"]["a"] == {
        "text": None, "chars": None, "tokens": None, "numeric_share": None,
    }


def test_counts_travel_with_the_text(tmp_path) -> None:
    out = build_site(tmp_path, {"a": {"page-1": "abc 12 34.5"}})
    cell = json.loads((out / "evidence.json").read_text(encoding="utf-8"))["pages"][0]["cells"]["a"]
    assert cell["chars"] == 11
    assert cell["tokens"] == 3
    assert cell["numeric_share"] == 0.667


def test_collect_says_which_parsers_ran_where(tmp_path, capsys) -> None:
    build_site(tmp_path, {"a": {"page-1": "one"}, "b": {"page-1": "two", "page-2": "three"}})
    output = capsys.readouterr().out
    assert "a                      ran on 1/2 pages" in output
    assert "b                      ran on 2/2 pages" in output


def test_build_inlines_the_evidence_and_escapes_angle_brackets(tmp_path) -> None:
    out = build_site(tmp_path, {"a": {"page-1": "</script><script>alert(1)</script>"}})
    page = (out / "index.html").read_text(encoding="utf-8")

    assert page.startswith("<script>const DATA = ")
    assert "</script><script>alert(1)" not in page, "parser output must not close the tag it lives in"
    assert "\\u003c/script>" in page
    encoded = page[len("<script>const DATA = "):page.index(";</script>")]
    assert json.loads(encoded)["pages"][0]["cells"]["a"]["text"].startswith("</script>")


def test_the_static_measure_is_the_domain_measure() -> None:
    for text in ("", "abc 12", "<td>x</td>", "1,234.5 6/7"):
        assert site_collect.measure(text) == ReadingStats.from_text(text).as_dict()
    assert site_collect.measure(None) == {"chars": None, "tokens": None, "numeric_share": None}


@pytest.mark.parametrize("shim", ["collect.py", "build.py", "prepare.py", "api/server.py"])
def test_every_entry_shim_imports_cleanly(shim) -> None:
    # `run_name="not_main"` imports the shim without running its main(); this is what proves the
    # sys.path juggling still finds the package when the script is run from its own directory.
    result = subprocess.run(
        [sys.executable, "-c", f"import runpy; runpy.run_path({shim!r}, run_name='not_main')"],
        cwd=ROOT, capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("shim", ["prepare.py", "api/server.py"])
def test_the_cli_shims_answer_help(shim) -> None:
    result = subprocess.run([sys.executable, shim, "--help"], cwd=ROOT,
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert "usage" in result.stdout.lower()
