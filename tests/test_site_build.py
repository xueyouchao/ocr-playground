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
    (tmp_path / "data" / "records").mkdir(parents=True)
    (tmp_path / "site").mkdir()
    (tmp_path / "data" / "systems.json").write_text(json.dumps(SYSTEMS), encoding="utf-8")
    for name, pages in records.items():
        (tmp_path / "data" / "records" / f"{name}.json").write_text(
            json.dumps(pages), encoding="utf-8"
        )
    (tmp_path / "site" / "template.html").write_text(TEMPLATE, encoding="utf-8")

    site_collect.main(tmp_path)
    site_build.main(tmp_path)
    return tmp_path


def test_the_offline_comparison_opens_with_chandra_and_nemotron() -> None:
    systems = json.loads((ROOT / "data" / "systems.json").read_text(encoding="utf-8"))

    assert [system["id"] for system in systems[:2]] == ["chandra-2", "nemotron-parse"]


def test_the_two_cpu_systems_render_last() -> None:
    systems = json.loads((ROOT / "data" / "systems.json").read_text(encoding="utf-8"))

    assert [system["id"] for system in systems[-2:]] == [
        "liteparse-tesseract", "paddleocr-structure",
    ]


def test_evidence_joins_records_against_the_roster(tmp_path) -> None:
    out = build_site(tmp_path, {
        "a": {"page-1": "one"},
        "b": {"page-1": "<b>two</b>", "page-2": "3 4"},
    })
    evidence = json.loads((out / "site" / "evidence.json").read_text(encoding="utf-8"))

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
    pages = json.loads((out / "site" / "evidence.json").read_text(encoding="utf-8"))["pages"]

    assert pages[0]["cells"]["a"]["text"] == "one"
    assert pages[0]["cells"]["a"]["chars"] == 3
    assert pages[1]["cells"]["a"] == {
        "text": None, "chars": None, "tokens": None, "numeric_share": None,
    }


def test_counts_travel_with_the_text(tmp_path) -> None:
    out = build_site(tmp_path, {"a": {"page-1": "abc 12 34.5"}})
    cell = json.loads((out / "site" / "evidence.json").read_text(encoding="utf-8"))["pages"][0]["cells"]["a"]
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
    page = (out / "site" / "index.html").read_text(encoding="utf-8")

    assert page.startswith("<script>const DATA = ")
    assert "</script><script>alert(1)" not in page, "parser output must not close the tag it lives in"
    assert "\\u003c/script>" in page
    encoded = page[len("<script>const DATA = "):page.index(";</script>")]
    assert json.loads(encoded)["pages"][0]["cells"]["a"]["text"].startswith("</script>")


def test_the_static_measure_is_the_domain_measure() -> None:
    for text in ("", "abc 12", "<td>x</td>", "1,234.5 6/7"):
        assert site_collect.measure(text) == ReadingStats.from_text(text).as_dict()
    assert site_collect.measure(None) == {"chars": None, "tokens": None, "numeric_share": None}


@pytest.mark.parametrize("shim", ["scripts/collect.py", "scripts/build.py", "scripts/prepare.py", "api/server.py"])
def test_every_entry_shim_imports_cleanly(shim) -> None:
    # `run_name="not_main"` imports the shim without running its main(); this is what proves the
    # sys.path juggling still finds the package when the script is run from its own directory.
    result = subprocess.run(
        [sys.executable, "-c", f"import runpy; runpy.run_path({shim!r}, run_name='not_main')"],
        cwd=ROOT, capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("shim", ["scripts/prepare.py", "api/server.py"])
def test_the_cli_shims_answer_help(shim) -> None:
    result = subprocess.run([sys.executable, shim, "--help"], cwd=ROOT,
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert "usage" in result.stdout.lower()


def test_the_page_warms_the_gpu_on_intent_not_on_arrival() -> None:
    """A warm GPU bills from boot until it scales down, so a page view alone must not rent one."""
    template = (ROOT / "site" / "template.html").read_text()
    assert 'const WARM_API = "/api/warm";' in template
    assert '"pointerenter", warmGpu' in template
    assert '"focusin", warmGpu' in template
    # A cooldown, not a one-shot flag: the page outlives the GPU, and the next visit must warm again.
    assert "WARM_COOLDOWN_MS" in template
    assert "warmGpu, { once: true }" not in template

    body = template[template.index("async function loadModes"):]
    assert "warmGpu" not in body[: body.index("\n}\n")], "page load must not start the GPU"


def test_the_picker_can_ask_for_a_comparison() -> None:
    """Several parsers at once: checkboxes, a comma-joined header, and a footer that names them all."""
    template = (ROOT / "site" / "template.html").read_text()

    assert 'id="mode-checks"' in template
    assert template.count('type="checkbox"') >= 2
    assert '"X-Mode": modes.join(",")' in template
    # An empty picker is an upload with nothing to read, so the last box cannot be unticked.
    assert "box.checked = true;" in template
    # Each ticked GPU parser is warmed, not just one: a comparison pays every cold start it asked for.
    assert "for (const mode of state.modes)" in template
    assert "warmedAt[mode]" in template


def test_the_mode_fallback_only_picks_a_parser_that_can_run() -> None:
    """Two ways the picker could quietly fight the visitor; both are one line each."""
    template = (ROOT / "site" / "template.html").read_text()
    body = template[template.index("async function loadModes"):]
    body = body[: body.index("\n}\n")]

    # Every option is still enabled until applyModes() runs, so the DOM cannot answer "what can
    # this host run" — an unavailable default would win and every upload would 503.
    assert "option.disabled" not in body, "ask MODE_AVAILABLE, not the DOM"
    assert "find((mode) => MODE_AVAILABLE[mode.id])" in body

    # A /api/health reply that lands after the visitor picked must not undo the pick.
    assert "if (!userChoseMode && MODE_AVAILABLE[health.default_mode])" in body
    assert "userChoseMode = true;" in template


def test_the_night_theme_is_chosen_before_the_first_paint() -> None:
    """Night must not start as a flash of white: the palette is picked in <head>, not on load."""
    template = (ROOT / "site" / "template.html").read_text()
    head = template[: template.index("</head>")]

    assert 'document.documentElement.dataset.theme' in head
    assert "prefers-color-scheme: dark" in head, "a machine already in night mode is the default"


def test_the_night_theme_is_a_palette_swap_not_a_second_page() -> None:
    template = (ROOT / "site" / "template.html").read_text()

    assert ':root[data-theme="dark"]' in template
    # Every surface the day theme paints white is a variable; one hard-coded white card is a lit
    # rectangle at midnight, and it is exactly the kind that gets added later.
    assert "background: #fff" not in template
    assert 'id="theme"' in template
    # A remembered choice, with the OS as the fallback rather than a one-way switch.
    assert "localStorage.setItem" in template
    assert "storedTheme()" in template


def test_the_night_theme_reaches_into_the_previews() -> None:
    """A preview is its own document, so it inherits neither the palette nor the mark ink."""
    template = (ROOT / "site" / "template.html").read_text()

    assert "PREVIEW_DARK_STYLE" in template
    assert "${previewStyle()}" in template
    # The marks inside a frame are inline styles: `color: inherit` would be light text on a light mark.
    assert "FIND_MARKS" in template
    assert "color:#fff6d8" in template


def test_a_refused_storage_write_cannot_freeze_the_toggle() -> None:
    """The applied theme is held in memory; storage is only how it survives a reload."""
    template = (ROOT / "site" / "template.html").read_text()
    body = template[template.index("function currentTheme"):]
    body = body[: body.index("\n}\n")]

    # Reading the theme back out of storage would answer with the OS preference whenever a write was
    # refused -- the page changes, the previews keep the old palette, and every click asks for the
    # same theme again.
    assert "chosenTheme" in body, "the current theme must not be read back from storage"
    assert "localStorage" not in body


def test_every_record_writer_agrees_with_the_build() -> None:
    """One module names the layout: a runner writing elsewhere would publish stale runs silently."""
    from ocr_playground.interfaces.site.layout import data
    from runners._common import RECORDS

    assert RECORDS == data(ROOT) / "records"
