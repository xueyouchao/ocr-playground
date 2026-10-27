"""Merge every recorded run plus the roster into one evidence file for the page.

The counts come from the domain's `ReadingStats`, the same object an upload is measured with, so the
bundled panes and the live pane cannot drift apart.
"""
from __future__ import annotations

import json
from pathlib import Path

from ...domain.reading import ReadingStats
from .layout import data, site


def measure(text: str | None) -> dict:
    """Describe the text. Counts only — there is no score here on purpose.

    The dictionary share that used to live here rewarded prose and punished numeric tables: on this
    document its denominator ranged from 2 to 187 words, and on page 4 (77% numbers, and the rest
    column abbreviations like "CLD" and "TDME") it scored a correct reading at 30%. A number that
    wrong is worse than no number, so it is gone rather than tuned. Accuracy needs ground truth.
    """
    if text is None:
        return {"chars": None, "tokens": None, "numeric_share": None}
    return ReadingStats.from_text(text).as_dict()


def main(root: Path) -> None:
    systems = json.loads((data(root) / "systems.json").read_text(encoding="utf-8"))
    recorded = {}
    for path in sorted((data(root) / "records").glob("*.json")):
        recorded[path.stem] = json.loads(path.read_text(encoding="utf-8"))

    page_count = max((len(pages) for pages in recorded.values()), default=0)
    pages = []
    for number in range(1, page_count + 1):
        key = f"page-{number}"
        cells = {}
        for system in systems:
            text = recorded.get(system["id"], {}).get(key)
            cells[system["id"]] = {"text": text, **measure(text)}
        pages.append({"page": number, "cells": cells})

    evidence = {
        "document": {
            "name": "efda8b60…a8a9.pdf",
            "pages": page_count,
            "note": "One printed page, five handwritten forms, from the demo corpus.",
        },
        "systems": [{k: s[k] for k in ("id", "name", "kind", "pipeline", "device", "needs")}
                    for s in systems],
        "pages": pages,
    }
    (site(root) / "evidence.json").write_text(
        json.dumps(evidence, indent=1, ensure_ascii=False), encoding="utf-8"
    )
    for system in systems:
        values = [page["cells"][system["id"]]["chars"] for page in pages]
        ran = [value for value in values if value is not None]
        print(
            f'{system["id"]:<22} '
            + (f"ran on {len(ran)}/{page_count} pages" if ran else "not run")
        )
