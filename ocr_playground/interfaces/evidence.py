"""The evidence shape, and the one function that produces it.

This is a presentation contract, not a domain one. `records/<id>.json`, `evidence.json` and the
upload API's response all share three keys, which is why the page needs exactly one branch for
"bundled sample or live upload" (`state.uploaded || DATA`). Keeping the translation here means the
domain aggregate never has to know the word "cells".
"""
from __future__ import annotations

from ..domain.document import ReadDocument


def document_evidence(document: ReadDocument) -> dict:
    """One reading as the page renders it: document, systems, pages[].cells[engine].

    A comparison keeps one cell per system per page. An engine that did not answer still gets a
    cell — with no counts and the reason — because the page draws one pane per system, and a missing
    key would be a pane that cannot explain itself.
    """
    pages = []
    for page in document.pages:
        cells: dict = {}
        for engine in document.engines:
            reading = page.reading_by(engine.id)
            if reading is None:
                cells[engine.id] = {
                    "text": "",
                    "chars": None,
                    "tokens": None,
                    "numeric_share": None,
                    "error": page.failure_for(engine.id) or "no output",
                }
            else:
                cells[engine.id] = {"text": reading.text, **reading.stats.as_dict()}
        entry: dict = {"page": page.number, "cells": cells}
        if page.preview is not None:
            entry["preview"] = page.preview
        pages.append(entry)

    return {
        "document": {
            "name": document.display_name,
            "pages": document.page_count,
            "note": document.note,
        },
        "systems": [engine.as_dict() for engine in document.engines],
        "pages": pages,
    }
