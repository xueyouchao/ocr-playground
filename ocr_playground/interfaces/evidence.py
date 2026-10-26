"""The evidence shape, and the one function that produces it.

This is a presentation contract, not a domain one. `records/<id>.json`, `evidence.json` and the
upload API's response all share three keys, which is why the page needs exactly one branch for
"bundled sample or live upload" (`state.uploaded || DATA`). Keeping the translation here means the
domain aggregate never has to know the word "cells".
"""
from __future__ import annotations

from ..domain.document import ReadDocument


def document_evidence(document: ReadDocument) -> dict:
    """One reading as the page renders it: document, systems, pages[].cells[engine]."""
    engine_id = document.engine.id
    pages = []
    for page in document.pages:
        entry: dict = {
            "page": page.number,
            "cells": {
                engine_id: {"text": page.reading.text, **page.reading.stats.as_dict()},
            },
        }
        if page.preview is not None:
            entry["preview"] = page.preview
        pages.append(entry)

    return {
        "document": {
            "name": document.display_name,
            "pages": document.page_count,
            "note": document.note,
        },
        "systems": [document.engine.as_dict()],
        "pages": pages,
    }
