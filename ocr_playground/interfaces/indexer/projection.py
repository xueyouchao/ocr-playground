"""The indexable projections: one row per page, and a stable JSON encoding.

These work on the evidence mapping rather than on the aggregate because that mapping is also what the
memoised function returns — the CocoIndex memo boundary is JSON, so the projection reads the same
bytes the next run will compare.
"""
from __future__ import annotations

import json
from collections.abc import Iterable


def page_rows(documents: Iterable[dict]) -> list[dict]:
    """Flatten a run's documents into the corpus index, sorted by path then page."""
    rows = []
    for document in documents:
        system = document["systems"][0]
        for page in document["pages"]:
            cell = page["cells"][system["id"]]
            rows.append({
                "document": document["document"]["path"],
                "digest": document["document"]["digest"],
                "page": page["page"],
                "engine": system["id"],
                "chars": cell["chars"],
                "tokens": cell["tokens"],
                "numeric_share": cell["numeric_share"],
                "text": cell["text"],
            })
    rows.sort(key=lambda row: (row["document"], row["page"]))
    return rows


def payload(value: object) -> str:
    """Sorted keys and a trailing newline: an unchanged run rewrites identical bytes."""
    return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
