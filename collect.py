"""Merge every recorded run plus the roster into one evidence file for the page."""
import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent


def measure(text: str | None) -> dict:
    """Describe the text. Counts only — there is no score here on purpose.

    The dictionary share that used to live here rewarded prose and punished numeric tables: on this
    document its denominator ranged from 2 to 187 words, and on page 4 (77% numbers, and the rest
    column abbreviations like "CLD" and "TDME") it scored a correct reading at 30%. A number that
    wrong is worse than no number, so it is gone rather than tuned. Accuracy needs ground truth.
    """
    if text is None:
        return {"chars": None, "tokens": None, "numeric_share": None}

    stripped = re.sub(r"<[^>]+>", " ", text)
    tokens = re.findall(r"\S+", stripped)
    numeric = [token for token in tokens if re.fullmatch(r"[\d.,:%/\-+()]+", token)]
    return {
        "chars": len(text),
        "tokens": len(tokens),
        "numeric_share": round(len(numeric) / len(tokens), 3) if tokens else 0.0,
    }


def main() -> None:
    systems = json.loads((HERE / "systems.json").read_text(encoding="utf-8"))
    recorded = {}
    for path in sorted((HERE / "records").glob("*.json")):
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
        "systems": [{k: s[k] for k in ("id", "name", "kind", "pipeline", "device", "needs")} for s in systems],
        "pages": pages,
    }
    (HERE / "evidence.json").write_text(json.dumps(evidence, indent=1, ensure_ascii=False), encoding="utf-8")
    for system in systems:
        values = [p["cells"][system["id"]]["chars"] for p in pages]
        ran = [v for v in values if v is not None]
        print(f'{system["id"]:<22} {"ran on " + str(len(ran)) + "/" + str(page_count) + " pages" if ran else "not run"}')


if __name__ == "__main__":
    main()
