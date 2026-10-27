"""Shared bits: where pages and records live, and one shape for every runner."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
# The importable tree lives under src/; the repository root holds data, site and scripts.
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ocr_playground.interfaces.site.layout import data  # noqa: E402

PAGES = Path("/tmp/ocr-playground/pages")
RECORDS = data(ROOT) / "records"


def page_paths() -> list[Path]:
    pages = sorted(PAGES.glob("page-*.png"), key=lambda path: int(path.stem.split("-")[1]))
    if not pages:
        raise SystemExit(f"no pages in {PAGES} — run 'make pages' first")
    return pages


def write(system_id: str, pages: dict[str, str]) -> None:
    RECORDS.mkdir(parents=True, exist_ok=True)
    target = RECORDS / f"{system_id}.json"
    target.write_text(json.dumps(pages, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"{system_id}: {len(pages)} pages, {sum(len(text) for text in pages.values()):,} chars -> {target}")
