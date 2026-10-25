"""Shared bits: where pages and records live, and one shape for every runner."""
import json
from pathlib import Path

PAGES = Path("/tmp/ocr-playground/pages")
RECORDS = Path(__file__).resolve().parents[1] / "records"


def page_paths() -> list[Path]:
    pages = sorted(PAGES.glob("page-*.png"), key=lambda path: int(path.stem.split("-")[1]))
    if not pages:
        raise SystemExit(f"no pages in {PAGES} — run prepare.py first")
    return pages


def write(system_id: str, pages: dict[str, str]) -> None:
    RECORDS.mkdir(parents=True, exist_ok=True)
    target = RECORDS / f"{system_id}.json"
    target.write_text(json.dumps(pages, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"{system_id}: {len(pages)} pages, {sum(len(text) for text in pages.values()):,} chars -> {target}")
