"""Render the sample PDF's pages with poppler, so the shootout needs no Python PDF library.

    python prepare.py [--dpi 200] [--pdf path]

Two renders: 200 dpi pages for the parsers, 110 dpi JPEG previews for the page's own rail. poppler
writes both directly, so there is no PIL dependency.
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
from collections.abc import Sequence
from pathlib import Path

OUT = Path("/tmp/ocr-playground/pages")


def _page_order(path: Path) -> int:
    return int(path.stem.split("-")[1])


def main(root: Path, argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dpi", type=int, default=200)
    parser.add_argument("--pdf", default=str(root / "corpus" / "document.pdf"))
    args = parser.parse_args(argv)

    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True)
    subprocess.run(
        ["pdftoppm", "-png", "-r", str(args.dpi), args.pdf, str(OUT / "page")], check=True
    )
    pages = sorted(OUT.glob("page-*.png"), key=_page_order)
    print(f"{len(pages)} pages at {args.dpi} dpi in {OUT}")

    previews = root / "assets" / "pages"
    if previews.exists():
        shutil.rmtree(previews)
    previews.mkdir(parents=True)
    subprocess.run(
        ["pdftoppm", "-jpeg", "-r", "110", "-jpegopt", "quality=78", args.pdf,
         str(previews / "page")],
        check=True,
    )
    written = sorted(previews.glob("page-*.jpg"), key=_page_order)
    print(
        f"{len(written)} previews in {previews} "
        f"({sum(path.stat().st_size for path in written) // 1024} KB)"
    )
