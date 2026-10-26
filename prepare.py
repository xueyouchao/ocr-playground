"""Render the PDF's pages to PNG with poppler, so the shootout needs no Python PDF library.

    python prepare.py [--dpi 200] [--pdf path]

`pdftoppm` writes page-1.png, page-2.png, … in the order the parsers are given them.
"""
import argparse
import shutil
import subprocess
from pathlib import Path

PDF = Path(__file__).resolve().parent / "corpus" / "document.pdf"
OUT = Path("/tmp/ocr-playground/pages")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dpi", type=int, default=200)
    parser.add_argument("--pdf", default=str(PDF))
    args = parser.parse_args()

    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True)
    subprocess.run(["pdftoppm", "-png", "-r", str(args.dpi), args.pdf, str(OUT / "page")], check=True)
    pages = sorted(OUT.glob("page-*.png"), key=lambda path: int(path.stem.split("-")[1]))
    print(f"{len(pages)} pages at {args.dpi} dpi in {OUT}")

    # A lighter copy for the page's own preview rail; poppler writes these directly, no PIL needed.
    previews = Path(__file__).resolve().parent / "assets" / "pages"
    if previews.exists():
        shutil.rmtree(previews)
    previews.mkdir(parents=True)
    subprocess.run(["pdftoppm", "-jpeg", "-r", "110", "-jpegopt", "quality=78",
                    args.pdf, str(previews / "page")], check=True)
    written = sorted(previews.glob("page-*.jpg"), key=lambda path: int(path.stem.split("-")[1]))
    print(f"{len(written)} previews in {previews} ({sum(p.stat().st_size for p in written) // 1024} KB)")


if __name__ == "__main__":
    main()
