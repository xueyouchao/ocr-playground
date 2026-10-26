"""LiteParse: the text layer laid out (no OCR), and optionally an OCR server plugged in.

    LIT_CLI=/path/to/lit python runners/liteparse_system.py  # text layer, lines joined
    python runners/liteparse.py --ocr-server-url http://…/ocr  # their "bring your own OCR"

The second form needs a server speaking LiteParse's OCR API. PaddleOCR's own wrapper lives at
https://github.com/run-llama/liteparse/tree/main/ocr/paddleocr — run that, then point here.
"""
import argparse
import os
import subprocess

from _common import page_paths, write

LIT = os.environ.get("LIT_CLI", "lit")  # LiteParse's CLI; set LIT_CLI if it is not on PATH


def parse(target: str, extra: list[str]) -> str:
    result = subprocess.run([LIT, "parse", target, "--format", "markdown", "-q", *extra],
                            capture_output=True, text=True)
    if result.returncode != 0:
        raise SystemExit(f"lit parse failed on {target}: {result.stderr[:400]}")
    return result.stdout.strip()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ocr-server-url", default=None)
    args = parser.parse_args()

    page_text = {page.stem: parse(str(page), []) for page in page_paths()}
    write("liteparse-tesseract", page_text)

    if args.ocr_server_url:
        with_ocr = {page.stem: parse(str(page), ["--ocr-server-url", args.ocr_server_url]) for page in page_paths()}
        write("liteparse-paddleocr", with_ocr)


if __name__ == "__main__":
    main()
