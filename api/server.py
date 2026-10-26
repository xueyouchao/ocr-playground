"""The upload API: hand it a PDF or an image, get back per-page text and previews.

    python api/server.py --port 8090

Scope is deliberately small. The sample document on the page was parsed by eight systems ahead of time;
an upload is parsed live by the parsers that can run on this machine in seconds, which today is
LiteParse + Tesseract. Everything else about the shape (per-page cells, counts) matches the static
evidence so the same renderer draws both.

Nothing is stored: the bytes live in a temporary directory that is deleted before the response is sent.
The bounds below are the whole defence, because this endpoint is unauthenticated and public.
"""
from __future__ import annotations

import argparse
import base64
import json
import re
import shutil
import subprocess
import sys
import tempfile
import threading
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote

MAX_UPLOAD_BYTES = 20 * 1024 * 1024
MAX_PAGES = 5
PARSE_TIMEOUT_SECONDS = 120
RENDER_DPI = 200
PREVIEW_DPI = 110

PDF_TYPES = {"application/pdf", "application/x-pdf"}
IMAGE_TYPES = {"image/png", "image/jpeg", "image/tiff", "image/webp"}

# Two parses at a time: the box has four cores and each parse is CPU-bound.
_SLOTS = threading.BoundedSemaphore(2)

LIT = shutil.which("lit") or "/home/ubuntu/ingestion-engine/.venv/bin/lit"

SYSTEMS = [
    {
        "id": "liteparse-tesseract",
        "name": "LiteParse + Tesseract",
        "kind": "baseline",
        "pipeline": "raster + bundled Tesseract, parsed live on upload",
        "device": "this server (CPU)",
        "needs": "nothing",
    }
]


def run(command: list[str], timeout: int = PARSE_TIMEOUT_SECONDS) -> str:
    result = subprocess.run(command, capture_output=True, text=True, timeout=timeout)
    if result.returncode != 0:
        raise RuntimeError(f"{command[0]} failed: {result.stderr.strip()[:400]}")
    return result.stdout


def measure(text: str) -> dict:
    """The same counts collect.py records for the static document."""
    stripped = re.sub(r"<[^>]+>", " ", text)
    tokens = re.findall(r"\S+", stripped)
    numeric = [token for token in tokens if re.fullmatch(r"[\d.,:%/\-+()]+", token)]
    return {
        "chars": len(text),
        "tokens": len(tokens),
        "numeric_share": round(len(numeric) / len(tokens), 3) if tokens else 0.0,
    }


def parse_page(image: Path) -> str:
    return run([LIT, "parse", str(image), "--format", "markdown", "-q"]).strip()


def preview_data_url(image: Path) -> str:
    target = image.with_suffix(".preview.jpg")
    run(["convert", str(image), "-resize", "1400x1400", "-quality", "78", str(target)])
    return "data:image/jpeg;base64," + base64.b64encode(target.read_bytes()).decode()


def render_pages(source: Path, scratch: Path, media_type: str) -> list[Path]:
    if media_type in IMAGE_TYPES:
        # Parse the bytes that arrived. Round-tripping a PNG through ImageMagick rewrites it as an
        # 8-bit grayscale palette image, and OCR reads half the page: 111 chars instead of 407 on the
        # sample form. Only the formats LiteParse cannot open take the conversion path.
        if media_type in {"image/png", "image/jpeg"}:
            return [source]
        single = scratch / "page-1.png"
        run(["convert", str(source), "-colorspace", "sRGB", str(single)])
        return [single]

    info = run(["pdfinfo", str(source)])
    match = re.search(r"^Pages:\s+(\d+)", info, re.MULTILINE)
    if not match:
        raise ValueError("could not read the page count from that PDF")
    count = int(match.group(1))
    if count > MAX_PAGES:
        raise ValueError(f"that PDF has {count} pages; this demo reads at most {MAX_PAGES}")
    run(["pdftoppm", "-png", "-r", str(RENDER_DPI), str(source), str(scratch / "page")])
    return sorted(scratch.glob("page-*.png"), key=lambda path: int(path.stem.split("-")[1]))


def parse_document(payload: bytes, media_type: str, name: str) -> dict:
    if media_type not in PDF_TYPES | IMAGE_TYPES:
        raise ValueError(f"{media_type or 'unknown type'} is not a PDF or an image")
    if not payload:
        raise ValueError("that upload is empty")

    suffix = Path(name).suffix or (".pdf" if media_type in PDF_TYPES else ".png")
    with tempfile.TemporaryDirectory(prefix="ocr-playground-") as scratch_name:
        scratch = Path(scratch_name)
        source = scratch / f"upload{suffix}"
        source.write_bytes(payload)

        images = render_pages(source, scratch, media_type)
        pages = []
        for number, image in enumerate(images, 1):
            text = parse_page(image)
            pages.append({
                "page": number,
                "preview": preview_data_url(image),
                "cells": {"liteparse-tesseract": {"text": text, **measure(text)}},
            })

    return {
        "document": {"name": Path(name).name or "upload", "pages": len(pages), "note": "uploaded just now"},
        "systems": SYSTEMS,
        "pages": pages,
    }


class Handler(BaseHTTPRequestHandler):
    server_version = "ocr-playground"
    protocol_version = "HTTP/1.1"
    # A client that declares a length and then goes quiet must not own a thread indefinitely; the
    # socket timeout bounds the read, and the parse slot is taken *before* the read so stalled
    # uploads queue behind the same two-slot bound as real work.
    timeout = 30

    def log_message(self, fmt: str, *args) -> None:  # quieter logs
        sys.stderr.write("%s %s\n" % (self.address_string(), fmt % args))

    def _json(self, status: HTTPStatus, body: dict) -> None:
        encoded = json.dumps(body).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(encoded)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(encoded)

    def do_OPTIONS(self) -> None:  # noqa: N802 - the stdlib spelling
        self.send_response(HTTPStatus.NO_CONTENT)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, X-Filename")
        self.send_header("Access-Control-Allow-Methods", "POST, GET, OPTIONS")
        self.send_header("Content-Length", "0")
        self.end_headers()

    def do_GET(self) -> None:  # noqa: N802
        if self.path.split("?")[0] == "/api/health":
            self._json(HTTPStatus.OK, {
                "ok": True,
                "systems": SYSTEMS,
                "limits": {"bytes": MAX_UPLOAD_BYTES, "pages": MAX_PAGES},
            })
            return
        self._json(HTTPStatus.NOT_FOUND, {"error": "not found"})

    def do_POST(self) -> None:  # noqa: N802
        if self.path.split("?")[0] != "/api/parse":
            self._json(HTTPStatus.NOT_FOUND, {"error": "not found"})
            return

        try:
            length = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            length = 0
        if length <= 0:
            self._json(HTTPStatus.LENGTH_REQUIRED, {"error": "send the file as the request body"})
            return
        if length > MAX_UPLOAD_BYTES:
            self._json(HTTPStatus.REQUEST_ENTITY_TOO_LARGE, {
                "error": f"limit is {MAX_UPLOAD_BYTES // (1024 * 1024)} MB",
            })
            return

        # The filename arrives percent-encoded: a header value is Latin-1, so "扫描.pdf" cannot ride
        # in it verbatim.
        name = unquote(self.headers.get("X-Filename") or "upload")
        media_type = (self.headers.get("Content-Type") or "").split(";")[0].strip().lower()

        if not _SLOTS.acquire(timeout=30):
            self._json(HTTPStatus.SERVICE_UNAVAILABLE, {"error": "busy, try again in a moment"})
            return
        try:
            payload = self.rfile.read(length)
        except TimeoutError:
            _SLOTS.release()
            self._json(HTTPStatus.REQUEST_TIMEOUT, {"error": "the upload stalled; send it again"})
            return
        if len(payload) != length:
            _SLOTS.release()
            self._json(HTTPStatus.REQUEST_TIMEOUT, {"error": "the upload ended early; send it again"})
            return
        try:
            document = parse_document(payload, media_type, name)
        except ValueError as error:
            self._json(HTTPStatus.BAD_REQUEST, {"error": str(error)})
            return
        except subprocess.TimeoutExpired:
            self._json(HTTPStatus.GATEWAY_TIMEOUT, {"error": "parsing took too long"})
            return
        except Exception as error:  # noqa: BLE001 - a bad upload must not kill the server
            self._json(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": f"could not read that file: {error}"})
            return
        finally:
            _SLOTS.release()

        self._json(HTTPStatus.OK, document)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8090)
    args = parser.parse_args()

    httpd = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"ocr-playground api on http://{args.host}:{args.port} (lit: {LIT})", flush=True)
    httpd.serve_forever()


if __name__ == "__main__":
    main()
