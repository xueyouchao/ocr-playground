"""The upload API: hand it a PDF or an image, get back per-page text and previews.

    GLM_OCR_BASE_URL=https://…/v1 python api/server.py --port 8090

An upload is parsed live by one of two paths, chosen per request:

    glm   (default) GLM-OCR, a 0.9B vision model on a scale-to-zero Modal L4. It reads handwriting
          and tables far better than a classic OCR pass, but the GPU is cold most of the time and
          the first request after idle can take several minutes while vLLM boots.
    fast  LiteParse + Tesseract, on this server's CPU: a couple of seconds per page, visibly worse
          on handwriting, and the only path available when no GLM endpoint is configured.

Everything about the shape (per-page cells, counts) matches the static evidence so the same renderer
draws both. Nothing is stored: the bytes live in a temporary directory that is deleted before the
response is sent. The bounds below are the whole defence, because this endpoint is unauthenticated
and public.
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote

MAX_UPLOAD_BYTES = 20 * 1024 * 1024
MAX_PAGES = 5
PARSE_TIMEOUT_SECONDS = 120
RENDER_DPI = 200

# GLM-OCR is reached through any OpenAI-compatible endpoint; `modal/serve_glm_ocr.py` serves exactly
# that behind `modal deploy`. A cold start is a 303 to the same URL that then parks until the
# container is up, which is why these timeouts are minutes rather than seconds.
GLM_BASE_URL = os.environ.get("GLM_OCR_BASE_URL", "").rstrip("/")
GLM_MODEL = os.environ.get("GLM_OCR_MODEL", "glm-ocr")
GLM_API_KEY = os.environ.get("GLM_OCR_API_KEY", "") or "not-needed"
# "Text Recognition:" is what GLM-OCR's own recipe sends; a generic markdown instruction measurably
# changes what the smaller VLMs emit.
GLM_PROMPT = os.environ.get("GLM_OCR_PROMPT", "Text Recognition:")
GLM_REQUEST_TIMEOUT = int(os.environ.get("GLM_OCR_REQUEST_TIMEOUT", "900"))
# The whole document, cold start included. Five pages at ~20 s each is 100 s; the rest is boot.
GLM_BUDGET_SECONDS = int(os.environ.get("GLM_OCR_BUDGET_SECONDS", "900"))

PDF_TYPES = {"application/pdf", "application/x-pdf"}
IMAGE_TYPES = {"image/png", "image/jpeg", "image/tiff", "image/webp"}

# Two parses at a time: the box has four cores and the fast path is CPU-bound. A GLM upload holds its
# slot while it waits on the GPU, so two cold starts do not stampede the endpoint.
_SLOTS = threading.BoundedSemaphore(2)

LIT = shutil.which("lit") or "/home/ubuntu/ingestion-engine/.venv/bin/lit"

DEFAULT_MODE = "glm"

# The ids and labels match `systems.json`, so an uploaded pane looks like the recorded one.
SYSTEMS = {
    "glm": {
        "id": "glm-ocr",
        "name": "GLM-OCR",
        "kind": "vlm",
        "pipeline": "0.9B VLM on a Modal L4 (vLLM) — cold starts can take minutes",
        "device": "GPU (Modal L4)",
        "needs": "GLM_OCR_BASE_URL",
    },
    "fast": {
        "id": "liteparse-tesseract",
        "name": "LiteParse + Tesseract",
        "kind": "baseline",
        "pipeline": "raster + bundled Tesseract, parsed live on upload",
        "device": "this server (CPU)",
        "needs": "nothing",
    },
}


class UpstreamError(RuntimeError):
    """A failure worth reporting to the uploader, with the status it should arrive as."""

    def __init__(self, message: str, status: HTTPStatus) -> None:
        super().__init__(message)
        self.status = status


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


def glm_page(image: Path, deadline: float) -> str:
    """One page through an OpenAI-compatible GLM-OCR endpoint, cold start included."""
    raw = image.read_bytes()
    # An upload keeps its own bytes, so the suffix is not proof of the format; the first two bytes
    # are. PDF pages and converted TIFF/WebP are PNG, an uploaded JPEG stays JPEG.
    mime = "image/jpeg" if raw.startswith(b"\xff\xd8") else "image/png"
    payload = {
        "model": GLM_MODEL,
        "temperature": 0,
        "max_tokens": 4096,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {
                    "url": f"data:{mime};base64," + base64.b64encode(raw).decode(),
                }},
                {"type": "text", "text": GLM_PROMPT},
            ],
        }],
    }
    data = json.dumps(payload).encode()
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {GLM_API_KEY}"}
    url = GLM_BASE_URL + "/chat/completions"

    # Modal answers a cold endpoint with a 303 to the same URL; following it *is* the cold-start wait.
    # urllib will not replay a POST, so the redirect is followed by hand rather than by urllib.
    body = None
    for _redirect in range(4):
        remaining = deadline - time.monotonic()
        if remaining <= 1:
            raise UpstreamError(
                "GLM-OCR did not finish within the time limit — try again once the GPU is warm, "
                "or use the fast option (LiteParse + Tesseract)",
                HTTPStatus.GATEWAY_TIMEOUT,
            )
        request = urllib.request.Request(url, data=data, headers=headers)
        try:
            with urllib.request.urlopen(request, timeout=min(GLM_REQUEST_TIMEOUT, remaining)) as response:
                body = json.loads(response.read().decode())
            break
        except urllib.error.HTTPError as error:
            if error.code in (303, 307, 308) and error.headers.get("Location"):
                url = error.headers["Location"]
                continue
            detail = error.read().decode(errors="replace")[:300]
            raise UpstreamError(
                f"the GLM-OCR endpoint answered {error.code}: {detail or 'no detail'}",
                HTTPStatus.BAD_GATEWAY,
            ) from error
        except (TimeoutError, urllib.error.URLError) as error:
            raise UpstreamError(
                f"the GLM-OCR endpoint did not answer: {error}",
                HTTPStatus.GATEWAY_TIMEOUT,
            ) from error

    if body is None:
        raise UpstreamError("the GLM-OCR endpoint kept redirecting", HTTPStatus.BAD_GATEWAY)
    try:
        content = body["choices"][0]["message"].get("content") or ""
    except (KeyError, IndexError, TypeError) as error:
        raise UpstreamError("the GLM-OCR endpoint returned an unexpected body", HTTPStatus.BAD_GATEWAY) from error
    return content.strip()


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


def parse_document(payload: bytes, media_type: str, name: str, mode: str = DEFAULT_MODE) -> dict:
    if media_type not in PDF_TYPES | IMAGE_TYPES:
        raise ValueError(f"{media_type or 'unknown type'} is not a PDF or an image")
    if not payload:
        raise ValueError("that upload is empty")
    system = SYSTEMS[mode]

    suffix = Path(name).suffix or (".pdf" if media_type in PDF_TYPES else ".png")
    with tempfile.TemporaryDirectory(prefix="ocr-playground-") as scratch_name:
        scratch = Path(scratch_name)
        source = scratch / f"upload{suffix}"
        source.write_bytes(payload)

        images = render_pages(source, scratch, media_type)
        # One budget for the whole document: the cold start is paid on the first page and the rest
        # inherit what is left of it.
        deadline = time.monotonic() + GLM_BUDGET_SECONDS
        pages = []
        for number, image in enumerate(images, 1):
            text = glm_page(image, deadline) if mode == "glm" else parse_page(image)
            pages.append({
                "page": number,
                "preview": preview_data_url(image),
                "cells": {system["id"]: {"text": text, **measure(text)}},
            })

    return {
        "document": {
            "name": Path(name).name or "upload",
            "pages": len(pages),
            "note": f"uploaded just now · read by {system['name']}",
        },
        "systems": [system],
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
        self.send_header("Access-Control-Allow-Headers", "Content-Type, X-Filename, X-Mode")
        self.send_header("Access-Control-Allow-Methods", "POST, GET, OPTIONS")
        self.send_header("Content-Length", "0")
        self.end_headers()

    def requested_mode(self) -> str:
        """`?mode=fast` or `X-Mode: fast`; GLM-OCR when neither says otherwise."""
        query = urllib.parse.parse_qs(urllib.parse.urlsplit(self.path).query)
        raw = (query.get("mode") or [self.headers.get("X-Mode") or DEFAULT_MODE])[0]
        mode = raw.strip().lower()
        if mode not in SYSTEMS:
            raise ValueError(f"{raw!r} is not an extraction mode; use glm or fast")
        if mode == "glm" and not GLM_BASE_URL:
            raise UpstreamError(
                "GLM-OCR is not configured on this server; use the fast option (LiteParse + Tesseract)",
                HTTPStatus.SERVICE_UNAVAILABLE,
            )
        return mode

    def do_GET(self) -> None:  # noqa: N802
        if self.path.split("?")[0] == "/api/health":
            self._json(HTTPStatus.OK, {
                "ok": True,
                "default_mode": DEFAULT_MODE,
                "modes": [
                    {
                        "id": "glm",
                        "name": "GLM-OCR",
                        "available": bool(GLM_BASE_URL),
                        "cold_start": True,
                        "detail": "0.9B VLM on a scale-to-zero GPU; the first upload after idle can take minutes",
                    },
                    {
                        "id": "fast",
                        "name": "LiteParse + Tesseract",
                        "available": True,
                        "cold_start": False,
                        "detail": "CPU OCR on this server; a couple of seconds per page",
                    },
                ],
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

        # Resolve the extraction path before taking a slot: an unknown mode, or GLM-OCR on a server
        # with no endpoint configured, should not queue behind real work.
        try:
            mode = self.requested_mode()
        except UpstreamError as error:
            self._json(error.status, {"error": str(error)})
            return
        except ValueError as error:
            self._json(HTTPStatus.BAD_REQUEST, {"error": str(error)})
            return

        if not _SLOTS.acquire(timeout=30):
            self._json(HTTPStatus.SERVICE_UNAVAILABLE, {"error": "busy, try again in a moment"})
            return
        # One try/finally around the read *and* the parse. Releasing per branch leaked the slot
        # whenever the read raised something other than TimeoutError -- ConnectionResetError most
        # often -- and two such failures left every later upload answering "busy" until a restart.
        started = time.monotonic()
        try:
            try:
                payload = self.rfile.read(length)
            except TimeoutError:
                self._json(HTTPStatus.REQUEST_TIMEOUT, {"error": "the upload stalled; send it again"})
                return
            except OSError:
                self._json(HTTPStatus.BAD_REQUEST, {"error": "the upload was cut off; send it again"})
                return
            if len(payload) != length:
                self._json(HTTPStatus.REQUEST_TIMEOUT, {"error": "the upload ended early; send it again"})
                return
            document = parse_document(payload, media_type, name, mode)
        except UpstreamError as error:
            self._json(error.status, {"error": str(error)})
            return
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

        # Cold starts are minutes long; one line per upload makes that visible in the service log.
        elapsed = time.monotonic() - started
        sys.stderr.write(
            f"parsed {len(document['pages'])} page(s) with {document['systems'][0]['id']} in {elapsed:.1f}s\n"
        )
        self._json(HTTPStatus.OK, document)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8090)
    args = parser.parse_args()

    httpd = ThreadingHTTPServer((args.host, args.port), Handler)
    glm_state = GLM_BASE_URL or "not configured (fast only)"
    print(f"ocr-playground api on http://{args.host}:{args.port}", flush=True)
    print(f"  default: glm — {glm_state}", flush=True)
    print(f"  lit: {LIT}", flush=True)
    httpd.serve_forever()


if __name__ == "__main__":
    main()
