"""The upload API: hand it a PDF or an image, get back per-page text and previews.

    GLM_OCR_BASE_URL=https://…/v1 python api/server.py --port 8090

The reading itself lives in `pipeline.py`, because the CocoIndex indexer (`indexer/flow.py`) reads a
corpus with the same two paths:

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
import json
import subprocess
import sys
import threading
import time
import urllib.parse
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote

# `python api/server.py` puts api/ on sys.path, not the repository root where the shared pipeline is.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline import (  # noqa: E402 - the path above has to be set first
    DEFAULT_MODE,
    GLM_BASE_URL,
    LIT,
    MAX_PAGES,
    SYSTEMS,
    UpstreamError,
    parse_document,
)

MAX_UPLOAD_BYTES = 20 * 1024 * 1024

# Two parses at a time: the box has four cores and the fast path is CPU-bound. A GLM upload holds its
# slot while it waits on the GPU, so two cold starts do not stampede the endpoint.
_SLOTS = threading.BoundedSemaphore(2)


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
            document = parse_document(
                payload,
                media_type,
                name,
                mode,
                note=f"uploaded just now · read by {SYSTEMS[mode]['name']}",
            )
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
