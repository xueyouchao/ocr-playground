"""The upload API: hand it a PDF or an image, get back per-page text and previews.

    GLM_OCR_BASE_URL=https://…/v1 python -m ocr_playground.interfaces.http --port 8090

The handler is a shell: it validates the transport (length, size cap, filename encoding, mode),
takes a parse slot, and asks the use case for a reading. The reading itself, and every failure it can
raise, lives outside this file. Nothing is stored: the bytes live in a temporary directory that is
deleted before the response is sent.

The bounds below are the whole defence, because this endpoint is unauthenticated and public.
"""
from __future__ import annotations

import argparse
import json
import select
import socket
import sys
import time
import urllib.parse
from collections.abc import Sequence
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import unquote

from ...application.capacity import SlotsBusy
from ...application.read_document import reading_note
from ...application.errors import (
    ClientGone,
    EngineFailure,
    EngineTimeout,
    EngineUnavailable,
)
from ...composition.container import Container, build_container
from ...domain.document import SourceDocument
from ...domain.errors import DomainError, TooManyModes
from ...domain.media import MediaType
from ...infrastructure.process import CommandTimedOut
from ..evidence import document_evidence
from .health import build_health


def failure_response(error: BaseException) -> tuple[HTTPStatus, str]:
    """The status and body for anything the reading raised.

    An unexpected exception is still an unreadable file from the client's point of view, so it keeps
    the wording the page has always shown rather than a bare "internal error".
    """
    if isinstance(error, ClientGone):
        # Nothing was left to answer to; a proxy that does see this should not retry the upload.
        return HTTPStatus.REQUEST_TIMEOUT, "the client closed the request"
    if isinstance(error, CommandTimedOut):
        return HTTPStatus.GATEWAY_TIMEOUT, "parsing took too long"
    if isinstance(error, EngineUnavailable):
        return HTTPStatus.SERVICE_UNAVAILABLE, str(error)
    if isinstance(error, EngineTimeout):
        return HTTPStatus.GATEWAY_TIMEOUT, str(error)
    if isinstance(error, EngineFailure):
        return HTTPStatus.BAD_GATEWAY, str(error)
    if isinstance(error, (DomainError, ValueError)):
        return HTTPStatus.BAD_REQUEST, str(error)
    return HTTPStatus.INTERNAL_SERVER_ERROR, f"could not read that file: {error}"


class UploadHTTPServer(ThreadingHTTPServer):
    """The server owns the container, so a handler reaches it through `self.server`."""

    daemon_threads = True

    def __init__(self, address: tuple[str, int], handler: type[BaseHTTPRequestHandler],
                 container: Container) -> None:
        self.container = container
        super().__init__(address, handler)


MAX_COMPARISON = 3
"""Parsers a single comparison may ask for; the picker offers the same number."""


class Handler(BaseHTTPRequestHandler):
    server_version = "ocr-playground"
    protocol_version = "HTTP/1.1"
    # A client that declares a length and then goes quiet must not own a thread indefinitely; the
    # socket timeout bounds the read, and the parse slot is taken *before* the read so stalled
    # uploads queue behind the same two-slot bound as real work.
    timeout = 30

    def client_is_gone(self) -> bool:
        """Has the browser closed the connection while we were working?

        A parse can hold a request for minutes waiting on a cold GPU, and the page gives up long
        before that. The socket says so first: readable with nothing to read is EOF. Cheap enough to
        ask between engines and between pages.
        """
        try:
            readable, _, _ = select.select([self.connection], [], [], 0)
        except (OSError, ValueError):
            return True
        if not readable:
            return False
        try:
            return self.connection.recv(1, socket.MSG_PEEK) == b""
        except (BlockingIOError, InterruptedError):
            return False
        except OSError:
            return True

    @property
    def container(self) -> Container:
        return self.server.container  # type: ignore[attr-defined]

    def log_message(self, fmt: str, *args) -> None:  # quieter logs
        sys.stderr.write(f"{self.address_string()} {fmt % args}\n")

    def _json(self, status: HTTPStatus, body: dict) -> None:
        # A POST that was answered before its body was read (a rejected mode, a busy slot) must not
        # leave those bytes in front of the next request on a kept-alive connection. `http.server`
        # honours close_connection but does not announce it, so the header says so explicitly —
        # without it the client reuses a socket that is already closed.
        closing = getattr(self, "_body_read", True) is False
        if closing:
            self.close_connection = True
        encoded = json.dumps(body).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(encoded)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Cache-Control", "no-store")
        if closing:
            self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(encoded)

    def _failure(self, error: BaseException) -> None:
        status, message = failure_response(error)
        self._json(status, {"error": message})

    def do_OPTIONS(self) -> None:  # noqa: N802 - the stdlib spelling
        self.send_response(HTTPStatus.NO_CONTENT)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, X-Filename, X-Mode")
        self.send_header("Access-Control-Allow-Methods", "POST, GET, OPTIONS")
        self.send_header("Content-Length", "0")
        self.end_headers()

    def requested_mode(self) -> str:
        """`?mode=fast` or `X-Mode: fast`; GLM-OCR when neither says otherwise.

        A comma-separated list (`glm,chandra`) asks for a comparison: the engines read the same
        pages concurrently and the response carries one system per engine.

        The modes are resolved through the engine registry, so an unknown mode and an unconfigured
        engine are rejected with their own errors rather than a shared 400.
        """
        query = urllib.parse.parse_qs(urllib.parse.urlsplit(self.path).query)
        raw = (
            query.get("mode")
            or [self.headers.get("X-Mode") or self.container.settings().default_mode]
        )[0]
        modes = [part.strip().lower() for part in raw.split(",") if part.strip()]
        for mode in modes:
            self.container.read_document().require_mode(mode)
        wanted = list(dict.fromkeys(modes))
        if len(wanted) > MAX_COMPARISON:
            # Three GPUs at once is what the plan and the pool can carry; a fourth row only queues.
            raise TooManyModes(MAX_COMPARISON)
        return ",".join(wanted)

    def do_GET(self) -> None:  # noqa: N802
        if self.path.split("?")[0] == "/api/health":
            settings = self.container.settings()
            warmer = self.container.warmer()
            self._json(HTTPStatus.OK, build_health(
                self.container.registry(),
                self.container.engines(),
                max_upload_bytes=settings.max_upload_bytes,
                max_pages=settings.max_pages,
                warm={
                    "enabled": settings.warm_enabled,
                    "engines": warmer.report(),
                },
            ))
            return
        self._json(HTTPStatus.NOT_FOUND, {"error": "not found"})

    def _discard_body(self, limit: int = 8192) -> None:
        """Take whatever a client sent with the warm request, so nothing is left for the next one.

        The endpoint takes no body, but `protocol_version` is HTTP/1.1: unread bytes sit in front of
        the next request line on a kept-alive connection and break it. A body too large to drain
        cheaply is answered by closing the connection instead.
        """
        if (self.headers.get("Transfer-Encoding") or "").lower() == "chunked":
            self.close_connection = True
            return
        try:
            length = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            length = 0
        if length <= 0:
            self._body_read = True  # nothing was declared, so nothing is left over
            return
        if length > limit:
            self.close_connection = True
            return
        self.rfile.read(length)
        self._body_read = True

    def _warm(self) -> None:
        """Start a cold engine booting on intent, and say whether it started.

        Deliberately not on `GET /api/health`: a container bills GPU time from boot until it scales
        down, so warming every page view would rent a GPU for a trickle of traffic. The page calls
        this once per selected parser when somebody picks one or reaches for the uploader, and the
        warmer throttles each engine on its own window.
        """
        self._discard_body()
        try:
            mode = self.requested_mode()
        except Exception as error:  # noqa: BLE001 - an unknown mode or an unset endpoint
            self._failure(error)
            return
        status = self.container.warmer().request(mode)
        # A warm rents GPU time from boot until it scales down, so which engine was warmed — and why —
        # has to be readable in the service log rather than inferred from the dashboard.
        sys.stderr.write(f"warm {mode}: {status.value}\n")
        self._json(HTTPStatus.OK, {"warm": status.value})

    def do_POST(self) -> None:  # noqa: N802
        self._body_read = False
        if self.path.split("?")[0] == "/api/warm":
            self._warm()
            return
        if self.path.split("?")[0] != "/api/parse":
            self._json(HTTPStatus.NOT_FOUND, {"error": "not found"})
            return

        try:
            length = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            length = 0
        settings = self.container.settings()
        if length <= 0:
            # No Content-Length is not necessarily no body: a chunked upload lands here with its
            # chunks still in the socket, so this rejection closes the connection rather than
            # leaving them to be parsed as the next request.
            self._json(HTTPStatus.LENGTH_REQUIRED, {"error": "send the file as the request body"})
            return
        if length > settings.max_upload_bytes:
            self._json(HTTPStatus.REQUEST_ENTITY_TOO_LARGE, {
                "error": f"limit is {settings.max_upload_bytes // (1024 * 1024)} MB",
            })
            return

        # The filename arrives percent-encoded: a header value is Latin-1, so "扫描.pdf" cannot ride
        # in it verbatim.
        name = unquote(self.headers.get("X-Filename") or "upload")
        media_type = MediaType.parse(self.headers.get("Content-Type"))

        # Resolve the extraction path before taking a slot: an unknown mode, or GLM-OCR on a server
        # with no endpoint configured, should not queue behind real work.
        try:
            mode = self.requested_mode()
        except Exception as error:  # noqa: BLE001 - mapped to the status it deserves
            self._failure(error)
            return

        started = time.monotonic()
        try:
            with self.container.parse_slots().hold():
                try:
                    payload = self.rfile.read(length)
                    self._body_read = True
                except TimeoutError:
                    self._json(HTTPStatus.REQUEST_TIMEOUT, {"error": "the upload stalled; send it again"})
                    return
                except OSError:
                    self._json(HTTPStatus.BAD_REQUEST, {"error": "the upload was cut off; send it again"})
                    return
                if len(payload) != length:
                    self._json(HTTPStatus.REQUEST_TIMEOUT, {"error": "the upload ended early; send it again"})
                    return

                service = self.container.read_document()
                engines = [service.require_mode(part) for part in mode.split(",")]
                document = service.execute(
                    SourceDocument.accepted(payload, media_type, name), mode, cancel=self.client_is_gone,
                    note=f"uploaded just now · {reading_note(engines)}",
                )
                evidence = document_evidence(document)
        except ClientGone:
            # The tab was closed: answering would only write into a socket that is already gone, and
            # the engines have stopped, so the slot is released here rather than minutes later.
            sys.stderr.write("the client hung up; the parse was abandoned\n")
            self.close_connection = True
            return
        except SlotsBusy:
            self._json(HTTPStatus.SERVICE_UNAVAILABLE, {"error": "busy, try again in a moment"})
            return
        except Exception as error:  # noqa: BLE001 - a bad upload must not kill the server
            self._failure(error)
            return

        # Cold starts are minutes long; one line per upload makes that visible in the service log.
        elapsed = time.monotonic() - started
        sys.stderr.write(
            f"parsed {document.page_count} page(s) with "
            f"{', '.join(engine.id for engine in document.engines)} in {elapsed:.1f}s\n"
        )
        self._json(HTTPStatus.OK, evidence)


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="ocr-playground-api")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8090)
    args = parser.parse_args(argv)

    container = build_container()
    settings = container.settings()
    httpd = UploadHTTPServer((args.host, args.port), Handler, container)
    available = container.engines()[settings.default_mode].available
    state = "configured" if available else "not configured; the page falls back"
    print(f"ocr-playground api on http://{args.host}:{args.port}", flush=True)
    print(f"  default: {settings.default_mode} — {state}", flush=True)
    print(f"  lit: {settings.lit_path}", flush=True)
    httpd.serve_forever()
