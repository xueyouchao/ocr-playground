"""The upload API's contract: status codes, bodies, and the defences around a public endpoint.

A real socket and a real server; only the use case is replaced, so these tests pin what a client sees
rather than which function the handler happened to call.
"""
from __future__ import annotations

import http.client
import json
import socket
import tempfile
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

import pytest

from fakes import (
    CHANDRA,
    FAST,
    GLM,
    FakeClock,
    FakeEngine,
    FakeSlots,
    FakeWarmEngine,
    StubService,
    build_rig,
    registry,
    sample_document,
)
from ocr_playground.application.errors import (
    ClientGone,
    EngineFailure,
    EngineTimeout,
    EngineUnavailable,
)
from ocr_playground.application.warming import EngineWarmer, WarmPolicy
from ocr_playground.composition.container import Container
from ocr_playground.domain.engine import EngineRegistry, EngineSpec
from ocr_playground.domain.errors import UnknownMode
from ocr_playground.infrastructure.process import CommandTimedOut
from ocr_playground.interfaces.evidence import document_evidence
from ocr_playground.interfaces.http import Handler, UploadHTTPServer, build_health, failure_response


@pytest.fixture
def api():
    servers: list[UploadHTTPServer] = []

    def make(*, service=None, rig=None, slots=None, **config) -> tuple[str, Container]:
        container = Container()
        if config:
            container.config.from_dict(config)
        if rig is not None:
            container.read_document.override(rig.service)
        if service is not None:
            container.read_document.override(service)
        container.parse_slots.override(slots or FakeSlots())
        httpd = UploadHTTPServer(("127.0.0.1", 0), Handler, container)
        threading.Thread(target=httpd.serve_forever, daemon=True).start()
        servers.append(httpd)
        return f"http://127.0.0.1:{httpd.server_address[1]}", container

    yield make
    for httpd in servers:
        httpd.shutdown()
        httpd.server_close()


def call(base: str, path: str, *, method: str = "GET", body: bytes | None = None,
         headers: dict[str, str] | None = None) -> tuple[int, dict, bytes]:
    request = urllib.request.Request(base + path, data=body, method=method, headers=headers or {})
    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            return response.status, dict(response.headers), response.read()
    except urllib.error.HTTPError as error:
        return error.code, dict(error.headers), error.read()


def payload(response: bytes) -> dict:
    return json.loads(response)


def test_health_reports_modes_and_limits(api) -> None:
    base, _ = api(glm_base_url="http://stub/v1")
    status, headers, body = call(base, "/api/health")

    assert status == 200
    assert headers["Content-Type"] == "application/json"
    assert headers["Access-Control-Allow-Origin"] == "*"
    assert headers["Cache-Control"] == "no-store"
    health = payload(body)
    assert health["ok"] is True
    assert health["default_mode"] == "chandra"
    modes = {mode["id"]: mode for mode in health["modes"]}
    assert tuple(modes) == (
        "glm", "fast", "chandra", "nemotron", "unlimited", "mineru", "paddle",
    )
    assert modes["glm"] == {
        "id": "glm",
        "name": "GLM-OCR",
        "available": True,
        "cold_start": True,
        "detail": "0.9B VLM on a scale-to-zero GPU; the first upload after idle takes about 3 minutes",
    }
    assert modes["fast"]["cold_start"] is False
    assert modes["fast"]["available"] is True
    # Endpoints nobody configured: listed, so the page can grey them out rather than fail an upload.
    assert modes["chandra"]["available"] is False
    assert {mode["cold_start"] for mode in modes.values()} == {True, False}
    assert health["limits"] == {"bytes": 20 * 1024 * 1024, "pages": 5}


def test_health_admits_when_glm_has_no_endpoint(api) -> None:
    base, _ = api()
    _, _, body = call(base, "/api/health")
    modes = {mode["id"]: mode for mode in payload(body)["modes"]}
    assert modes["glm"]["available"] is False
    assert modes["fast"]["available"] is True


def test_options_preflights_the_upload(api) -> None:
    base, _ = api()
    status, headers, body = call(base, "/api/parse", method="OPTIONS")
    assert status == 204
    assert body == b""
    assert headers["Access-Control-Allow-Methods"] == "POST, GET, OPTIONS"
    assert "X-Filename" in headers["Access-Control-Allow-Headers"]
    assert "X-Mode" in headers["Access-Control-Allow-Headers"]


def test_unknown_paths_are_never_parsed(api) -> None:
    base, _ = api()
    assert call(base, "/api/nope")[0] == 404
    assert call(base, "/", method="POST", body=b"x")[0] == 404
    assert call(base, "/api/parse")[0] == 404


def test_an_upload_without_a_body_is_refused(api) -> None:
    base, _ = api()
    status, _, body = call(base, "/api/parse", method="POST")
    assert status == 411
    assert payload(body) == {"error": "send the file as the request body"}


def test_an_upload_over_the_size_limit_is_refused_before_the_body(api) -> None:
    base, _ = api()
    status, _, body = call(base, "/api/parse", method="POST",
                           headers={"Content-Length": str(20 * 1024 * 1024 + 1)})
    assert status == 413
    assert payload(body) == {"error": "limit is 20 MB"}


def test_an_unknown_mode_names_the_two_choices(api) -> None:
    base, _ = api(service=StubService(document=sample_document()))
    status, _, body = call(
        base, "/api/parse", method="POST", body=b"x",
        headers={"Content-Type": "image/png", "X-Mode": "nope"},
    )
    assert status == 400
    assert payload(body) == {"error": "'nope' is not an extraction mode; use glm or fast"}


def test_a_comparison_is_capped_at_three_parsers(api) -> None:
    """Three GPUs at once is what the pool carries; a fourth row would only queue behind the others."""
    service = StubService(document=sample_document())
    extra = [
        EngineSpec(id=f"extra{n}", name=f"Extra {n}", kind="vlm", pipeline="p", device="d",
                   needs="a GPU")
        for n in (1, 2)
    ]
    service.registry = EngineRegistry(
        {"glm": GLM, "fast": FAST, **{spec.id: spec for spec in extra}}, "glm"
    )
    service._engines.update(
        {spec.id: type("Engine", (), {"spec": spec})() for spec in extra}
    )
    base, _ = api(service=service)

    status, _, body = call(
        base, "/api/parse", method="POST", body=b"x",
        headers={"Content-Type": "image/png", "X-Mode": "glm,fast,extra1,extra2"},
    )

    assert status == 400, "four parsers were accepted"
    assert payload(body) == {"error": "a comparison reads at most 3 parsers at once"}


def test_a_doubled_mode_is_not_a_second_parser(api) -> None:
    """The cap counts parsers, not commas: asking for the same one twice is still one row."""
    base, _ = api(service=StubService(document=sample_document()))
    status, _, _ = call(
        base, "/api/parse", method="POST", body=b"x",
        headers={"Content-Type": "image/png", "X-Mode": "glm,glm,glm,glm"},
    )
    assert status != 400


def test_glm_without_an_endpoint_says_so_and_offers_the_fast_path(api, tmp_path) -> None:
    base, _ = api(rig=build_rig(tmp_path, glm_available=False))
    status, _, body = call(
        base, "/api/parse", method="POST", body=b"x",
        headers={"Content-Type": "image/png", "X-Mode": "glm"},
    )
    assert status == 503
    assert payload(body)["error"] == (
        "GLM-OCR is not configured on this server; use the fast option (LiteParse + Tesseract)"
    )


def test_a_busy_server_asks_the_client_to_retry(api) -> None:
    base, _ = api(service=StubService(document=sample_document()), slots=FakeSlots(grant=False))
    status, _, body = call(base, "/api/parse", method="POST", body=b"x",
                           headers={"Content-Type": "image/png", "X-Mode": "glm"})
    assert status == 503
    assert payload(body) == {"error": "busy, try again in a moment"}


def test_the_parse_slot_is_always_released(api) -> None:
    slots = FakeSlots()
    base, container = api(service=StubService(error=ConnectionResetError("client went away")),
                          slots=slots)

    status, _, _ = call(base, "/api/parse", method="POST", body=b"x",
                        headers={"Content-Type": "image/png", "X-Mode": "glm"})
    assert status == 500

    deadline = time.monotonic() + 2
    while slots.released < slots.acquired and time.monotonic() < deadline:
        time.sleep(0.01)
    assert slots.released == slots.acquired == 1, "a failed read must not leak its slot"

    container.read_document.override(StubService(document=sample_document()))
    status, _, _ = call(base, "/api/parse", method="POST", body=b"x",
                        headers={"Content-Type": "image/png", "X-Mode": "glm"})
    assert status == 200, "the next upload must still be served"


def test_a_parse_returns_the_evidence_shape(api, png_bytes) -> None:
    stub = StubService(document=sample_document())
    base, _ = api(service=stub)
    status, headers, body = call(
        base, "/api/parse?mode=fast", method="POST", body=png_bytes,
        headers={"Content-Type": "image/png; charset=binary", "X-Filename": "form%20one.png"},
    )

    assert status == 200
    assert payload(body) == document_evidence(sample_document())
    assert headers["Access-Control-Allow-Origin"] == "*"
    source, mode, kwargs = stub.calls[0]
    assert source.name == "form one.png"
    assert source.media_type.value == "image/png"
    assert mode == "fast"
    assert kwargs["note"] == "uploaded just now · read by LiteParse + Tesseract"


def test_a_non_ascii_filename_survives_percent_encoding(api, png_bytes) -> None:
    stub = StubService(document=sample_document())
    base, _ = api(service=stub)
    call(
        base, "/api/parse", method="POST", body=png_bytes,
        headers={"Content-Type": "image/png", "X-Mode": "fast",
                 "X-Filename": urllib.parse.quote("扫描.pdf", safe="")},
    )
    assert stub.calls[0][0].name == "扫描.pdf"


def test_no_mode_means_chandra(api, png_bytes) -> None:
    from ocr_playground.infrastructure.engines import CHANDRA_SPEC

    stub = StubService(document=sample_document())
    stub.require_mode = lambda mode: type("Engine", (), {"spec": CHANDRA_SPEC})()
    base, _ = api(service=stub, glm_base_url="http://stub/v1")
    status, _, _ = call(base, "/api/parse", method="POST", body=png_bytes,
                        headers={"Content-Type": "image/png"})
    assert status == 200
    assert stub.calls[0][1] == "chandra"


def test_a_trailing_slash_on_the_endpoint_is_ignored(api) -> None:
    base, _ = api(glm_base_url="http://stub/v1/")
    _, _, body = call(base, "/api/health")
    assert payload(body)["modes"][0]["available"] is True


@pytest.mark.parametrize(
    ("error", "expected_status", "expected_error"),
    [
        (EngineUnavailable("no endpoint"), 503, "no endpoint"),
        (EngineTimeout("the GPU took too long"), 504, "the GPU took too long"),
        (EngineFailure("bad body"), 502, "bad body"),
        (CommandTimedOut("pdftoppm timed out"), 504, "parsing took too long"),
        (ValueError("that upload is empty"), 400, "that upload is empty"),
        (RuntimeError("boom"), 500, "could not read that file: boom"),
    ],
)
def test_reading_failures_map_to_status_codes(api, png_bytes, error, expected_status,
                                              expected_error) -> None:
    base, _ = api(service=StubService(error=error))
    status, _, body = call(base, "/api/parse", method="POST", body=png_bytes,
                           headers={"Content-Type": "image/png", "X-Mode": "glm"})
    assert status == expected_status
    assert payload(body) == {"error": expected_error}


def test_failure_response_maps_every_error_family() -> None:
    assert failure_response(CommandTimedOut("x")) == (504, "parsing took too long")
    assert failure_response(EngineUnavailable("a")) == (503, "a")
    assert failure_response(EngineTimeout("b")) == (504, "b")
    assert failure_response(EngineFailure("c")) == (502, "c")
    assert failure_response(UnknownMode("x", ("glm", "fast")))[0] == 400
    assert failure_response(ValueError("v")) == (400, "v")
    assert failure_response(RuntimeError("boom")) == (500, "could not read that file: boom")


def test_a_truncated_upload_is_reported_not_parsed(api) -> None:
    base, _ = api(service=StubService(document=sample_document()))
    host, port = base.removeprefix("http://").split(":")
    with socket.create_connection((host, int(port)), timeout=5) as connection:
        connection.sendall(
            b"POST /api/parse HTTP/1.1\r\nHost: test\r\nContent-Type: image/png\r\n"
            b"X-Mode: glm\r\nContent-Length: 100\r\n\r\nonly-ten-b"
        )
        connection.shutdown(socket.SHUT_WR)
        chunks = []
        while chunk := connection.recv(4096):
            chunks.append(chunk)
    response = b"".join(chunks).decode(errors="replace")
    assert " 408 " in response.splitlines()[0]
    assert "the upload ended early" in response


def test_health_payload_is_built_from_the_registry_and_engines() -> None:
    health = build_health(registry(), {"glm": _Engine(False), "fast": _Engine(True)},
                          max_upload_bytes=10, max_pages=2)
    assert [mode["id"] for mode in health["modes"]] == ["glm", "fast"]
    assert health["modes"][0]["available"] is False
    assert health["modes"][0]["name"] == "GLM-OCR"
    assert health["limits"] == {"bytes": 10, "pages": 2}


class _Engine:
    def __init__(self, available: bool) -> None:
        self.spec = FAST
        self._available = available

    @property
    def available(self) -> bool:
        return self._available


# --- the intent ping -------------------------------------------------------------------


def warmer_over(engines: dict, jobs: list, **policy: object) -> EngineWarmer:
    """A warmer with one engine per mode — what a page that ticked several parsers asks for."""
    return EngineWarmer(
        engines=engines,
        registry=EngineRegistry(
            {mode: one.spec for mode, one in engines.items()}, default_mode="glm",
        ),
        clock=FakeClock(),
        policy=WarmPolicy(**policy),  # type: ignore[arg-type]
        submit=jobs.append,
    )


def warmer_for(engine, jobs: list, **policy: object) -> EngineWarmer:
    return warmer_over({"glm": engine, "fast": FakeEngine(FAST)}, jobs, **policy)


def test_the_intent_ping_starts_the_gpu(api):
    engine = FakeWarmEngine(GLM)
    jobs: list = []
    base, container = api(glm_base_url="http://127.0.0.1:9/v1")
    container.warmer.override(warmer_for(engine, jobs))

    status, headers, body = call(base, "/api/warm", method="POST", headers={"X-Mode": "glm"})

    assert status == 200, body
    assert json.loads(body) == {"warm": "started"}
    assert headers.get("Access-Control-Allow-Origin") == "*"
    assert headers.get("Cache-Control") == "no-store"
    assert len(jobs) == 1


def test_a_warm_names_the_engine_it_started(api, capfd):
    """A warm rents GPU time from boot to scale-down, so the log has to say which engine it was for:
    "why did mineru start when I had unticked it" should not need the dashboard to answer."""
    engine = FakeWarmEngine(GLM)
    base, container = api(glm_base_url="http://127.0.0.1:9/v1")
    container.warmer.override(warmer_for(engine, []))

    call(base, "/api/warm", method="POST", headers={"X-Mode": "glm"})

    assert "warm glm: started" in capfd.readouterr().err


def test_a_second_ping_inside_the_interval_costs_nothing(api):
    base, container = api(glm_base_url="http://127.0.0.1:9/v1")
    container.warmer.override(warmer_for(FakeWarmEngine(GLM), [], min_interval_seconds=300))

    first = json.loads(call(base, "/api/warm", method="POST", headers={"X-Mode": "glm"})[2])
    second = json.loads(call(base, "/api/warm", method="POST", headers={"X-Mode": "glm"})[2])

    assert first == {"warm": "started"}
    assert second == {"warm": "throttled"}


def test_a_page_that_ticked_several_parsers_warms_each_one(api):
    """warmGpu() fires one POST per selected parser; a process-wide slot left the rest cold."""
    engines = {"chandra": FakeWarmEngine(CHANDRA), "glm": FakeWarmEngine(GLM),
               "fast": FakeEngine(FAST)}
    jobs: list = []
    base, container = api(
        chandra_base_url="http://127.0.0.1:9/v1",
        glm_base_url="http://127.0.0.1:9/v1",
    )
    container.warmer.override(warmer_over(engines, jobs))

    replies = [
        json.loads(call(base, "/api/warm", method="POST", headers={"X-Mode": mode})[2])
        for mode in ("chandra", "glm")  # the two cold parsers; the CPU path is never warmed
    ]

    assert replies == [{"warm": "started"}, {"warm": "started"}]
    assert len(jobs) == 2


def test_the_intent_ping_can_name_its_mode(api):
    base, container = api(glm_base_url="http://127.0.0.1:9/v1")
    jobs: list = []
    container.warmer.override(warmer_for(FakeWarmEngine(GLM), jobs))

    status, _, body = call(base, "/api/warm", method="POST", headers={"X-Mode": "fast"})

    assert status == 200
    assert json.loads(body) == {"warm": "not_warmable"}  # the local stack is always ready
    assert jobs == []


def test_the_intent_ping_rejects_a_mode_that_does_not_exist(api):
    base, _ = api(glm_base_url="http://127.0.0.1:9/v1")
    status, _, body = call(base, "/api/warm", method="POST", headers={"X-Mode": "magic"})
    assert status == 400
    assert "magic" in json.loads(body)["error"]


def test_the_intent_ping_is_a_post_only(api):
    base, _ = api(glm_base_url="http://127.0.0.1:9/v1")
    status, _, _ = call(base, "/api/warm")
    assert status == 404  # a crawler or a link prefetch must not rent a GPU


def test_warming_an_unconfigured_engine_is_refused(api):
    base, _ = api()  # no CHANDRA_OCR_BASE_URL, and Chandra is the default mode
    status, _, body = call(base, "/api/warm", method="POST")
    assert status == 503
    assert "not configured" in json.loads(body)["error"]


def test_a_warm_request_with_a_body_leaves_the_connection_usable(api):
    """HTTP/1.1 keeps the connection alive, so an unread body would become the next request line."""
    base, container = api(glm_base_url="http://127.0.0.1:9/v1")
    container.warmer.override(warmer_for(FakeWarmEngine(GLM), []))
    parts = urllib.parse.urlsplit(base)
    connection = http.client.HTTPConnection(parts.hostname, parts.port, timeout=10)
    headers = {"Content-Type": "application/json", "X-Mode": "glm"}
    try:
        connection.request("POST", "/api/warm", body=b'{"eager": true}', headers=headers)
        first = connection.getresponse()
        first.read()
        connection.request("POST", "/api/warm", body=b'{"eager": true}', headers=headers)
        second = connection.getresponse()
        second.read()
    finally:
        connection.close()

    assert (first.status, second.status) == (200, 200), second.read()


def test_a_rejected_mode_does_not_poison_the_next_request(api):
    """The parse path can answer before it reads the body; those bytes must not become a request."""
    base, _ = api(glm_base_url="http://stub/v1")  # chandra has no endpoint here
    parts = urllib.parse.urlsplit(base)
    connection = http.client.HTTPConnection(parts.hostname, parts.port, timeout=10)
    try:
        connection.request("POST", "/api/parse", body=b"%PDF-1.4 leftover", headers={
            "Content-Type": "application/pdf", "X-Mode": "chandra", "X-Filename": "x.pdf",
        })
        rejected = connection.getresponse()
        rejected.read()
        connection.request("GET", "/api/health")
        after = connection.getresponse()
        after.read()
    finally:
        connection.close()

    assert rejected.status == 503, rejected.status
    assert after.status == 200, "the next request must not inherit the rejected body"


def test_each_mode_names_its_own_engine_when_unconfigured(api):
    base, _ = api(glm_base_url="http://stub/v1")
    # MinerU is `hybrid`, so it is the one a kind-based ranking would send to another GPU.
    for mode, name in (("chandra", "Chandra OCR 2"), ("nemotron", "Nemotron 3.5 Parse"),
                       ("unlimited", "Unlimited-OCR"), ("mineru", "MinerU 2.5 (VLM)"),
                       ("paddle", "PP-StructureV3 (GPU)")):
        status, _, body = call(base, "/api/parse", method="POST", body=b"%PDF", headers={
            "Content-Type": "application/pdf", "X-Mode": mode, "X-Filename": "x.pdf",
        })
        assert status == 503, mode
        assert name in json.loads(body)["error"]
        assert "fast option" in json.loads(body)["error"]


def test_a_chunked_upload_is_refused_without_poisoning_the_connection(api):
    """No Content-Length can still mean bytes are coming; the refusal must close, not assume empty."""
    base, _ = api(glm_base_url="http://stub/v1")
    parts = urllib.parse.urlsplit(base)
    connection = http.client.HTTPConnection(parts.hostname, parts.port, timeout=10)
    try:
        connection.putrequest("POST", "/api/parse")
        connection.putheader("Content-Type", "application/pdf")
        connection.putheader("Transfer-Encoding", "chunked")
        connection.endheaders()
        connection.send(b"5\r\n%PDF-\r\n0\r\n\r\n")
        refused = connection.getresponse()
        refused.read()
        connection.request("GET", "/api/health")
        after = connection.getresponse()
        after.read()
    finally:
        connection.close()

    assert refused.status == 411, refused.status
    assert after.status == 200


def test_a_comma_separated_mode_returns_a_comparison(api):
    """One upload, several engines: the response carries a system and a cell for each."""
    base, _ = api(rig=build_rig(Path(tempfile.mkdtemp()), pages=1))
    status, _, body = call(base, "/api/parse", method="POST", body=b"%PDF", headers={
        "Content-Type": "application/pdf", "X-Mode": "glm,fast", "X-Filename": "x.pdf",
    })

    assert status == 200
    evidence = payload(body)
    assert [system["id"] for system in evidence["systems"]] == ["glm-ocr", "liteparse-tesseract"]
    assert set(evidence["pages"][0]["cells"]) == {"glm-ocr", "liteparse-tesseract"}
    assert evidence["document"]["note"] == (
        "uploaded just now · read side by side by GLM-OCR + LiteParse + Tesseract"
    )


def test_a_comparison_with_one_unconfigured_engine_is_refused(api):
    base, _ = api(glm_base_url="http://stub/v1")  # chandra has no endpoint here
    status, _, body = call(base, "/api/parse", method="POST", body=b"%PDF", headers={
        "Content-Type": "application/pdf", "X-Mode": "glm,chandra", "X-Filename": "x.pdf",
    })
    assert status == 503
    assert "Chandra OCR 2" in payload(body)["error"]


class HangingService:
    """A use case that hangs the way a cold GPU does, and records whether it was told to stop."""

    def __init__(self) -> None:
        self.started = threading.Event()
        self.cancelled = threading.Event()
        self._stub = StubService()

    def require_mode(self, mode: str):
        return self._stub.require_mode(mode)

    def execute(self, source, mode, *, cancel=None, **kwargs):
        self.started.set()
        while not (cancel and cancel()):
            time.sleep(0.01)
        self.cancelled.set()
        raise ClientGone()


def test_a_client_that_hangs_up_stops_the_parse(api) -> None:
    """A parse can hold its thread for minutes waiting on a boot. If the tab is closed the work must
    stop then and there, not run to the budget and answer into a socket that is already gone."""
    service = HangingService()
    base, _ = api(service=service)
    host, port = base.removeprefix("http://").split(":")
    body = b"\x89PNG" + b"x" * 64
    head = (
        f"POST /api/parse HTTP/1.1\r\nHost: {host}\r\nX-Mode: fast\r\n"
        f"Content-Type: image/png\r\nContent-Length: {len(body)}\r\n\r\n"
    ).encode()

    with socket.create_connection((host, int(port)), timeout=5) as client:
        client.sendall(head + body)
        assert service.started.wait(5), "the parse never started"
    # leaving the block closed the socket, which is exactly what closing the tab does
    assert service.cancelled.wait(5), "the parse kept running after the client hung up"
