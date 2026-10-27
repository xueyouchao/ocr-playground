"""The VLM adapter: payload, redirects, budget and the way it names a failure.

A local stub stands in for the Modal endpoint; no GPU and no network.
"""
from __future__ import annotations

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from ocr_playground.application.errors import ClientGone, EngineFailure, EngineTimeout
from ocr_playground.domain.document import PageImage
from ocr_playground.infrastructure.engines import GLM_SPEC, VlmEngine, vlm
from ocr_playground.infrastructure.engines.vlm import MAX_UPLOADS


class Stub:
    """A scripted OpenAI-compatible endpoint that records what it was asked."""

    def __init__(self, script) -> None:
        self.requests: list[dict] = []
        self.gets: list[str] = []
        self.get_status = 200
        # A cold edge answers from the headers and closes while a large body is still in flight.
        self.drop_over: int | None = None
        self.drop_budget = 0
        self.dropped: list[str] = []
        # A probe the edge holds open while a container boots, then drops without answering.
        self.park_probes = 0
        self.park_for = 0.0
        self.parked: list[str] = []
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), self._handler(script))
        self.server.daemon_threads = True
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.server.server_address[1]}"

    def _handler(self, script):
        stub = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *args) -> None:
                pass

            def do_GET(self) -> None:  # noqa: N802 - the warm ping
                stub.gets.append(self.path)
                self.send_response(stub.get_status)
                self.send_header("Content-Length", "0")
                self.end_headers()

            def do_POST(self) -> None:  # noqa: N802
                length = int(self.headers.get("Content-Length") or 0)
                if stub.drop_over is not None and length > stub.drop_over and stub.drop_budget:
                    stub.drop_budget -= 1
                    stub.dropped.append(self.path)
                    self.close_connection = True
                    self.connection.close()  # before the page is read: what the edge does
                    return
                body = json.loads(self.rfile.read(length) or b"{}")
                if stub.park_probes and not body.get("messages"):
                    # Recorded before the hold: the client times out and carries on without us, so
                    # the assertion must not race the sleep.
                    stub.parked.append(self.path)
                    stub.park_probes -= 1
                    time.sleep(stub.park_for)
                    self.close_connection = True
                    return
                stub.requests.append({"path": self.path, "body": body})
                status, headers, payload = script(self.path, body)
                encoded = payload if isinstance(payload, bytes) else json.dumps(payload).encode()
                self.send_response(status)
                for key, value in headers.items():
                    self.send_header(key, value)
                self.send_header("Content-Length", str(len(encoded)))
                self.end_headers()
                self.wfile.write(encoded)

        return Handler

    def close(self) -> None:
        self.server.shutdown()
        self.server.server_close()


def completion(text: str) -> tuple[int, dict, dict]:
    return 200, {}, {"choices": [{"message": {"content": text}}]}


@pytest.fixture
def endpoint():
    created: list[Stub] = []

    def make(script, **overrides) -> tuple[Stub, VlmEngine]:
        stub = Stub(script)
        created.append(stub)
        settings = {
            "model": "glm-ocr",
            "prompt": "Text Recognition:",
            "api_key": "not-needed",
            "request_timeout": 30,
        }
        settings.update(overrides)
        return stub, VlmEngine(spec=GLM_SPEC, base_url=stub.url, **settings)

    yield make
    for stub in created:
        stub.close()


@pytest.fixture(autouse=True)
def _no_retry_pause(monkeypatch):
    """The pauses exist for a real wire; a test that sleeps them out only moves the clock."""
    monkeypatch.setattr(vlm, "_RETRY_PAUSE", 0.01)
    monkeypatch.setattr(vlm, "_HOP_PAUSE", 0.01)


def page(tmp_path: Path, payload: bytes = b"pngbytes", name: str = "page-1.png") -> PageImage:
    target = tmp_path / name
    target.write_bytes(payload)
    return PageImage(number=1, path=target)


def test_text_is_extracted_and_stripped(endpoint, tmp_path) -> None:
    stub, engine = endpoint(lambda path, body: completion("  the page text \n"))
    assert engine.read(page(tmp_path), time.monotonic() + 30) == "the page text"
    assert stub.requests[0]["path"] == "/chat/completions"


def test_payload_is_the_glm_recipe(endpoint, tmp_path) -> None:
    stub, engine = endpoint(lambda path, body: completion("x"))
    engine.read(page(tmp_path), time.monotonic() + 30)

    body = stub.requests[0]["body"]
    assert body["model"] == "glm-ocr"
    assert body["temperature"] == 0
    assert body["max_tokens"] == 4096
    assert body["messages"][0]["content"][1] == {"type": "text", "text": "Text Recognition:"}
    assert body["messages"][0]["content"][0]["image_url"]["url"].startswith("data:image/png;base64,")


def test_jpeg_is_announced_as_jpeg(endpoint, tmp_path) -> None:
    stub, engine = endpoint(lambda path, body: completion("x"))
    engine.read(page(tmp_path, b"\xff\xd8\xff\xe0jpeg", "upload.jpg"), time.monotonic() + 30)

    url = stub.requests[0]["body"]["messages"][0]["content"][0]["image_url"]["url"]
    assert url.startswith("data:image/jpeg;base64,")


@pytest.mark.parametrize("code", [303, 307, 308])
@pytest.mark.parametrize("relative", [False, True])
def test_a_handoff_is_walked_with_a_kilobyte_not_the_page(endpoint, tmp_path, code, relative) -> None:
    """The page must not ride a handoff: a cold edge drops a body still in flight (Errno 32).

    urllib would replay the POST as a GET with no body, which a wrapper answers 405 — after the boot
    is paid for — so the redirect is followed by hand. What follows it is a kilobyte whose only job
    is to learn whether a container is serving; the page goes up once, to one that is.
    """
    cold = {"yes": True}

    def script(path, body):
        if cold["yes"]:
            cold["yes"] = False
            location = "/attempt" if relative else stub.url + "/attempt"
            return code, {"Location": location}, b""
        if not body.get("messages"):
            return 400, {}, {"error": "messages is required"}
        return completion("read once a container answered")

    stub, engine = endpoint(script)
    stub.get_status = 405  # what a wrapper answers a bodyless GET

    assert engine.read(page(tmp_path), time.monotonic() + 30) == "read once a container answered"
    assert [request["path"] for request in stub.requests] == [
        "/chat/completions", "/attempt", "/chat/completions",
    ]
    assert stub.requests[1]["body"] == {"model": "glm-ocr", "messages": []}, "a kilobyte, not the page"
    assert stub.requests[2]["body"]["messages"][0]["content"][0]["image_url"], "the page, once"
    assert stub.gets == [], "the handoff must never be replayed as a GET"


def test_a_cold_start_may_hand_off_more_than_four_times(endpoint, tmp_path) -> None:
    """The budget bounds a cold start, not a hop count: a boot can hand off more than four times."""
    handoffs = {"n": 0}
    first = {"page": True}

    def script(path, body):
        if first["page"]:
            first["page"] = False
            return 303, {"Location": "/cold/1"}, b""
        if body.get("messages"):
            return completion("read after six handoffs")
        handoffs["n"] += 1
        if handoffs["n"] <= 6:
            return 303, {"Location": f"/cold/{handoffs['n'] + 1}"}, b""
        return 400, {}, {"error": "messages is required"}

    stub, engine = endpoint(script)
    assert engine.read(page(tmp_path), time.monotonic() + 30) == "read after six handoffs"
    assert handoffs["n"] == 7, "six handoffs, then a container answered"
    assert stub.requests[-1]["path"] == "/chat/completions", "the page goes to the base URL"


def test_a_redirect_loop_ends_at_the_budget(endpoint, tmp_path) -> None:
    """How many hops a cold start takes is the edge's business, so the deadline is the only bound.

    A hop that answers in a millisecond is not a boot in progress, so the pause keeps a broken edge
    from being dialled as fast as a socket allows — but it is a deadline that ends it, not a count.
    """
    holder: dict = {}

    def script(path, body):
        return 303, {"Location": holder["stub"].url + "/again"}, b""

    stub, engine = endpoint(script)
    holder["stub"] = stub

    with pytest.raises(EngineTimeout, match="time limit"):
        engine.read(page(tmp_path), time.monotonic() + 1.5)
    assert len(stub.requests) < 500, "a loop must not become a socket-speed spin"


def test_a_page_dropped_on_the_wire_is_retried_with_a_kilobyte(endpoint, tmp_path) -> None:
    """What the deployed box did: the edge closes while the page is still going up (Errno 32).

    The client learns nothing from a failed upload except that nothing is serving the request yet, so
    it asks again with a kilobyte, and puts the page up once a container answers.
    """
    def script(path, body):
        if not body.get("messages"):
            return 400, {}, {"error": "messages is required"}
        return completion("read on the retry")

    stub, engine = endpoint(script)
    stub.drop_over = 1024  # anything larger than a ping dies before it is read
    stub.drop_budget = 1  # once: the cold edge drops the first upload, then a container is up

    assert engine.read(page(tmp_path, b"png" + b"\0" * 4096), time.monotonic() + 30) == "read on the retry"

    assert stub.dropped == ["/chat/completions"], "the page was dropped exactly once"
    assert not stub.requests[0]["body"].get("messages"), "the retry asks with a kilobyte"
    assert stub.requests[1]["body"]["messages"][0]["content"][0]["image_url"], "then the page"


def test_a_flapping_endpoint_is_not_handed_the_page_forever(endpoint, tmp_path) -> None:
    """Every upload of the page is real GPU work, so a wire that keeps dropping it stops being offered.

    The probes keep succeeding — a container is there, it is the upload that dies — so without the cap
    the loop would happily re-send the page until the budget ran out.
    """
    def script(path, body):
        return 400, {}, {"error": "messages is required"}

    stub, engine = endpoint(script)
    stub.drop_over = 1024  # every page upload dies before it is read; the probes survive
    stub.drop_budget = 10_000

    with pytest.raises(EngineFailure, match="kept dropping the page"):
        engine.read(page(tmp_path, b"png" + b"\0" * 4096), time.monotonic() + 30)

    assert len(stub.dropped) == MAX_UPLOADS, "the page is offered exactly MAX_UPLOADS times"
    assert len(stub.requests) == MAX_UPLOADS, "one kilobyte probe per attempt, then it stops"
    assert all(not request["body"].get("messages") for request in stub.requests)


def test_a_probe_parked_by_the_edge_is_asked_again(endpoint, tmp_path, monkeypatch) -> None:
    """The edge may hold a kilobyte open while a container boots. That is not an answer and not a
    failure: the probe is asked again, and the page goes up once something answers."""
    monkeypatch.setattr(vlm, "_PROBE_SLICE", 0.15)
    cold = {"yes": True}

    def script(path, body):
        if cold["yes"] and body.get("messages"):
            cold["yes"] = False
            return 303, {"Location": "/cold"}, b""
        if not body.get("messages"):
            return 400, {}, {"error": "messages is required"}
        return completion("read after a parked probe")

    stub, engine = endpoint(script)
    stub.park_probes, stub.park_for = 1, 0.4

    assert engine.read(
        page(tmp_path, b"png" + b"\0" * 4096), time.monotonic() + 30
    ) == "read after a parked probe"
    assert stub.parked == ["/cold"], "the probe was the thing the edge parked"
    assert stub.requests[-1]["body"]["messages"], "the page went up once a container answered"


def test_a_parked_probe_notices_a_client_that_left(endpoint, tmp_path, monkeypatch) -> None:
    """A cancelled read must notice while it is parked, not after the edge finally answers."""
    monkeypatch.setattr(vlm, "_PROBE_SLICE", 0.15)
    started = time.monotonic()

    def script(path, body):
        if body.get("messages"):
            return 303, {"Location": "/cold"}, b""
        return 400, {}, {"error": "messages is required"}

    stub, engine = endpoint(script)
    stub.park_probes, stub.park_for = 5, 0.4

    def cancel() -> bool:
        return time.monotonic() - started > 0.05

    with pytest.raises(ClientGone):
        engine.read(page(tmp_path, b"png" + b"\0" * 4096), time.monotonic() + 30, cancel)
    assert time.monotonic() - started < 1.0, "it sat on the parked probe instead of noticing"


def test_a_cancelled_read_stops_walking_the_handoffs(endpoint, tmp_path) -> None:
    """The tab was closed while the engine was still handing off. Stop asking, and put the page on the
    wire no more than the once it already cost — the point is to stop paying, not to finish."""
    checks = {"n": 0}

    def cancel() -> bool:
        checks["n"] += 1
        return checks["n"] > 3

    stub, engine = endpoint(lambda path, body: (303, {"Location": "/cold"}, b""))

    with pytest.raises(ClientGone):
        engine.read(page(tmp_path, b"png" + b"\0" * 4096), time.monotonic() + 30, cancel)

    assert checks["n"] > 3, "the token was never asked"
    assert len(stub.requests) < 8, "it walked on instead of stopping"
    assert sum(1 for request in stub.requests if request["body"].get("messages")) == 1


def test_an_http_error_is_a_bad_gateway(endpoint, tmp_path) -> None:
    _, engine = endpoint(lambda path, body: (500, {}, b"model exploded"))
    with pytest.raises(EngineFailure) as caught:
        engine.read(page(tmp_path), time.monotonic() + 30)
    assert "answered 500" in str(caught.value)
    assert "model exploded" in str(caught.value)


def test_an_unexpected_body_is_named(endpoint, tmp_path) -> None:
    _, engine = endpoint(lambda path, body: (200, {}, {"unexpected": True}))
    with pytest.raises(EngineFailure, match="unexpected body"):
        engine.read(page(tmp_path), time.monotonic() + 30)


def test_an_exhausted_budget_explains_the_trade(endpoint, tmp_path) -> None:
    _, engine = endpoint(lambda path, body: completion("never reached"))
    with pytest.raises(EngineTimeout) as caught:
        engine.read(page(tmp_path), time.monotonic() + 0.5)
    assert "fast option" in str(caught.value)


def test_an_endpoint_that_never_answers_ends_at_the_budget(tmp_path) -> None:
    import socket

    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        dead_port = probe.getsockname()[1]

    engine = VlmEngine(spec=GLM_SPEC,
        base_url=f"http://127.0.0.1:{dead_port}", model="glm-ocr",
        prompt="Text Recognition:", api_key="k", request_timeout=5,
    )
    with pytest.raises(EngineTimeout, match="time limit"):
        engine.read(page(tmp_path), time.monotonic() + 1.5)  # a dropped wire is worth retrying


def test_availability_is_whether_an_endpoint_is_configured() -> None:
    def engine(base_url: str) -> VlmEngine:
        return VlmEngine(spec=GLM_SPEC, base_url=base_url, model="m", prompt="p", api_key="k",
                            request_timeout=1)

    assert engine("https://gpu.example/v1").available is True
    assert engine("").available is False


def test_identity_names_model_prompt_and_endpoint() -> None:
    engine = VlmEngine(spec=GLM_SPEC,
        base_url="https://gpu.example/v1", model="glm-ocr",
        prompt="Text Recognition:", api_key="k", request_timeout=1,
    )
    assert engine.identity() == (
        "glm-ocr model=glm-ocr prompt='Text Recognition:' endpoint=https://gpu.example/v1"
    )


# --- waking a scale-to-zero endpoint --------------------------------------------------


def test_warm_pings_the_cheapest_endpoint_without_generating(endpoint):
    stub, engine = endpoint(lambda path, body: completion("never"))
    assert engine.warm(5) is True
    assert stub.gets == ["/models"]  # GET, and nothing was generated
    assert stub.requests == []


def test_warm_reports_a_refused_endpoint_instead_of_raising(endpoint):
    stub, engine = endpoint(lambda path, body: completion("never"))
    stub.get_status = 503
    assert engine.warm(5) is False


def test_warm_is_false_when_no_endpoint_is_configured():
    engine = VlmEngine(spec=GLM_SPEC, base_url="", model="m", prompt="p", api_key="k", request_timeout=1)
    assert engine.warm(5) is False


def test_warm_is_false_when_the_host_is_unreachable():
    engine = VlmEngine(spec=GLM_SPEC, base_url="http://127.0.0.1:9/v1", model="m", prompt="p",
                          api_key="k", request_timeout=1)
    assert engine.warm(1) is False  # a dead endpoint must not fail the page that asked
