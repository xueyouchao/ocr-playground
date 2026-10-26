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

from ocr_playground.application.errors import EngineFailure, EngineTimeout
from ocr_playground.domain.document import PageImage
from ocr_playground.infrastructure.engines import GLM_SPEC, VlmEngine


class Stub:
    """A scripted OpenAI-compatible endpoint that records what it was asked."""

    def __init__(self, script) -> None:
        self.requests: list[dict] = []
        self.gets: list[str] = []
        self.get_status = 200
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
                body = json.loads(self.rfile.read(length) or b"{}")
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


@pytest.mark.parametrize("relative", [False, True])
def test_a_307_is_replayed_by_hand_with_the_same_post(endpoint, tmp_path, relative) -> None:
    def script(path, body):
        if path == "/chat/completions":
            location = "/warm" if relative else stub.url + "/warm"
            return 307, {"Location": location}, b""
        return completion("warm text")

    stub, engine = endpoint(script)
    assert engine.read(page(tmp_path), time.monotonic() + 30) == "warm text"
    assert [request["path"] for request in stub.requests] == ["/chat/completions", "/warm"]
    assert stub.requests[0]["body"] == stub.requests[1]["body"], "the page must be re-sent"


def test_a_redirect_loop_is_reported(endpoint, tmp_path) -> None:
    holder: dict = {}

    def script(path, body):
        return 307, {"Location": holder["stub"].url + "/again"}, b""

    stub, engine = endpoint(script)
    holder["stub"] = stub

    with pytest.raises(EngineFailure) as caught:
        engine.read(page(tmp_path), time.monotonic() + 30)
    assert "kept redirecting" in str(caught.value)
    assert len(stub.requests) == 4


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


def test_an_unreachable_endpoint_is_a_timeout(tmp_path) -> None:
    import socket

    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        dead_port = probe.getsockname()[1]

    engine = VlmEngine(spec=GLM_SPEC,
        base_url=f"http://127.0.0.1:{dead_port}", model="glm-ocr",
        prompt="Text Recognition:", api_key="k", request_timeout=5,
    )
    with pytest.raises(EngineTimeout, match="did not answer"):
        engine.read(page(tmp_path), time.monotonic() + 30)


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
