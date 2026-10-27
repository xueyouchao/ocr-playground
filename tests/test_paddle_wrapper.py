"""The Modal wrapper's page path, without Modal, PaddleOCR or FastAPI.

The FastAPI handler is glue; what can regress silently is decode -> scratch file -> pipeline ->
markdown. These tests load the wrapper by path with a stub `modal` module so those helpers run in
the offline suite, where none of the wrapper's real dependencies are installed.
"""
from __future__ import annotations

import base64
import importlib.util
import sys
import threading
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _load_wrapper():
    """Import serve_paddle.py with just enough of Modal for its decorators to run."""

    class Image:
        @classmethod
        def debian_slim(cls, **kwargs):
            return cls()

        def apt_install(self, *args, **kwargs):
            return self

        def pip_install(self, *args, **kwargs):
            return self

    class Volume:
        @classmethod
        def from_name(cls, *args, **kwargs):
            return cls()

    class App:
        def __init__(self, *args, **kwargs):
            pass

        def function(self, *args, **kwargs):
            return lambda fn: fn

    modal = types.ModuleType("modal")
    modal.Image = Image
    modal.Volume = Volume
    modal.App = App
    modal.asgi_app = lambda *args, **kwargs: (lambda fn: fn)

    previous = sys.modules.get("modal")
    sys.modules["modal"] = modal
    try:
        spec = importlib.util.spec_from_file_location(
            "serve_paddle_under_test", ROOT / "modal" / "serve_paddle.py"
        )
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    finally:
        if previous is None:
            del sys.modules["modal"]
        else:
            sys.modules["modal"] = previous
    return module


wrapper = _load_wrapper()

PNG = b"\x89PNG\r\n\x1a\n" + b"page bytes"
JPEG = b"\xff\xd8\xff\xe0" + b"page bytes"


def _data_url(payload: bytes, mime: str = "image/png") -> str:
    return f"data:{mime};base64,{base64.b64encode(payload).decode()}"


def test_page_bytes_decodes_the_image_part() -> None:
    messages = [
        {"role": "system", "content": "read this page"},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "pp-structurev3"},
                {"type": "image_url", "image_url": {"url": _data_url(PNG)}},
            ],
        },
    ]

    assert wrapper.page_bytes(messages) == PNG


class _Result:
    def __init__(self, markdown):
        self.markdown = markdown


class _Pipeline:
    def __init__(self, markdown):
        self.markdown = markdown
        self.path: Path | None = None
        self.payload = b""

    def predict(self, path):
        self.path = Path(path)
        self.payload = self.path.read_bytes()
        return [_Result(self.markdown)]


def test_read_page_writes_the_exact_bytes_and_returns_markdown() -> None:
    pipeline = _Pipeline({"markdown_texts": "# page one"})

    assert wrapper.read_page(pipeline, PNG) == "# page one"
    assert pipeline.payload == PNG
    assert pipeline.path is not None
    assert pipeline.path.suffix == ".png"
    assert not pipeline.path.exists(), "the scratch directory is deleted before returning"


def test_read_page_keeps_the_jpeg_suffix_and_holds_the_lock() -> None:
    lock = threading.Lock()
    seen: dict[str, object] = {}

    class Probe(_Pipeline):
        def predict(self, path):
            seen["suffix"] = Path(path).suffix
            seen["locked"] = lock.locked()
            return super().predict(path)

    assert wrapper.read_page(Probe({"markdown_texts": "x"}), JPEG, lock) == "x"
    assert seen == {"suffix": ".jpg", "locked": True}
    assert not lock.locked(), "the lock is released even though the page raised nothing"


def test_markdown_text_accepts_a_plain_string_result() -> None:
    assert wrapper.markdown_text(_Result("# raw markdown")) == "# raw markdown"
    assert wrapper.markdown_text(_Result({"markdown_texts": ""})) == ""
