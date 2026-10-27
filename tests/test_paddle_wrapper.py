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
from io import BytesIO
from pathlib import Path

from PIL import Image

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
            "serve_paddle_under_test", ROOT / "deploy" / "modal" / "serve_paddle.py"
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

    def predict(self, path, **options):
        self.path = Path(path)
        self.payload = self.path.read_bytes()
        return [_Result(self.markdown)]


def test_read_page_writes_the_exact_bytes_and_returns_markdown() -> None:
    markdown = "# page one " + "text " * 10
    pipeline = _Pipeline({"markdown_texts": markdown})

    assert wrapper.read_page(pipeline, PNG) == markdown
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

    assert wrapper.read_page(Probe({"markdown_texts": "x " * 20}), JPEG, lock) == "x " * 20
    assert seen == {"suffix": ".jpg", "locked": True}
    assert not lock.locked(), "the lock is released even though the page raised nothing"


def test_markdown_text_accepts_a_plain_string_result() -> None:
    assert wrapper.markdown_text(_Result("# raw markdown")) == "# raw markdown"
    assert wrapper.markdown_text(_Result({"markdown_texts": ""})) == ""


class _Sequence:
    """A pipeline that returns one reading per call and remembers each payload and option set."""

    def __init__(self, *readings: str) -> None:
        self.readings = list(readings)
        self.payloads: list[bytes] = []
        self.calls: list[dict] = []

    def predict(self, path, **options):
        self.payloads.append(Path(path).read_bytes())
        self.calls.append(options)
        return [_Result(self.readings[len(self.payloads) - 1])]


def _image_bytes(color: tuple[int, int, int] = (10, 20, 30), image_format: str = "PNG") -> bytes:
    buffer = BytesIO()
    Image.new("RGB", (8, 8), color).save(buffer, format=image_format)
    return buffer.getvalue()


def test_a_dark_page_becomes_an_inverted_png() -> None:
    inverted_bytes = wrapper.invert_dark_page(_image_bytes((10, 20, 30)))

    assert inverted_bytes is not None
    with Image.open(BytesIO(inverted_bytes)) as inverted:
        assert inverted.format == "PNG"
        assert inverted.size == (8, 8)
        assert inverted.convert("RGB").getpixel((0, 0)) == (245, 235, 225)


def test_a_dark_jpeg_also_comes_back_as_a_png() -> None:
    inverted_bytes = wrapper.invert_dark_page(_image_bytes((0, 0, 0), "JPEG"))

    assert inverted_bytes is not None
    with Image.open(BytesIO(inverted_bytes)) as inverted:
        assert inverted.format == "PNG"


def test_a_light_page_is_not_inverted() -> None:
    assert wrapper.invert_dark_page(_image_bytes((255, 255, 255))) is None


def test_the_brightness_boundary_is_exact() -> None:
    edge = wrapper.DARK_MEAN
    assert wrapper.invert_dark_page(_image_bytes((edge - 1,) * 3)) is not None
    assert wrapper.invert_dark_page(_image_bytes((edge,) * 3)) is None


def test_bytes_that_do_not_decode_are_left_alone() -> None:
    assert wrapper.invert_dark_page(PNG) is None, "the fake PNG fixture is not a decodable image"
    assert wrapper.invert_dark_page(b"not an image") is None


class _OcrPipeline:
    def __init__(self, result):
        self.result = result
        self.path: Path | None = None
        self.options: dict = {}

    def predict(self, path, **options):
        self.path = Path(path)
        self.options = options
        return [self.result]


class _OcrPage:
    def __init__(self, texts):
        self.rec_texts = texts


def test_ocr_lines_joins_rec_texts_at_the_larger_detection_limit() -> None:
    pipeline = _OcrPipeline({"rec_texts": ["first line", "second line"]})

    assert wrapper.ocr_lines(pipeline, PNG) == "first line\nsecond line"
    assert pipeline.path is not None
    assert pipeline.path.suffix == ".png"
    assert not pipeline.path.exists(), "the scratch directory is deleted before returning"
    assert pipeline.options == {
        "text_det_limit_side_len": wrapper.MAX_TEXT_SIDE,
        "text_det_limit_type": "max",
    }


def test_ocr_lines_accepts_an_attribute_result() -> None:
    pipeline = _OcrPipeline(_OcrPage(["a line"]))

    assert wrapper.ocr_lines(pipeline, JPEG) == "a line"
    assert pipeline.path is not None
    assert pipeline.path.suffix == ".jpg"


def test_ocr_lines_with_no_detections_is_empty() -> None:
    assert wrapper.ocr_lines(_OcrPipeline({}), PNG) == ""


def test_visible_characters_ignore_markup_and_placeholders() -> None:
    placeholder = '<div style="text-align: center;"><img src="imgs/box.jpg" alt="Image" /></div>'

    assert wrapper.visible_characters(placeholder) == 0
    assert wrapper.visible_characters("# 19950826 I1.FDIR T.S. IRIS Ferry") > wrapper.MIN_TEXT


def test_a_page_with_text_is_not_retried() -> None:
    pipeline = _Sequence("# " + "page text " * 12)

    def invert(payload):
        raise AssertionError("a page with text must not be inverted")

    assert wrapper.read_page(pipeline, PNG, invert=invert).startswith("# page text")
    assert pipeline.payloads == [PNG]
    assert pipeline.calls == [{}], "a normal page keeps the default detector limit"


def test_a_page_with_no_text_is_retried_inverted() -> None:
    pipeline = _Sequence("", "# recovered text from the light version")

    assert wrapper.read_page(pipeline, PNG, invert=lambda payload: b"INVERTED") == (
        "# recovered text from the light version"
    )
    assert pipeline.payloads == [PNG, b"INVERTED"]


def test_a_retry_that_finds_less_keeps_the_first_reading() -> None:
    placeholder = '<div style="text-align: center;"><img src="imgs/box.jpg" /></div>'
    pipeline = _Sequence(placeholder, "", "")

    assert wrapper.read_page(pipeline, PNG, invert=lambda payload: b"INVERTED", ocr=lambda p: "") == placeholder


def test_a_page_that_cannot_be_inverted_keeps_the_first_reading() -> None:
    placeholder = '<div style="text-align: center;"><img src="imgs/box.jpg" /></div>'
    pipeline = _Sequence(placeholder, "")

    assert wrapper.read_page(pipeline, PNG, invert=lambda payload: None, ocr=lambda p: "") == placeholder
    assert pipeline.payloads == [PNG]


def test_a_page_with_no_document_text_falls_back_to_plain_ocr() -> None:
    pipeline = _Sequence("# Using the model")

    text = wrapper.read_page(
        pipeline,
        PNG,
        invert=lambda payload: None,
        ocr=lambda payload: "# " + "screenshot body text " * 50,
    )

    assert text.startswith("# screenshot body text")
    assert pipeline.payloads == [PNG], "the layout pass is not repeated for the OCR fallback"
