"""PP-StructureV3 on Modal with PaddleOCR's full server models, behind the same OpenAI-compatible
shape as the other parsers.

PP-StructureV3 is a pipeline, not a chat model: its documented entry point is
`PPStructureV3(...).predict(path)`, which returns layout, text, tables and reading order already
assembled. So this wrapper writes the posted page to a scratch file, runs the pipeline once, and
returns `result.markdown["markdown_texts"]` — the same call the batch probe in
`paddle_structure_gpu.py` makes, one page at a time.

A screenshot of an application is outside the layout model's world: PP-DocLayout-L classifies the whole
page as one image, no text block reaches OCR, and only a stray title survives. A dark page (a dark-mode
screenshot, or a white-on-black scan) fails the same way. When the layout reading has almost no text,
the wrapper retries: first inverted if the page is dark, then with OCR alone — PP-OCRv5 detection and
recognition at a 2560 px side limit, so the small text of a 1920 px screenshot is not halved away — and
keeps whichever reading has more text. Pages that already read take only the first pass.

Server models rather than the mobile set: the CPU row in this shootout fell back to
`PP-DocLayout-S` / `PP-OCRv5_mobile_*` to get under a minute a page, so those are the models a CPU
box can afford, not the pipeline's own. On this app's A10G (24 GB, the class the other rows run on
an L4) the configured defaults — `PP-DocLayout-L`, `PP-OCRv5_server_det`, `PP-OCRv5_server_rec` — are
what PP-StructureV3 is meant to run.

    modal deploy deploy/modal/serve_paddle.py
    PADDLE_OCR_BASE_URL=https://…/v1 make api
"""
import base64
import re
import tempfile
import threading
from contextlib import nullcontext
from io import BytesIO
from pathlib import Path

import modal

# The batch probe already downloaded PaddleOCR's model set here, so this app boots warm on it.
paddlex_cache = modal.Volume.from_name("ocr-playground-paddlex", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.12")
    # PaddleOCR links against OpenCV and friends at import time; a slim image has none of them.
    .apt_install("libgl1", "libglib2.0-0", "libsm6", "libxext6", "libxrender1")
    # Pinned to the versions the recorded upload ran on: PPStructureV3's options and the markdown
    # result shape are not stable across minor releases, so a rebuild must not pick new ones.
    .pip_install("paddlepaddle-gpu==3.3.1", index_url="https://www.paddlepaddle.org.cn/packages/stable/cu126/")
    .pip_install(
        "paddleocr==3.7.0",
        "paddlex[ocr]==3.7.2",
        "fastapi[standard]==0.162.0",
        "pydantic==2.14.1",
        "pillow==12.1.0",
    )
)

app = modal.App("ocr-playground-paddle")
MODEL = "paddleocr"


def page_bytes(messages: list[dict]) -> bytes:
    """The base64 page carried by an OpenAI-style message list."""
    image_url = next(
        part["image_url"]["url"]
        for message in messages
        for part in (message.get("content") or [])
        if isinstance(part, dict) and part.get("type") == "image_url"
    )
    return base64.b64decode(image_url.split(",", 1)[1])


def markdown_text(result) -> str:
    """PP-StructureV3 puts markdown in a dict; older results are already a string."""
    markdown = result.markdown
    return markdown.get("markdown_texts", "") if isinstance(markdown, dict) else str(markdown or "")


_TAG = re.compile(r"<[^>]+>")
MIN_TEXT = 20
DARK_MEAN = 110
# PP-OCRv5's detector defaults to a 960 px side limit. A 1920 px browser screenshot is halved there,
# and 12-14 px body text is then too small to detect; 2560 keeps 4K screenshots above that floor too.
MAX_TEXT_SIDE = 2560


def visible_characters(text: str) -> int:
    """Letters and digits outside HTML tags: a placeholder `<img>` counts as no text."""
    return sum(character.isalnum() for character in _TAG.sub(" ", text))


def invert_dark_page(payload: bytes) -> bytes | None:
    """The page as a light one, or None when it is already light or cannot be decoded."""
    try:
        from PIL import Image, ImageOps, ImageStat
    except ImportError:  # pragma: no cover - the wrapper image pins Pillow
        return None
    try:
        with Image.open(BytesIO(payload)) as source:
            rgb = source.convert("RGB")
            if ImageStat.Stat(rgb.convert("L").resize((64, 64))).mean[0] >= DARK_MEAN:
                return None
            buffer = BytesIO()
            ImageOps.invert(rgb).save(buffer, format="PNG")
            return buffer.getvalue()
    except Exception:  # noqa: BLE001 - a page Pillow cannot decode is simply not inverted
        return None


def _predict(pipeline, payload: bytes, lock=None) -> str:
    suffix = ".jpg" if payload.startswith(b"\xff\xd8") else ".png"
    with tempfile.TemporaryDirectory() as scratch:
        page = Path(scratch) / f"page{suffix}"
        page.write_bytes(payload)
        with (lock or nullcontext()):
            result = pipeline.predict(str(page))[0]
    return markdown_text(result)


_ocr = None
_ocr_build = threading.Lock()


def load_ocr():
    """PP-OCRv5 without layout, built on the first page that needs it."""
    global _ocr
    with _ocr_build:
        if _ocr is None:
            from paddleocr import PaddleOCR

            _ocr = PaddleOCR(
                text_detection_model_name="PP-OCRv5_server_det",
                text_recognition_model_name="PP-OCRv5_server_rec",
                use_doc_orientation_classify=False,
                use_doc_unwarping=False,
                use_textline_orientation=True,
            )
    return _ocr


def ocr_lines(pipeline, payload: bytes, lock=None) -> str:
    """Every line PP-OCRv5 finds in the page, without layout."""
    suffix = ".jpg" if payload.startswith(b"\xff\xd8") else ".png"
    with tempfile.TemporaryDirectory() as scratch:
        page = Path(scratch) / f"page{suffix}"
        page.write_bytes(payload)
        with (lock or nullcontext()):
            result = pipeline.predict(
                str(page), text_det_limit_side_len=MAX_TEXT_SIDE, text_det_limit_type="max"
            )[0]
    try:
        texts = result["rec_texts"]
    except (KeyError, TypeError):
        texts = getattr(result, "rec_texts", [])
    return "\n".join(str(text) for text in texts)


def ocr_text(payload: bytes, lock=None) -> str:
    """The lazily built PP-OCRv5 pipeline, reading the page without layout."""
    return ocr_lines(load_ocr(), payload, lock)


def read_page(pipeline, payload: bytes, lock=None, invert=None, ocr=None) -> str:
    """Run one page and return its markdown; the scratch file is gone before the caller sees it."""
    source, text = payload, _predict(pipeline, payload, lock)
    if visible_characters(text) < MIN_TEXT:
        inverted = (invert or invert_dark_page)(payload)
        if inverted:
            alternate = _predict(pipeline, inverted, lock)
            if visible_characters(alternate) > visible_characters(text):
                source, text = inverted, alternate
    if visible_characters(text) < MIN_TEXT:
        alternate = ocr(source) if ocr is not None else ocr_text(source, lock)
        if visible_characters(alternate) > visible_characters(text):
            text = alternate
    return text


def load_pipeline():
    """The pipeline the batch probe runs, built once per container."""
    from paddleocr import PPStructureV3

    return PPStructureV3(
        layout_detection_model_name="PP-DocLayout-L",
        text_detection_model_name="PP-OCRv5_server_det",
        text_recognition_model_name="PP-OCRv5_server_rec",
        # Orientation and unwarping are separate models with their own cold-cache downloads; the
        # pages this reads are upright scans, and the batch row was recorded without them.
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=True,
        use_table_recognition=True,
        use_formula_recognition=True,
        # PP-DocLayout-L classes page 3 of this corpus as a chart. Without this branch the pipeline
        # crops it to an image and emits a placeholder with no text at all.
        use_chart_recognition=True,
    )


@app.function(
    image=image,
    # 24 GB, same class as the L4 the other rows use; the L4 pool was out of capacity when this
    # wrapper was deployed, and Modal's list fallback stayed pinned to its first entry.
    gpu="A10G",
    timeout=1800,
    scaledown_window=120,
    startup_timeout=2400,
    # The pipeline loads a dozen models; a second copy on a request burst would only contend for
    # host memory, and the predict lock already serialises predictions inside one container.
    max_containers=1,
    volumes={"/root/.paddlex": paddlex_cache},
)
@modal.asgi_app()
def web():
    from fastapi import FastAPI
    from pydantic import BaseModel

    # Built here, not on the first request: the container is only ready once the models are loaded,
    # so a warm ping and the first real page both find a pipeline that is already there.
    pipeline = load_pipeline()
    # One pipeline instance per container, and PPStructureV3's predictors carry per-call state.
    # FastAPI serves requests concurrently, so predictions take turns.
    import threading

    predict_lock = threading.Lock()

    api = FastAPI()

    class Chat(BaseModel):
        model: str = ""
        messages: list[dict] = []
        max_tokens: int = 8192
        temperature: float = 0.0

    @api.get("/v1/models")
    def models():
        return {"data": [{"id": MODEL}]}

    @api.post("/v1/chat/completions")
    def chat(body: Chat):
        # One prediction per page: the pipeline is stateful about reading order and markdown
        # assembly, and a single-page call is exactly what the batch row recorded.
        text = read_page(pipeline, page_bytes(body.messages), predict_lock)
        return {"choices": [{"message": {"role": "assistant", "content": text}}]}

    return api
