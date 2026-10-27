"""PP-StructureV3 on Modal with PaddleOCR's full server models, behind the same OpenAI-compatible
shape as the other parsers.

PP-StructureV3 is a pipeline, not a chat model: its documented entry point is
`PPStructureV3(...).predict(path)`, which returns layout, text, tables and reading order already
assembled. So this wrapper writes the posted page to a scratch file, runs the pipeline once, and
returns `result.markdown["markdown_texts"]` — the same call the batch probe in
`paddle_structure_gpu.py` makes, one page at a time.

Server models rather than the mobile set: the CPU row in this shootout fell back to
`PP-DocLayout-S` / `PP-OCRv5_mobile_*` to get under a minute a page, so those are the models a CPU
box can afford, not the pipeline's own. On this app's A10G (24 GB, the class the other rows run on
an L4) the configured defaults — `PP-DocLayout-L`, `PP-OCRv5_server_det`, `PP-OCRv5_server_rec` — are
what PP-StructureV3 is meant to run.

    modal deploy modal/serve_paddle.py
    PADDLE_OCR_BASE_URL=https://…/v1 make api
"""
import modal

# The batch probe already downloaded PaddleOCR's model set here, so this app boots warm on it.
paddlex_cache = modal.Volume.from_name("ocr-playground-paddlex", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.12")
    # PaddleOCR links against OpenCV and friends at import time; a slim image has none of them.
    .apt_install("libgl1", "libglib2.0-0", "libsm6", "libxext6", "libxrender1")
    .pip_install("paddlepaddle-gpu", index_url="https://www.paddlepaddle.org.cn/packages/stable/cu126/")
    .pip_install("paddleocr", "paddlex[ocr]", "fastapi[standard]")
)

app = modal.App("ocr-playground-paddle")
MODEL = "paddleocr"


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
        import base64
        import tempfile
        from pathlib import Path

        image_url = next(
            part["image_url"]["url"]
            for message in body.messages
            for part in message["content"]
            if part["type"] == "image_url"
        )
        payload = base64.b64decode(image_url.split(",", 1)[1])

        with tempfile.TemporaryDirectory() as scratch:
            suffix = ".jpg" if payload.startswith(b"\xff\xd8") else ".png"
            page = Path(scratch) / f"page{suffix}"
            page.write_bytes(payload)
            # One prediction per page: the pipeline is stateful about reading order and markdown
            # assembly, and a single-page call is exactly what the batch row recorded.
            with predict_lock:
                result = pipeline.predict(str(page))[0]

        markdown = result.markdown
        text = (
            markdown.get("markdown_texts", "")
            if isinstance(markdown, dict)
            else str(markdown or "")
        )
        return {"choices": [{"message": {"role": "assistant", "content": text}}]}

    return api
