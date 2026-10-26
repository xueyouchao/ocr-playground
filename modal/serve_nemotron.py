"""Nemotron 3.5 Parse on Modal, transformers backend.

Its own card defines the task prompt; the model ignores a chat-style instruction, so the endpoint
supplies the control-token prompt rather than passing a visitor's text through.

    modal deploy serve_nemotron.py
    runners/vlm_endpoint.py --id nemotron-parse --model nemotron --base-url https://…/v1
"""
from pathlib import Path

import modal

hf_cache = modal.Volume.from_name("ocr-playground-hf", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch", "torchvision", index_url="https://download.pytorch.org/whl/cu126")
    .pip_install(
        "transformers==4.57.1",
        "accelerate",
        "pillow",
        "fastapi[standard]",
        "pymupdf",
        "einops",
        "addict",
        "easydict",
        "psutil",
        "matplotlib",
        "sentencepiece",
        "huggingface_hub[hf_transfer]",
        "timm",
        "beautifulsoup4",  # the model repo's postprocessing imports bs4
    )
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
    # NVIDIA's postprocessing helpers, downloaded from the model repo: they turn the raw
    # <class_X><bbox> stream into markdown with HTML tables, which is what the model card documents.
    .add_local_dir(Path(__file__).resolve().parent / "nemotron_helper", remote_path="/opt/nemotron")
)

app = modal.App("ocr-playground-nemotron")
MODEL = "nvidia/Nemotron-3.5-Parse"
PROMPT = "</s><s><predict_bbox><predict_classes><output_markdown><predict_text_in_pic>"
_state: dict = {}


def strip_chart_scaffolding(markdown: str) -> str:
    """Drop the chart wrapper the renderer emits before its table.

    `<chart type="multi_line">`, `subtitle:` and `bind: x ; ; x` are chart-plotting metadata, not
    document text; the table underneath them is the content and keeps its caption.
    """
    import re

    text = re.sub(r"</?chart\b[^>]*>\n?", "", markdown)
    lines = [line for line in text.splitlines() if not re.match(r"^\s*(subtitle|bind):", line)]
    return re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()


@app.function(image=image, gpu="L4", volumes={"/root/.cache/huggingface": hf_cache},
              timeout=1800, scaledown_window=120, startup_timeout=2400)
@modal.asgi_app()
def web():
    from fastapi import FastAPI
    from pydantic import BaseModel

    api = FastAPI()

    class Chat(BaseModel):
        model: str = ""
        messages: list[dict] = []
        max_tokens: int = 8744
        temperature: float = 0.0

    def load():
        import torch
        from transformers import AutoModel, AutoProcessor, GenerationConfig

        _state["torch"] = torch
        _state["model"] = AutoModel.from_pretrained(MODEL, trust_remote_code=True, dtype=torch.bfloat16).to("cuda").eval()
        _state["processor"] = AutoProcessor.from_pretrained(MODEL, trust_remote_code=True)
        config = GenerationConfig.from_pretrained(MODEL)
        config.max_new_tokens = 8744
        _state["config"] = config

    @api.get("/v1/debug")
    def debug():
        import importlib.util

        return {
            "timm": importlib.util.find_spec("timm") is not None,
            "transformers": __import__("transformers").__version__,
        }

    @api.get("/v1/models")
    def models():
        return {"data": [{"id": "nemotron"}]}

    @api.post("/v1/chat/completions")
    def chat(body: Chat):
        import base64
        import io

        from PIL import Image

        if "model" not in _state:
            load()
        import sys

        if "/opt/nemotron" not in sys.path:
            sys.path.insert(0, "/opt/nemotron")
        image_url = next(part["image_url"]["url"] for message in body.messages
                         for part in message["content"] if part["type"] == "image_url")
        picture = Image.open(io.BytesIO(base64.b64decode(image_url.split(",", 1)[1]))).convert("RGB")
        inputs = _state["processor"](images=[picture], text=PROMPT, return_tensors="pt",
                                     add_special_tokens=False).to("cuda")
        with _state["torch"].inference_mode():
            outputs = _state["model"].generate(**inputs, generation_config=_state["config"])
        raw = _state["processor"].batch_decode(outputs, skip_special_tokens=True)[0]

        from postprocessing import postprocess_document

        # Without this the pane shows the control-token stream and its normalised boxes; with it the
        # output is the markdown the model card says to use.
        document = postprocess_document(raw, image_size=picture.size)
        content = strip_chart_scaffolding(document["markdown"])
        return {"choices": [{"message": {"role": "assistant", "content": content}}]}

    return api
