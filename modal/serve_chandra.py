"""Chandra OCR 2 on Modal through Datalab's own package, HF backend.

The card's documented path is `InferenceManager` with a prompt type, not a raw chat prompt: the model
was trained on named tasks, so passing "convert to markdown" would measure a different model than the
one Datalab ships. `prompt_type="ocr_layout"` is the layout-preserving markdown mode.

    modal deploy serve_chandra.py
    runners/vlm_endpoint.py --id chandra-2 --model chandra --base-url https://…/v1
"""
import modal

hf_cache = modal.Volume.from_name("ocr-playground-hf", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch", "torchvision", index_url="https://download.pytorch.org/whl/cu126")
    .pip_install(
        "transformers",  # Chandra 2 is qwen3_5; needs a transformers newer than the 4.x pin
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
    )
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
)

app = modal.App("ocr-playground-chandra")
MODEL = "datalab-to/chandra-ocr-2"
chandra_image = image.pip_install("chandra-ocr")
_state: dict = {}


@app.function(image=chandra_image, gpu="L4", volumes={"/root/.cache/huggingface": hf_cache},
              timeout=1800, scaledown_window=120, startup_timeout=2400)
@modal.asgi_app()
def web():
    from fastapi import FastAPI
    from pydantic import BaseModel

    api = FastAPI()

    class Chat(BaseModel):
        model: str = ""
        messages: list[dict] = []
        max_tokens: int = 8192
        temperature: float = 0.0

    @api.get("/v1/models")
    def models():
        return {"data": [{"id": "chandra"}]}

    @api.post("/v1/chat/completions")
    def chat(body: Chat):
        import base64
        import io

        from PIL import Image
        from chandra.model.hf import generate_hf, load_model
        from chandra.model.schema import BatchInputItem
        from chandra.output import parse_markdown

        if "model" not in _state:
            _state["model"] = load_model()
        image_url = next(part["image_url"]["url"] for message in body.messages
                         for part in message["content"] if part["type"] == "image_url")
        picture = Image.open(io.BytesIO(base64.b64decode(image_url.split(",", 1)[1]))).convert("RGB")
        result = generate_hf([BatchInputItem(image=picture, prompt_type="ocr_layout")], _state["model"])[0]
        # `generate_hf` returns GenerationResult, which carries only `raw`: the model's stream with a
        # data-bbox wrapper per region. The page shows text, so the package's own converter turns
        # that into the markdown its card documents — same layout, no literal markup in the cell.
        text = parse_markdown(result.raw)
        return {"choices": [{"message": {"role": "assistant", "content": text}}]}

    return api
