"""Unlimited-OCR (Baidu) on Modal, transformers backend.

The card's entry point is `model.infer(...)`, which writes a result file per page; the wrapper calls
it exactly as documented and returns the file's text.
"""
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
    )
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
)

app = modal.App("ocr-playground-unlimited")
MODEL = "baidu/Unlimited-OCR"
PROMPT = "<image>document parsing."
_state: dict = {}


@app.function(image=image, gpu="L4", volumes={"/root/.cache/huggingface": hf_cache},
              timeout=1800, scaledown_window=300, startup_timeout=2400)
@modal.asgi_app()
def web():
    from fastapi import FastAPI
    from pydantic import BaseModel

    api = FastAPI()

    class Chat(BaseModel):
        model: str = ""
        messages: list[dict] = []
        max_tokens: int = 32768
        temperature: float = 0.0

    def load():
        import torch
        from transformers import AutoModel, AutoTokenizer

        _state["torch"] = torch
        _state["tokenizer"] = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
        _state["model"] = AutoModel.from_pretrained(
            MODEL, trust_remote_code=True, use_safetensors=True, torch_dtype=torch.bfloat16
        ).eval().cuda()

    @api.get("/v1/models")
    def models():
        return {"data": [{"id": "unlimited-ocr"}]}

    @api.post("/v1/chat/completions")
    def chat(body: Chat):
        import base64
        import tempfile
        from pathlib import Path

        if "model" not in _state:
            load()
        image_url = next(part["image_url"]["url"] for message in body.messages
                         for part in message["content"] if part["type"] == "image_url")
        with tempfile.TemporaryDirectory(prefix="unlimited-") as scratch:
            picture = Path(scratch) / "page.png"
            picture.write_bytes(base64.b64decode(image_url.split(",", 1)[1]))
            _state["model"].infer(
                _state["tokenizer"],
                prompt=PROMPT,
                image_file=str(picture),
                output_path=scratch,
                base_size=1024,
                image_size=640,
                crop_mode=True,
                max_length=32768,
                no_repeat_ngram_size=35,
                ngram_window=128,
                save_results=True,
            )
            written = sorted(Path(scratch).rglob("*"))
            text = ""
            for candidate in written:
                if candidate.suffix in {".md", ".txt", ".mmd"} and candidate.name != "page.png":
                    text += candidate.read_text(encoding="utf-8", errors="ignore")
        return {"choices": [{"message": {"role": "assistant", "content": text}}]}

    return api
