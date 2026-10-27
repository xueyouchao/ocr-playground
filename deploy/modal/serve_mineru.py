"""MinerU 2.5 (VLM) on Modal, behind the same OpenAI-compatible shape as the other parsers.

MinerU has no server of its own: it is a CLI that reads a file and writes a `<name>_content_list.json`
beside it. So this wrapper writes the posted page to a scratch directory, runs the CLI once, and reads
the content list back — the same call the batch probe in `mineru_gpu.py` makes, one page at a time.

Page by page rather than document by document, for the reason the batch script records: the VLM
backend silently skipped pages in a whole-document run. A single-page job cannot skip anything.

The VLM backend can also return an empty page; it did so for two pages of this corpus. That page is
retried once and then read by the pipeline backend, which is slower but does not come back empty.

    modal deploy deploy/modal/serve_mineru.py
    MINERU_BASE_URL=https://…/v1 make api
"""
import modal

# The batch probe already downloaded MinerU's own models here, so the wrapper boots warm on them.
model_cache = modal.Volume.from_name("ocr-playground-mineru", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("libgl1", "libglib2.0-0", "libsm6", "libxext6", "libxrender1")
    .pip_install("torch", "torchvision", index_url="https://download.pytorch.org/whl/cu126")
    # colorlog rides along with `fastapi[standard]`, and its newest releases ship only an sdist whose
    # setup.py imports the module it is still defining — the image dies on a package MinerU merely
    # logs through. Pinned inside the same resolve, because a separate layer would only be upgraded
    # again. 6.8.2 has a wheel.
    .pip_install("colorlog==6.8.2", "fastapi[standard]")
    .pip_install("mineru[core]<3")  # 4.x replaced -p/-o with a server-backed CLI
    # A volume cannot shadow a non-empty path like /root/.cache, so the caches move to /models.
    .env({
        "MINERU_MODEL_SOURCE": "huggingface",
        "HF_HOME": "/models/huggingface",
        "MODELSCOPE_CACHE": "/models/modelscope",
    })
)

app = modal.App("ocr-playground-mineru")
BACKEND = "vlm-auto-engine"


def read_image(path, backend: str) -> str:
    """One page through MinerU's CLI, as the markdown it writes beside the content list."""
    import json
    import subprocess
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as scratch:
        command = ["mineru", "-p", str(path), "-o", scratch, "-b", backend]
        if backend.startswith("pipeline"):
            command += ["-m", "ocr"]
        run = subprocess.run(command, capture_output=True, text=True)
        lists = sorted(Path(scratch).rglob("*_content_list.json"))
        if run.returncode != 0 or not lists:
            raise RuntimeError(
                f"mineru rc={run.returncode} ({backend}): {(run.stderr or run.stdout)[-800:]}"
            )
        # The markdown file is the page as MinerU renders it, tables included. The content list is
        # the fallback: joining its items flattens a table into one paragraph per cell.
        markdown = [page for page in Path(scratch).rglob("*.md") if page.is_file()]
        if markdown:
            return markdown[0].read_text(encoding="utf-8").strip()
        items = json.loads(lists[0].read_text(encoding="utf-8"))

    return "\n\n".join(
        str(item.get("table_body") or item.get("text") or "").strip()
        for item in items
        if str(item.get("text") or item.get("table_body") or "").strip()
    )


@app.function(
    image=image,
    gpu=["L4", "A10G"],
    timeout=1800,
    scaledown_window=120,
    # One request per engine per upload, and a client that gives up asks again: without a cap the
    # retry buys a second copy of a model that is already loading. Extra requests queue inside the
    # one container instead of renting another GPU.
    max_containers=1,
    startup_timeout=2400,
    volumes={"/models": model_cache},
)
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
        return {"data": [{"id": "mineru"}]}

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
            # The VLM backend returns an empty page on some scans (two pages of this corpus). Retry
            # it once — the batch probe does the same — and only then pay for the pipeline backend,
            # which is slower but does not come back empty even when the VLM does so twice.
            text = ""
            for _ in range(2):
                text = read_image(page, BACKEND)
                if text.strip():
                    break
            if not text.strip():
                text = read_image(page, "pipeline")

        return {"choices": [{"message": {"role": "assistant", "content": text}}]}

    return api
