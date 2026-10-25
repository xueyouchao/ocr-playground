"""Run Chandra on named page images *inside* Modal and bring the text back.

The web endpoint is fine for ordinary pages, but the two dense handwritten pages outlive the edge
proxy's patience for a single request. `modal run` has no such limit: the function does the work and
returns the result.
"""
import modal

app = modal.App("ocr-playground-chandra-pages")

hf_cache = modal.Volume.from_name("ocr-playground-hf", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch", "torchvision", index_url="https://download.pytorch.org/whl/cu126")
    .pip_install(
        "transformers",
        "accelerate",
        "pillow",
        "fastapi[standard]",
        "einops",
        "addict",
        "easydict",
        "psutil",
        "matplotlib",
        "sentencepiece",
        "huggingface_hub[hf_transfer]",
        "chandra-ocr",
    )
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
)


@app.function(image=image, gpu="L4", volumes={"/root/.cache/huggingface": hf_cache}, timeout=3600)
def parse(pages: dict) -> dict:
    import io

    from PIL import Image
    from chandra.model.hf import generate_hf, load_model
    from chandra.model.schema import BatchInputItem

    model = load_model()
    out = {}
    for name, data in pages.items():
        picture = Image.open(io.BytesIO(data)).convert("RGB")
        result = generate_hf([BatchInputItem(image=picture, prompt_type="ocr_layout")], model)[0]
        out[name] = result.raw  # GenerationResult carries raw/html/markdown; raw is the model output
        print(f"{name}: {len(out[name])} chars", flush=True)
    return out


@app.local_entrypoint()
def main(paths: str = ""):
    from pathlib import Path

    files = {Path(item).stem: Path(item).read_bytes() for item in paths.split(",") if item}
    for name, text in parse.remote(files).items():
        print(f"RESULT {name} {len(text)}")
        Path(f"/tmp/chandra-{name}.md").write_text(text, encoding="utf-8")
