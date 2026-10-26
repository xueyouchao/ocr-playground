"""Serve one OCR VLM on Modal as an OpenAI-compatible endpoint. Prototype, scale-to-zero.

    modal deploy modal/serve_glm_ocr.py
    # then: runners/vlm_endpoint.py --id glm-ocr --model glm-ocr --base-url https://…modal.run/v1
    modal app stop ocr-playground-glm-ocr      # when the row is recorded

The vLLM image is the upstream one: building vLLM from wheels on Modal costs ten minutes per cold
build, and the upstream image already carries a CUDA-matched vLLM.
"""
import modal

app = modal.App("ocr-playground-glm-ocr")

hf_cache = modal.Volume.from_name("ocr-playground-hf", create_if_missing=True)

# vLLM compiles the backbone with torch.compile on startup — 27 s of the ~60 s engine init — and
# writes the result to ~/.cache/vllm. That path is not mounted, so every scale-to-zero wake-up pays
# the compile again. Mounting it makes the first cold start the only one that compiles.
compile_cache = modal.Volume.from_name("ocr-playground-vllm-cache", create_if_missing=True)

# add_python: the upstream image ships vLLM's own interpreter, which Modal cannot fingerprint.
image = (
    modal.Image.from_registry("vllm/vllm-openai:latest", add_python="3.12")
    .entrypoint([])
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
)

MODEL = "zai-org/GLM-OCR"


@app.function(
    image=image,
    gpu="L4",                     # 24GB: a 0.9B VLM plus its KV cache fits with room to spare
    volumes={
        "/root/.cache/huggingface": hf_cache,
        # VLLM_CACHE_ROOT defaults to /root/.cache/vllm; the mount path has to match it.
        "/root/.cache/vllm": compile_cache,
    },
    timeout=3600,
    scaledown_window=300,
    startup_timeout=1800,
)
@modal.web_server(port=8000, startup_timeout=1800)
def serve() -> None:
    import subprocess

    subprocess.Popen(
        [
            "vllm", "serve", MODEL,
            "--served-model-name", "glm-ocr",
            "--host", "0.0.0.0",
            "--port", "8000",
            "--max-model-len", "16384",
            "--limit-mm-per-prompt", '{"image": 1}',
            "--gpu-memory-utilization", "0.90",
        ]
    )
