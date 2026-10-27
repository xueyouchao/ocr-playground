"""MinerU on a GPU, its VLM backend (the one MinerU recommends when the hardware allows it).

The CPU row earlier used `-b pipeline -m ocr`, which is the mobile-ish path; this runs MinerU's own
vision-language backend on an L4 and returns its per-page markdown from the content list.

    modal run deploy/modal/mineru_gpu.py            # writes data/records/mineru-gpu.json
    modal run deploy/modal/mineru_gpu.py --backend pipeline
"""
import modal

app = modal.App("ocr-playground-mineru-gpu")

model_cache = modal.Volume.from_name("ocr-playground-mineru", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("libgl1", "libglib2.0-0", "libsm6", "libxext6", "libxrender1")
    .pip_install("torch", "torchvision", index_url="https://download.pytorch.org/whl/cu126")
    .pip_install("mineru[core]<3")  # 4.x replaced -p/-o with a server-backed CLI
    # Volumes cannot shadow a non-empty path like /root/.cache, so the model caches move to /models.
    .env({
        "MINERU_MODEL_SOURCE": "huggingface",
        "HF_HOME": "/models/huggingface",
        "MODELSCOPE_CACHE": "/models/modelscope",
    })
)


def single_page_pdf(pdf: bytes, number: int) -> bytes:
    import io

    from pypdf import PdfReader, PdfWriter

    writer = PdfWriter()
    writer.add_page(PdfReader(io.BytesIO(pdf)).pages[number - 1])
    buffer = io.BytesIO()
    writer.write(buffer)
    return buffer.getvalue()


def group_by_page(items: list) -> dict:
    pages: dict[int, list[str]] = {}
    for item in items:
        number = int(item.get("page_idx", 0)) + 1
        text = item.get("table_body") or item.get("text") or ""
        if text.strip():
            pages.setdefault(number, []).append(text.strip())
    return pages


@app.function(image=image, gpu="L4", timeout=3600, volumes={"/models": model_cache})
def parse(pdf: bytes, backend: str = "vlm-auto-engine") -> dict:
    import io
    import json
    import subprocess
    import tempfile
    from pathlib import Path

    from pypdf import PdfReader

    # Page by page: the VLM backend silently skips pages in a whole-document run (page 3 of this
    # scan vanished), and a single-page job cannot skip anything. It also keeps page_idx honest,
    # since every run starts at page 1.
    page_count = len(PdfReader(io.BytesIO(pdf)).pages)
    pages: dict[str, str] = {}
    for number in range(1, page_count + 1):
        text = ""
        # The VLM backend can return an empty page; it did so for page 3 in a whole-document run and
        # for page 5 here. Retry once, then fall back to the pipeline backend for that page only.
        for attempt in range(2):
            text = run_page(pdf, number, backend)
            if text:
                break
            print(f"page-{number}: attempt {attempt + 1} returned nothing", flush=True)
        if not text:
            text = run_page(pdf, number, "pipeline")
            print(f"page-{number}: fell back to pipeline, {len(text)} chars", flush=True)
        pages[f"page-{number}"] = text
        print(f"page-{number}: {len(text)} chars", flush=True)
    return pages


def run_page(pdf: bytes, number: int, backend: str) -> str:
    import json
    import subprocess
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as scratch:
        source = Path(scratch) / "page.pdf"
        source.write_bytes(single_page_pdf(pdf, number))
        command = ["mineru", "-p", str(source), "-o", scratch, "-b", backend]
        if backend.startswith("pipeline"):
            command += ["-m", "ocr"]
        run = subprocess.run(command, capture_output=True, text=True)
        lists = sorted(Path(scratch).rglob("*_content_list.json"))
        if run.returncode != 0 or not lists:
            produced = sorted(str(path.relative_to(scratch)) for path in Path(scratch).rglob("*"))[:20]
            raise SystemExit(
                f"page {number}: mineru rc={run.returncode}, {backend}\n"
                f"--- stdout ---\n{run.stdout[-1500:]}\n--- stderr ---\n{run.stderr[-1500:]}\n"
                f"--- wrote ---\n" + "\n".join(produced)
            )
        items = json.loads(lists[0].read_text(encoding="utf-8"))
    return "\n\n".join(
        str(item.get("table_body") or item.get("text") or "").strip()
        for item in items
        if str(item.get("text") or item.get("table_body") or "").strip()
    )


@app.local_entrypoint()
def main(backend: str = "vlm-auto-engine", pdf: str = ""):
    import json
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parents[2]
    if str(root / "src") not in sys.path:
        sys.path.insert(0, str(root / "src"))
    from ocr_playground.interfaces.site.layout import data

    default_pdf = data(root) / "corpus" / "document.pdf"
    source = Path(pdf) if pdf else default_pdf
    results = parse.remote(source.read_bytes(), backend)

    target = data(root) / "records" / "mineru-gpu.json"
    target.write_text(json.dumps(results, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"{target}: {sum(len(v) for v in results.values()):,} chars over {len(results)} pages")
