"""PP-StructureV3 on a GPU with PaddleOCR's full server models.

The CPU row in this shootout had to fall back to the mobile models to get under a minute a page; this
one is what the pipeline actually is when it has the hardware it was designed for. Run in-container:
the first call downloads the model set and inference on a dense page can outlive an HTTP request.

    modal run modal/paddle_structure_gpu.py --paths /tmp/ocr-playground/pages/page-1.png,…
"""
import modal

app = modal.App("ocr-playground-paddle-gpu")

# PaddleOCR's model cache, so a second run does not re-download the server models.
paddlex_cache = modal.Volume.from_name("ocr-playground-paddlex", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("libgl1", "libglib2.0-0", "libsm6", "libxext6", "libxrender1")
    .pip_install("paddlepaddle-gpu", index_url="https://www.paddlepaddle.org.cn/packages/stable/cu126/")
    .pip_install("paddleocr", "paddlex[ocr]")
)


@app.function(image=image, gpu="L4", timeout=3600,
              volumes={"/root/.paddlex": paddlex_cache})
def parse(pages: dict) -> dict:
    from paddleocr import PPStructureV3

    pipeline = PPStructureV3(
        layout_detection_model_name="PP-DocLayout-L",
        text_detection_model_name="PP-OCRv5_server_det",
        text_recognition_model_name="PP-OCRv5_server_rec",
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=True,
        use_table_recognition=True,
        use_formula_recognition=True,
        # PP-DocLayout-L classes page 3 as a chart; without this branch the pipeline crops it to an
        # image and emits a placeholder with no text at all.
        use_chart_recognition=True,
    )
    results = {}
    for name, data in pages.items():
        target = f"/tmp/{name}.png"
        open(target, "wb").write(data)
        prediction = pipeline.predict(target)[0]
        markdown = prediction.markdown
        results[name] = markdown.get("markdown_texts", "") if isinstance(markdown, dict) else str(markdown)
        print(f"{name}: {len(results[name])} chars", flush=True)
    return results


@app.local_entrypoint()
def main(paths: str = ""):
    import json
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from ocr_playground.interfaces.site.layout import data

    files = {Path(item).stem: Path(item).read_bytes() for item in paths.split(",") if item}
    results = parse.remote(files)
    target = data(root) / "records" / "paddleocr-gpu.json"
    target.write_text(json.dumps(results, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"{target}: {sum(len(v) for v in results.values()):,} chars over {len(results)} pages")
