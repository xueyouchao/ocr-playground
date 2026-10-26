"""The extraction pipeline itself, with no HTTP around it.

One document in, per-page text out. `api/server.py` calls this for an upload; `indexer/flow.py` calls
the same functions through CocoIndex for a corpus on disk. Keeping the pipeline here is the point: the
live demo and the batch index must not drift into two different readers.

    glm   GLM-OCR, a 0.9B vision model behind an OpenAI-compatible endpoint. Best on handwriting and
          tables; the endpoint scales to zero, so the first request after idle can take minutes.
    fast  LiteParse + Tesseract on this machine's CPU: a couple of seconds per page, visibly weaker
          on the same pages, and the only path available with no endpoint configured.
"""
from __future__ import annotations

import base64
import json
import os
import re
import shutil
import subprocess
import tempfile
import time
import urllib.error
import urllib.request
from http import HTTPStatus
from pathlib import Path

MAX_PAGES = 5
PARSE_TIMEOUT_SECONDS = 120
RENDER_DPI = 200

# GLM-OCR is reached through any OpenAI-compatible endpoint; `modal/serve_glm_ocr.py` serves exactly
# that behind `modal deploy`. A cold start is a 303 to the same URL that then parks until the
# container is up, which is why these timeouts are minutes rather than seconds.
GLM_BASE_URL = os.environ.get("GLM_OCR_BASE_URL", "").rstrip("/")
GLM_MODEL = os.environ.get("GLM_OCR_MODEL", "glm-ocr")
GLM_API_KEY = os.environ.get("GLM_OCR_API_KEY", "") or "not-needed"
# "Text Recognition:" is what GLM-OCR's own recipe sends; a generic markdown instruction measurably
# changes what the smaller VLMs emit.
GLM_PROMPT = os.environ.get("GLM_OCR_PROMPT", "Text Recognition:")
GLM_REQUEST_TIMEOUT = int(os.environ.get("GLM_OCR_REQUEST_TIMEOUT", "900"))
# The whole document, cold start included. Five pages at ~20 s each is 100 s; the rest is boot.
GLM_BUDGET_SECONDS = int(os.environ.get("GLM_OCR_BUDGET_SECONDS", "900"))

PDF_TYPES = {"application/pdf", "application/x-pdf"}
IMAGE_TYPES = {"image/png", "image/jpeg", "image/tiff", "image/webp"}

LIT = shutil.which("lit") or "/home/ubuntu/ingestion-engine/.venv/bin/lit"

DEFAULT_MODE = "glm"

# The ids and labels match `systems.json`, so an uploaded pane looks like the recorded one.
SYSTEMS = {
    "glm": {
        "id": "glm-ocr",
        "name": "GLM-OCR",
        "kind": "vlm",
        "pipeline": "0.9B VLM on a Modal L4 (vLLM) — cold starts can take minutes",
        "device": "GPU (Modal L4)",
        "needs": "GLM_OCR_BASE_URL",
    },
    "fast": {
        "id": "liteparse-tesseract",
        "name": "LiteParse + Tesseract",
        "kind": "baseline",
        "pipeline": "raster + bundled Tesseract, parsed live on upload",
        "device": "this server (CPU)",
        "needs": "nothing",
    },
}


class UpstreamError(RuntimeError):
    """A failure worth reporting to the uploader, with the status it should arrive as."""

    def __init__(self, message: str, status: HTTPStatus) -> None:
        super().__init__(message)
        self.status = status


def run(command: list[str], timeout: int = PARSE_TIMEOUT_SECONDS) -> str:
    result = subprocess.run(command, capture_output=True, text=True, timeout=timeout)
    if result.returncode != 0:
        raise RuntimeError(f"{command[0]} failed: {result.stderr.strip()[:400]}")
    return result.stdout


def measure(text: str) -> dict:
    """The same counts collect.py records for the static document."""
    stripped = re.sub(r"<[^>]+>", " ", text)
    tokens = re.findall(r"\S+", stripped)
    numeric = [token for token in tokens if re.fullmatch(r"[\d.,:%/\-+()]+", token)]
    return {
        "chars": len(text),
        "tokens": len(tokens),
        "numeric_share": round(len(numeric) / len(tokens), 3) if tokens else 0.0,
    }


def parse_page(image: Path) -> str:
    return run([LIT, "parse", str(image), "--format", "markdown", "-q"]).strip()


def glm_page(image: Path, deadline: float) -> str:
    """One page through an OpenAI-compatible GLM-OCR endpoint, cold start included."""
    raw = image.read_bytes()
    # An upload keeps its own bytes, so the suffix is not proof of the format; the first two bytes
    # are. PDF pages and converted TIFF/WebP are PNG, an uploaded JPEG stays JPEG.
    mime = "image/jpeg" if raw.startswith(b"\xff\xd8") else "image/png"
    payload = {
        "model": GLM_MODEL,
        "temperature": 0,
        "max_tokens": 4096,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {
                    "url": f"data:{mime};base64," + base64.b64encode(raw).decode(),
                }},
                {"type": "text", "text": GLM_PROMPT},
            ],
        }],
    }
    data = json.dumps(payload).encode()
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {GLM_API_KEY}"}
    url = GLM_BASE_URL + "/chat/completions"

    # Modal answers a cold endpoint with a 303 to the same URL; following it *is* the cold-start wait.
    # urllib will not replay a POST, so the redirect is followed by hand rather than by urllib.
    body = None
    for _redirect in range(4):
        remaining = deadline - time.monotonic()
        if remaining <= 1:
            raise UpstreamError(
                "GLM-OCR did not finish within the time limit — try again once the GPU is warm, "
                "or use the fast option (LiteParse + Tesseract)",
                HTTPStatus.GATEWAY_TIMEOUT,
            )
        request = urllib.request.Request(url, data=data, headers=headers)
        try:
            with urllib.request.urlopen(request, timeout=min(GLM_REQUEST_TIMEOUT, remaining)) as response:
                body = json.loads(response.read().decode())
            break
        except urllib.error.HTTPError as error:
            if error.code in (303, 307, 308) and error.headers.get("Location"):
                url = error.headers["Location"]
                continue
            detail = error.read().decode(errors="replace")[:300]
            raise UpstreamError(
                f"the GLM-OCR endpoint answered {error.code}: {detail or 'no detail'}",
                HTTPStatus.BAD_GATEWAY,
            ) from error
        except (TimeoutError, urllib.error.URLError) as error:
            raise UpstreamError(
                f"the GLM-OCR endpoint did not answer: {error}",
                HTTPStatus.GATEWAY_TIMEOUT,
            ) from error

    if body is None:
        raise UpstreamError("the GLM-OCR endpoint kept redirecting", HTTPStatus.BAD_GATEWAY)
    try:
        content = body["choices"][0]["message"].get("content") or ""
    except (KeyError, IndexError, TypeError) as error:
        raise UpstreamError("the GLM-OCR endpoint returned an unexpected body", HTTPStatus.BAD_GATEWAY) from error
    return content.strip()


def preview_data_url(image: Path) -> str:
    target = image.with_suffix(".preview.jpg")
    run(["convert", str(image), "-resize", "1400x1400", "-quality", "78", str(target)])
    return "data:image/jpeg;base64," + base64.b64encode(target.read_bytes()).decode()


def render_pages(source: Path, scratch: Path, media_type: str, max_pages: int | None = MAX_PAGES) -> list[Path]:
    if media_type in IMAGE_TYPES:
        # Parse the bytes that arrived. Round-tripping a PNG through ImageMagick rewrites it as an
        # 8-bit grayscale palette image, and OCR reads half the page: 111 chars instead of 407 on the
        # sample form. Only the formats LiteParse cannot open take the conversion path.
        if media_type in {"image/png", "image/jpeg"}:
            return [source]
        single = scratch / "page-1.png"
        run(["convert", str(source), "-colorspace", "sRGB", str(single)])
        return [single]

    info = run(["pdfinfo", str(source)])
    match = re.search(r"^Pages:\s+(\d+)", info, re.MULTILINE)
    if not match:
        raise ValueError("could not read the page count from that PDF")
    count = int(match.group(1))
    # An upload is capped because it is a public request; an index run is not, unless the caller says
    # so, because capping a corpus silently would publish a document with pages missing.
    if max_pages is not None and count > max_pages:
        raise ValueError(f"that PDF has {count} pages; this demo reads at most {max_pages}")
    run(["pdftoppm", "-png", "-r", str(RENDER_DPI), str(source), str(scratch / "page")])
    return sorted(scratch.glob("page-*.png"), key=lambda path: int(path.stem.split("-")[1]))


def parse_document(
    payload: bytes,
    media_type: str,
    name: str,
    mode: str = DEFAULT_MODE,
    *,
    previews: bool = True,
    note: str | None = None,
    max_pages: int | None = MAX_PAGES,
) -> dict:
    """Read one document. The evidence shape is the one the page renders."""
    if media_type not in PDF_TYPES | IMAGE_TYPES:
        raise ValueError(f"{media_type or 'unknown type'} is not a PDF or an image")
    if not payload:
        raise ValueError("that upload is empty")
    system = SYSTEMS[mode]

    suffix = Path(name).suffix or (".pdf" if media_type in PDF_TYPES else ".png")
    with tempfile.TemporaryDirectory(prefix="ocr-playground-") as scratch_name:
        scratch = Path(scratch_name)
        source = scratch / f"upload{suffix}"
        source.write_bytes(payload)

        images = render_pages(source, scratch, media_type, max_pages)
        # One budget for the whole document: the cold start is paid on the first page and the rest
        # inherit what is left of it.
        deadline = time.monotonic() + GLM_BUDGET_SECONDS
        pages = []
        for number, image in enumerate(images, 1):
            text = glm_page(image, deadline) if mode == "glm" else parse_page(image)
            cell = {system["id"]: {"text": text, **measure(text)}}
            page = {"page": number, "cells": cell}
            if previews:
                page["preview"] = preview_data_url(image)
            pages.append(page)

    return {
        "document": {
            "name": Path(name).name or "upload",
            "pages": len(pages),
            "note": note or f"read by {system['name']}",
        },
        "systems": [system],
        "pages": pages,
    }
