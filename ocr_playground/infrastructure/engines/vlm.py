"""The VLM parsers, all behind one OpenAI-compatible shape.

GLM-OCR is served by vLLM; Chandra OCR 2, Nemotron 3.5 Parse and Unlimited-OCR are each a small
FastAPI wrapper around the model's own documented entry point. All four answer
`POST /v1/chat/completions` with an image part in the message, so one adapter reads all four, and
what differs between them is the *spec* — who the model is, and what it costs to run.

Those wrappers ignore the client's text: each model was trained on its own task prompt, and passing
"convert to markdown" to them would measure a different model than the one their card ships. The
prompt below is therefore mostly an identity marker rather than an instruction (GLM-OCR is the
exception — it is a VLM with a real instruction, and the deployment's prompt is a setting).

Modal answers a cold endpoint with a `303` to the same URL carrying an attempt token, and following
that `Location` *is* the cold-start wait. urllib refuses to replay a POST on a 307/308, so the
redirect is followed by hand; the budget is what stops that loop from being unbounded.
"""
from __future__ import annotations

import base64
import json
import time
import urllib.error
import urllib.parse
import urllib.request

from ...application.errors import EngineFailure, EngineTimeout
from ...domain.document import PageImage
from ...domain.engine import EngineSpec

GLM_SPEC = EngineSpec(
    id="glm-ocr",
    name="GLM-OCR",
    kind="vlm",
    pipeline="0.9B VLM on a Modal L4 (vLLM) — cold starts take about 3 minutes",
    device="GPU (Modal L4)",
    needs="GLM_OCR_BASE_URL",
)

CHANDRA_SPEC = EngineSpec(
    id="chandra-2",
    name="Chandra OCR 2",
    kind="vlm",
    pipeline="Chandra 2 on a Modal L4 — layout-preserving markdown, its own ocr_layout task",
    device="GPU (Modal L4)",
    needs="CHANDRA_OCR_BASE_URL",
)

NEMOTRON_SPEC = EngineSpec(
    id="nemotron-parse",
    name="Nemotron 3.5 Parse",
    kind="vlm",
    pipeline="NVIDIA Nemotron 3.5 Parse on a Modal L4 — markdown with HTML tables",
    device="GPU (Modal L4)",
    needs="NEMOTRON_PARSE_BASE_URL",
)

MINERU_SPEC = EngineSpec(
    id="mineru-gpu",
    name="MinerU 2.5 (VLM)",
    kind="vlm",
    pipeline="MinerU2.5-1.2B with its layout and table models on a Modal L4 — page-by-page markdown",
    device="GPU (Modal L4)",
    needs="MINERU_BASE_URL",
)

UNLIMITED_SPEC = EngineSpec(
    id="unlimited-ocr",
    name="Unlimited-OCR",
    kind="vlm",
    pipeline="Baidu Unlimited-OCR on a Modal L4 — long pages, crop-and-stitch",
    device="GPU (Modal L4)",
    needs="UNLIMITED_OCR_BASE_URL",
)


class VlmEngine:
    """One VLM parser behind an OpenAI-compatible endpoint."""

    def __init__(
        self,
        *,
        spec: EngineSpec,
        base_url: str,
        model: str,
        prompt: str,
        api_key: str,
        request_timeout: float,
    ) -> None:
        self.spec = spec
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._prompt = prompt
        self._api_key = api_key
        self._request_timeout = request_timeout

    @property
    def available(self) -> bool:
        return bool(self._base_url)

    def identity(self) -> str:
        return (
            f"{self.spec.id} model={self._model} prompt={self._prompt!r} endpoint={self._base_url}"
        )

    def warm(self, timeout: float = 120) -> bool:
        """Ask a scale-to-zero endpoint to boot, without generating anything.

        `GET /models` is the cheapest thing each wrapper answers. Against a cold container the
        request simply hangs until the engine is up — that wait is the point: it happens here, while
        somebody is still choosing a file, instead of after they press read.
        """
        if not self.available:
            return False
        request = urllib.request.Request(
            self._base_url + "/models",
            headers={"Authorization": f"Bearer {self._api_key}"},
            method="GET",
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return response.status == 200
        except (OSError, urllib.error.URLError):
            return False

    def read(self, page: PageImage, deadline: float | None = None) -> str:
        raw = page.path.read_bytes()
        # An upload keeps its own bytes, so the suffix is not proof of the format; the first two
        # bytes are. PDF pages and converted TIFF/WebP are PNG, an uploaded JPEG stays JPEG.
        mime = "image/jpeg" if raw.startswith(b"\xff\xd8") else "image/png"
        payload = {
            "model": self._model,
            "temperature": 0,
            "max_tokens": 4096,
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {
                        "url": f"data:{mime};base64," + base64.b64encode(raw).decode(),
                    }},
                    {"type": "text", "text": self._prompt},
                ],
            }],
        }
        data = json.dumps(payload).encode()
        headers = {"Content-Type": "application/json", "Authorization": f"Bearer {self._api_key}"}
        url = self._base_url + "/chat/completions"
        budget = time.monotonic() + self._request_timeout * 4 if deadline is None else deadline

        body = None
        for _redirect in range(4):
            remaining = budget - time.monotonic()
            if remaining <= 1:
                raise EngineTimeout(
                    f"{self.spec.name} did not finish within the time limit — try again once the "
                    "GPU is warm, or use the fast option (LiteParse + Tesseract)"
                )
            request = urllib.request.Request(url, data=data, headers=headers)
            try:
                with urllib.request.urlopen(
                    request, timeout=min(self._request_timeout, remaining)
                ) as response:
                    body = json.loads(response.read().decode())
                break
            except urllib.error.HTTPError as error:
                if error.code in (303, 307, 308) and error.headers.get("Location"):
                    # A relative Location is legal in HTTP; Modal sends an absolute one.
                    url = urllib.parse.urljoin(url, error.headers["Location"])
                    continue
                detail = error.read().decode(errors="replace")[:300]
                raise EngineFailure(
                    f"the {self.spec.name} endpoint answered {error.code}: {detail or 'no detail'}"
                ) from error
            except (TimeoutError, urllib.error.URLError) as error:
                raise EngineTimeout(
                    f"the {self.spec.name} endpoint did not answer: {error}"
                ) from error

        if body is None:
            raise EngineFailure(f"the {self.spec.name} endpoint kept redirecting")
        try:
            content = body["choices"][0]["message"].get("content") or ""
        except (KeyError, IndexError, TypeError) as error:
            raise EngineFailure(
                f"the {self.spec.name} endpoint returned an unexpected body"
            ) from error
        return content.strip()
