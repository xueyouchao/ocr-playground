"""The GPU parsers, all behind one OpenAI-compatible shape.

GLM-OCR is served by vLLM; Chandra OCR 2, Nemotron 3.5 Parse, Unlimited-OCR and MinerU 2.5 are each
a small FastAPI wrapper around the model's own documented entry point, and PP-StructureV3 is a
wrapper around PaddleOCR's pipeline. All six answer `POST /v1/chat/completions` with an image part
in the message, so one adapter reads them, and what differs is the *spec* — who the parser is, and
what it costs to run.

Those wrappers ignore the client's text: each model was trained on its own task prompt, and passing
"convert to markdown" to them would measure a different model than the one their card ships. The
prompt below is therefore mostly an identity marker rather than an instruction (GLM-OCR is the
exception — it is a VLM with a real instruction, and the deployment's prompt is a setting).

Modal answers a cold endpoint with a `303` to the same URL carrying an attempt token, and following
that `Location` *is* the cold-start wait. The redirect is therefore followed by hand: urllib would
replay a `303` *as a GET, without the body*, which the wrappers — POST-only routes — can only answer
with a 405, after paying for the whole boot. The page is not re-sent on that walk either — a cold edge
answers from the request headers and drops a body still in flight (`Errno 32`) — so it is walked with
a kilobyte until a container answers. The budget is the only bound: how many handoffs a boot needs is
the edge's business.
"""
from __future__ import annotations

import base64
import http.client
import json
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

from ...application.errors import ClientGone, EngineFailure, EngineTimeout
from ...domain.document import PageImage
from ...domain.engine import EngineSpec
from ...domain.ports import Cancel

# The statuses Modal uses to hand a cold request to the attempt URL that waits for the container.
_HANDOFFS = (303, 307, 308)
# A page that keeps dying on the wire is re-sent at most this often: each one is real GPU work, so a
# flapping endpoint must not turn one upload into ten.
MAX_UPLOADS = 6
# A handoff is how a cold start reports progress, and how many there are is the edge's business: one
# per attempt, over minutes, is legitimate. So the *budget* is the only bound on the chain; these
# pauses only keep a broken edge from being dialled as fast as the socket allows (a hop carries a
# kilobyte, and a hop that comes back in a millisecond is not a boot in progress).
_HOP_PAUSE = 0.05
_RETRY_PAUSE = 2.0
# A kilobyte probe that has not been answered in this long has told us nothing new: the edge is
# holding it while a container boots. Asking again is cheap and is the only thing that lets a
# cancelled read notice within seconds instead of sitting out the whole budget.
_PROBE_SLICE = 10.0

GLM_SPEC = EngineSpec(
    id="glm-ocr",
    name="GLM-OCR",
    kind="vlm",
    pipeline="0.9B VLM on a Modal L4 (vLLM) — cold starts take about 3 minutes",
    device="remote · GPU L4 (Modal)",
    needs="GLM_OCR_BASE_URL",
)

CHANDRA_SPEC = EngineSpec(
    id="chandra-2",
    name="Chandra OCR 2",
    kind="vlm",
    pipeline="Chandra 2 on a Modal L4 — layout-preserving markdown, its own ocr_layout task",
    device="remote · GPU L4 (Modal)",
    needs="CHANDRA_OCR_BASE_URL",
)

NEMOTRON_SPEC = EngineSpec(
    id="nemotron-parse",
    name="Nemotron 3.5 Parse",
    kind="vlm",
    pipeline="NVIDIA Nemotron 3.5 Parse on a Modal L4 — markdown with HTML tables",
    device="remote · GPU L4 (Modal)",
    needs="NEMOTRON_PARSE_BASE_URL",
)

MINERU_SPEC = EngineSpec(
    id="mineru-gpu",
    name="MinerU 2.5 (VLM)",
    kind="hybrid",
    pipeline="MinerU2.5-1.2B with its layout and table models on a Modal L4 — page-by-page markdown",
    device="remote · GPU L4 (Modal)",
    needs="MINERU_BASE_URL",
)

# A pipeline rather than a chat model, but it answers through the same OpenAI-compatible shape as
# the rest, so one adapter reads it too. Separate layout, OCR, table and formula models run in
# sequence, so it is `staged` like the CPU path; the upload runs on an A10G rather than the L4 the
# recorded row used, because that pool was dry at deploy time.
PADDLE_SPEC = EngineSpec(
    id="paddleocr-gpu",
    name="PP-StructureV3 (GPU)",
    kind="staged",
    pipeline="PaddleOCR's server models on a Modal A10G — layout, OCR, tables, formulas, reading order",
    device="remote · GPU A10G (Modal)",
    needs="PADDLE_OCR_BASE_URL",
)

UNLIMITED_SPEC = EngineSpec(
    id="unlimited-ocr",
    name="Unlimited-OCR",
    kind="vlm",
    pipeline="Baidu Unlimited-OCR on a Modal L4 — long pages, crop-and-stitch",
    device="remote · GPU L4 (Modal)",
    needs="UNLIMITED_OCR_BASE_URL",
)


def _pause(seconds: float, cancel: Cancel | None) -> None:
    """Sleep, but wake up for a client that has gone: a cancelled read must not sit out its backoff."""
    until = time.monotonic() + max(0.0, seconds)
    while True:
        left = until - time.monotonic()
        if left <= 0:
            return
        if cancel is not None and cancel():
            raise ClientGone()
        time.sleep(min(left, 0.25))


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    """Hand a redirect back to `read` instead of replaying the request behind its back.

    urllib answers a `303` by re-sending the request as a GET with no body, so a cold endpoint's
    handoff would arrive at the attempt URL as a bodyless GET on a POST-only route. Returning `None`
    leaves the error to the caller, which replays the POST itself, body and all.
    """

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


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

    def warm(self, timeout: float = 300) -> bool:
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

    def read(
        self, page: PageImage, deadline: float | None = None, cancel: Cancel | None = None
    ) -> str:
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
        # What loses a cold read is the upload, not the redirect: the edge answers a cold request
        # from the headers alone and drops the connection while a multi-megabyte page is still going
        # up, which the client sees as a broken pipe. So the page rides only a request that is
        # expected to be served: a cold endpoint is first asked with a kilobyte, and the page goes up
        # once a container has answered something.
        ping = json.dumps({"model": self._model, "messages": []}).encode()
        headers = {"Content-Type": "application/json", "Authorization": f"Bearer {self._api_key}"}
        url = self._base_url + "/chat/completions"
        budget = time.monotonic() + self._request_timeout * 4 if deadline is None else deadline
        opener = urllib.request.build_opener(_NoRedirect())

        ready = True  # optimistic: an endpoint that is up answers the page straight away
        uploads = failures = 0
        while True:
            remaining = budget - time.monotonic()
            if remaining <= 1:
                raise EngineTimeout(
                    f"{self.spec.name} did not finish within the time limit — try again once the "
                    "GPU is warm, or use the fast option (LiteParse + Tesseract)"
                )
            if cancel is not None and cancel():
                raise ClientGone()
            if ready:
                uploads += 1
                if uploads > MAX_UPLOADS:
                    raise EngineFailure(f"the {self.spec.name} endpoint kept dropping the page")
            request = urllib.request.Request(
                url, data=data if ready else ping, headers=headers, method="POST"
            )
            # A page is given the time it needs — inference is slow — but a probe is only a question,
            # so it is asked in slices that end early enough for a cancelled read to be noticed.
            wait = min(self._request_timeout, remaining)
            try:
                with opener.open(request, timeout=wait if ready else min(wait, _PROBE_SLICE)) as response:
                    answer = json.loads(response.read().decode())
            except TimeoutError:
                if ready:
                    raise EngineTimeout(
                        f"the {self.spec.name} endpoint did not answer within the time limit"
                    ) from None
                # Parked with a kilobyte in hand: still booting, or still nobody home. Ask again.
                url, failures = self._base_url + "/chat/completions", failures + 1
                _pause(min(_RETRY_PAUSE, max(0.0, budget - time.monotonic() - 1)), cancel)
                continue
            except urllib.error.HTTPError as error:
                # A cold start hands off however many times the container needs; only the budget,
                # checked above, decides when that has gone on too long.
                location = error.headers.get("Location") if error.code in _HANDOFFS else None
                detail = "" if location else error.read().decode(errors="replace")[:300]
                error.close()
                if location:
                    # A relative Location is legal in HTTP; Modal sends an absolute one.
                    url = urllib.parse.urljoin(url, location)
                    ready = False  # nothing is serving this yet, so stop sending the page
                    left = budget - time.monotonic() - 1  # the hop may have used what was left
                    if left > 0:
                        _pause(min(_HOP_PAUSE, left), cancel)
                    continue
                if not ready:
                    # A 400 to the kilobyte is still a container reading it, so one is serving now.
                    ready, url = True, self._base_url + "/chat/completions"
                    continue
                raise EngineFailure(
                    f"the {self.spec.name} endpoint answered {error.code}: {detail or 'no detail'}"
                ) from error
            except (OSError, http.client.HTTPException) as error:
                if ready:  # the page died on the wire; say so, it is the interesting failure
                    sys.stderr.write(
                        f"{self.spec.id}: the page upload was dropped ({error}); retrying with a "
                        "kilobyte until a container answers\n"
                    )
                ready, url = False, self._base_url + "/chat/completions"
                failures += 1
                # Back off: a boot is minutes long, so hammering it only wastes connections, while
                # an endpoint that is genuinely gone should not be dialled 400 times.
                pause = min(_RETRY_PAUSE * 2 ** min(failures - 1, 4), 30.0)
                # The clock is read again here: the request that just failed may have eaten the rest
                # of the budget, and a timeout the caller is waiting on must not be delayed by a sleep
                # that was sized before the request ran.
                left = budget - time.monotonic() - 1
                if left > 0:
                    _pause(min(pause, left), cancel)
                continue

            if not ready:
                ready, url = True, self._base_url + "/chat/completions"
                continue
            body = answer
            break

        try:
            content = body["choices"][0]["message"].get("content") or ""
        except (KeyError, IndexError, TypeError) as error:
            raise EngineFailure(
                f"the {self.spec.name} endpoint returned an unexpected body"
            ) from error
        return content.strip()
