"""Which modes this deployment can run, for `GET /api/health`.

The page reads this to disable GLM-OCR when the container has no endpoint, and to select the fast
path rather than fail the first upload. The wording is interface copy — it explains a trade to a
person, not a fact about the domain — so it lives here and not in the engine.
"""
from __future__ import annotations

from collections.abc import Mapping

from ...domain.engine import EngineRegistry
from ...domain.ports import PageOcr

_GPU_COLD = "scale-to-zero GPU; the first upload after idle takes about 3 minutes"

MODE_COPY: dict[str, dict[str, object]] = {
    "glm": {
        "cold_start": True,
        "detail": f"0.9B VLM on a {_GPU_COLD}",
    },
    "fast": {
        "cold_start": False,
        "detail": "CPU OCR on this server; a couple of seconds per page",
    },
    # Per-page costs measured from the wrappers' own request durations. These differ by an order of
    # magnitude, which is the point of showing them: a parser that takes a minute a page is a
    # different choice from one that takes five seconds, and the comparison waits for the slowest.
    "chandra": {
        "cold_start": True,
        "detail": "Chandra 2 on a scale-to-zero GPU; the slowest here, about a minute a page",
    },
    "nemotron": {
        "cold_start": True,
        "detail": "Nemotron 3.5 Parse on a scale-to-zero GPU; about 5 s a warm page, markdown with tables",
    },
    "unlimited": {
        "cold_start": True,
        "detail": "Unlimited-OCR on a scale-to-zero GPU; about 45 s a page, crops and stitches long pages",
    },
    "mineru": {
        "cold_start": True,
        "detail": "MinerU 2.5 on a scale-to-zero GPU; its VLM backend, about 70 s a page",
    },
    "paddle": {
        "cold_start": True,
        "detail": "PP-StructureV3 on a scale-to-zero A10G; server models with tables and charts, about 2 s a warm page",
    },
}


def build_health(
    registry: EngineRegistry,
    engines: Mapping[str, PageOcr],
    *,
    max_upload_bytes: int,
    max_pages: int,
) -> dict:
    modes = []
    for mode in registry.modes:
        spec = registry.spec(mode)
        copy = MODE_COPY.get(mode, {})
        modes.append({
            "id": mode,
            "name": spec.name,
            "available": engines[mode].available,
            "cold_start": copy.get("cold_start", False),
            "detail": copy.get("detail", spec.pipeline),
        })
    return {
        "ok": True,
        "default_mode": registry.default_mode,
        "modes": modes,
        "limits": {"bytes": max_upload_bytes, "pages": max_pages},
    }
