"""Which modes this deployment can run, for `GET /api/health`.

The page reads this to disable GLM-OCR when the container has no endpoint, and to select the fast
path rather than fail the first upload. The wording is interface copy — it explains a trade to a
person, not a fact about the domain — so it lives here and not in the engine.
"""
from __future__ import annotations

from collections.abc import Mapping

from ...domain.engine import EngineRegistry
from ...domain.ports import PageOcr

MODE_COPY: dict[str, dict[str, object]] = {
    "glm": {
        "cold_start": True,
        "detail": "0.9B VLM on a scale-to-zero GPU; the first upload after idle can take minutes",
    },
    "fast": {
        "cold_start": False,
        "detail": "CPU OCR on this server; a couple of seconds per page",
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
