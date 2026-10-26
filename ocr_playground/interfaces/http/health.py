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
    # Measured cold starts after the weights are cached; the first boot of an app pays for its
    # download on top, which is why the panel leads with the slower GLM figure.
    "chandra": {
        "cold_start": True,
        "detail": "Chandra 2 on a scale-to-zero GPU; about a minute cold, returns layout markdown",
    },
    "nemotron": {
        "cold_start": True,
        "detail": "Nemotron 3.5 Parse on a scale-to-zero GPU; half a minute cold, markdown with tables",
    },
    "unlimited": {
        "cold_start": True,
        "detail": "Unlimited-OCR on a scale-to-zero GPU; about a minute cold, then ~30 s a page",
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
