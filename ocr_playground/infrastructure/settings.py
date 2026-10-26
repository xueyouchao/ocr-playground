"""Configuration, read from the environment once and then injected.

Frozen and explicit: every knob the program has is a field here, so a test overrides a value rather
than patching a module global, and the container can be handed a different mapping without touching
`os.environ`.
"""
from __future__ import annotations

import dataclasses
import os
import shutil
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

DEFAULT_LIT = "/home/ubuntu/ingestion-engine/.venv/bin/lit"


@dataclass(frozen=True, slots=True)
class Settings:
    # The GLM-OCR endpoint: any OpenAI-compatible server. Empty means "fast path only".
    glm_base_url: str = ""
    glm_model: str = "glm-ocr"
    glm_api_key: str = "not-needed"
    glm_prompt: str = "Text Recognition:"
    glm_request_timeout: int = 900

    # One budget for a whole document, cold start included. Five pages at ~20 s is 100 s; the rest
    # is boot.
    glm_budget_seconds: int = 900

    # Upload bounds. The defaults are the public deployment's; the indexer passes its own page limit.
    max_pages: int = 5
    max_upload_bytes: int = 20 * 1024 * 1024
    parse_slots: int = 2
    slot_wait_seconds: int = 30
    upload_read_timeout: int = 30

    # Rendering and previews.
    render_dpi: int = 200
    parse_timeout_seconds: int = 120
    preview_max_px: int = 1400
    preview_quality: int = 78

    default_mode: str = "glm"
    lit_path: str = ""

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> Settings:
        values = os.environ if env is None else env
        lit = values.get("LIT_CLI") or shutil.which("lit") or DEFAULT_LIT
        return cls(
            glm_base_url=values.get("GLM_OCR_BASE_URL", "").rstrip("/"),
            glm_model=values.get("GLM_OCR_MODEL", "glm-ocr"),
            glm_api_key=values.get("GLM_OCR_API_KEY", "") or "not-needed",
            glm_prompt=values.get("GLM_OCR_PROMPT", "Text Recognition:"),
            glm_request_timeout=int(values.get("GLM_OCR_REQUEST_TIMEOUT", "900")),
            glm_budget_seconds=int(values.get("GLM_OCR_BUDGET_SECONDS", "900")),
            lit_path=lit,
        )

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> Settings:
        """Build from a configuration mapping, coercing to each field's type.

        This is what the container calls with its `Configuration` provider, so `from_dict` and
        `from_env` both end up here and there is one set of defaults.
        """
        defaults = cls()
        kwargs: dict[str, Any] = {}
        for field in dataclasses.fields(cls):
            fallback = getattr(defaults, field.name)
            raw = values.get(field.name, fallback)
            if isinstance(fallback, bool):
                kwargs[field.name] = bool(raw)
            elif isinstance(fallback, int):
                kwargs[field.name] = int(raw) if str(raw).strip() else fallback
            else:
                kwargs[field.name] = str(raw)
        # The endpoint's trailing slash is a formatting detail, not configuration.
        kwargs["glm_base_url"] = kwargs["glm_base_url"].rstrip("/")
        kwargs["glm_api_key"] = kwargs["glm_api_key"] or "not-needed"
        kwargs["lit_path"] = kwargs["lit_path"] or (shutil.which("lit") or DEFAULT_LIT)
        return cls(**kwargs)

    def as_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)
