"""The indexer's own configuration, which is deliberately not the API's.

An index run and an upload answer different questions. The API caps at five pages because it is a
public request; a corpus run reads every page unless told otherwise, because silently publishing a
document with its last pages missing would be worse than a slow run.
"""
from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from ...domain.engine import EngineRegistry
from ...domain.ports import PageOcr
from ..site.layout import data

SUFFIX_TYPES = {
    ".pdf": "application/pdf",
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".tif": "image/tiff",
    ".tiff": "image/tiff",
    ".webp": "image/webp",
}


class ConfigError(RuntimeError):
    """The environment names a run this checkout cannot perform."""


@dataclass(frozen=True, slots=True)
class IndexerConfig:
    corpus: Path
    out: Path
    db: Path
    mode: str
    max_pages: int

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> IndexerConfig:
        values = os.environ if env is None else env
        chandra_configured = bool(values.get("CHANDRA_OCR_BASE_URL", "").strip())
        glm_configured = bool(values.get("GLM_OCR_BASE_URL", "").strip())
        return cls(
            # Relative to the checkout, like the rest of the layout: the bundled scan moved to
            # data/corpus/, and an unset OCR_CORPUS should still point at it.
            corpus=Path(values.get("OCR_CORPUS", str(data(Path.cwd()) / "corpus"))).resolve(),
            out=Path(values.get("OCR_OUT", "out")),
            db=Path(values.get("OCR_DB", "cocoindex.db")),
            # The uploader defaults to Chandra; the indexer follows the same preference among
            # configured endpoints, and stays on the free CPU path when none is named. Naming a mode
            # whose endpoint is absent is an error, not a silent downgrade.
            mode=values.get("OCR_MODE", "").strip().lower()
            or ("chandra" if chandra_configured else "glm" if glm_configured else "fast"),
            # 0 means every page.
            max_pages=int(values.get("OCR_MAX_PAGES", "0")),
        )

    def validate(self, registry: EngineRegistry, engines: Mapping[str, PageOcr]) -> None:
        """Fail before touching the corpus, with a message that names the environment variable."""
        if self.mode not in registry:
            raise ConfigError(
                f"OCR_MODE={self.mode!r} is not an extraction mode; use {' or '.join(registry.modes)}"
            )
        if not engines[self.mode].available:
            raise ConfigError(
                f"OCR_MODE={self.mode} needs {registry.spec(self.mode).needs}; "
                "leave the mode unset to use the CPU path"
            )
        if not self.corpus.is_dir():
            raise ConfigError(f"OCR_CORPUS={str(self.corpus)!r} is not a directory")
        if self.max_pages < 0:
            raise ConfigError("OCR_MAX_PAGES must be 0 (every page) or a positive number")
