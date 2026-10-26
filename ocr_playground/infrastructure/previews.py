"""A page thumbnail as a data URL, so the browser needs no second request for it."""
from __future__ import annotations

import base64
from pathlib import Path

from ..domain.document import PageImage
from ..domain.ports import PreviewEncoder
from .process import ProcessRunner


class ImageMagickPreviewEncoder(PreviewEncoder):
    def __init__(self, runner: ProcessRunner, max_px: int = 1400, quality: int = 78) -> None:
        self._runner = runner
        self._max_px = max_px
        self._quality = quality

    def encode(self, page: PageImage) -> str:
        target = page.path.with_suffix(".preview.jpg")
        self._runner.run([
            "convert", str(page.path),
            "-resize", f"{self._max_px}x{self._max_px}",
            "-quality", str(self._quality),
            str(target),
        ])
        return "data:image/jpeg;base64," + base64.b64encode(Path(target).read_bytes()).decode()
