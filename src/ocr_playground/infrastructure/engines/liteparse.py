"""LiteParse + Tesseract: the fast, local reader.

A couple of seconds per page on this server's CPU, visibly weaker on handwriting — which is exactly
why the page offers it beside the GPU path instead of instead of it.
"""
from __future__ import annotations

from ...domain.document import PageImage
from ...domain.engine import EngineSpec
from ..process import ProcessRunner
from ..versions import first_line

FAST_SPEC = EngineSpec(
    id="liteparse-tesseract",
    name="LiteParse + Tesseract",
    kind="staged",
    pipeline="raster + bundled Tesseract, parsed live on upload",
    device="native · CPU ×4 (Tesseract)",
    needs="nothing",
)


class LiteParseEngine:
    spec = FAST_SPEC

    def __init__(self, runner: ProcessRunner, lit_path: str) -> None:
        self._runner = runner
        self._lit = lit_path

    @property
    def available(self) -> bool:
        # The tool is installed in the image; if it is missing the parse fails loudly, which is a
        # better signal than silently hiding the only path a deployment without a GPU has.
        return True

    def identity(self) -> str:
        return (
            f"liteparse-tesseract lit={first_line([self._lit, '--version'])} "
            f"tesseract={first_line(['tesseract', '--version'])}"
        )

    def read(self, page: PageImage, deadline: float | None = None) -> str:
        return self._runner.run(
            [self._lit, "parse", str(page.path), "--format", "markdown", "-q"]
        ).strip()
