"""poppler and ImageMagick: bytes to page images.

The rules encoded here were learned the awkward way:

- A PNG or JPEG upload is parsed as it arrived. Round-tripping it through ImageMagick rewrites it as
  an 8-bit grayscale palette image and OCR then reads half the page (111 chars instead of 407 on the
  sample form). Only the formats LiteParse cannot open take the conversion path.
- The page cap is checked from `pdfinfo` before a single page is rendered, so a 200-page PDF is
  refused in milliseconds rather than minutes.
- Pages are sorted numerically: `page-10` follows `page-9`, not `page-1`.
"""
from __future__ import annotations

import re
from pathlib import Path

from ..domain.document import PageImage, PageLimit, SourceDocument
from ..domain.errors import UnreadablePageCount
from ..domain.ports import PageRenderer
from .process import ProcessRunner

# Formats LiteParse opens itself, and therefore the ones we must not re-encode.
_PASSTHROUGH = frozenset({"image/png", "image/jpeg"})
_PAGE_NUMBER = re.compile(r"page-(\d+)\.png$")


def _page_order(path: Path) -> int:
    match = _PAGE_NUMBER.search(path.name)
    return int(match.group(1)) if match else 0


class PopplerPageRenderer(PageRenderer):
    def __init__(self, runner: ProcessRunner, dpi: int = 200) -> None:
        self._runner = runner
        self._dpi = dpi

    def render(self, source: SourceDocument, scratch: Path, limit: PageLimit) -> list[PageImage]:
        uploaded = scratch / f"upload{source.suffix}"
        uploaded.write_bytes(source.payload)

        if source.media_type.is_image:
            if source.media_type.value in _PASSTHROUGH:
                return [PageImage(number=1, path=uploaded)]
            single = scratch / "page-1.png"
            self._runner.run(["convert", str(uploaded), "-colorspace", "sRGB", str(single)])
            return [PageImage(number=1, path=single)]

        info = self._runner.run(["pdfinfo", str(uploaded)])
        match = re.search(r"^Pages:\s+(\d+)", info, re.MULTILINE)
        if not match:
            raise UnreadablePageCount()
        limit.check(int(match.group(1)))

        self._runner.run([
            "pdftoppm", "-png", "-r", str(self._dpi), str(uploaded), str(scratch / "page"),
        ])
        rendered = sorted(scratch.glob("page-*.png"), key=_page_order)
        return [PageImage(number=number, path=path) for number, path in enumerate(rendered, 1)]
