"""The poppler/ImageMagick adapters: the rules that were learned the hard way.

The runner is fake, so the assertions are about *what commands are issued* — the page cap being
checked before rendering, a PNG never being re-encoded, pages sorted numerically.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from ocr_playground.domain.document import PageLimit, SourceDocument
from ocr_playground.domain.errors import PageLimitExceeded, UnreadablePageCount
from ocr_playground.infrastructure.previews import ImageMagickPreviewEncoder
from ocr_playground.infrastructure.rendering import PopplerPageRenderer


class FakeRunner:
    def __init__(self, *, pdfinfo: str = "Pages:          1\n", created: int = 0,
                 convert_bytes: bytes = b"jpegbytes") -> None:
        self.pdfinfo = pdfinfo
        self.created = created
        self.convert_bytes = convert_bytes
        self.commands: list[list[str]] = []

    def run(self, command, timeout=None) -> str:
        self.commands.append(list(command))
        if command[0] == "pdfinfo":
            return self.pdfinfo
        if command[0] == "pdftoppm":
            target = Path(command[-1]).parent
            for number in range(1, self.created + 1):
                (target / f"page-{number}.png").write_bytes(b"png")
            return ""
        if command[0] == "convert":
            Path(command[-1]).write_bytes(self.convert_bytes)
            return ""
        raise AssertionError(f"unexpected command {command}")


def source(media_type: str, name: str, payload: bytes = b"bytes") -> SourceDocument:
    return SourceDocument.accepted(payload, media_type, name)


def renderer(runner: FakeRunner, dpi: int = 200) -> PopplerPageRenderer:
    return PopplerPageRenderer(runner=runner, dpi=dpi)


def test_a_long_pdf_is_refused_before_a_page_is_rendered(tmp_path) -> None:
    runner = FakeRunner(pdfinfo="Title: x\nPages:          9\n")
    with pytest.raises(PageLimitExceeded, match=r"that PDF has 9 pages; this demo reads at most 5"):
        renderer(runner).render(source("application/pdf", "x.pdf"), tmp_path, PageLimit(5))
    assert [command[0] for command in runner.commands] == ["pdfinfo"]


def test_the_cap_can_be_lifted_for_an_index_run(tmp_path) -> None:
    runner = FakeRunner(pdfinfo="Pages:          9\n", created=9)
    pages = renderer(runner).render(source("application/pdf", "x.pdf"), tmp_path, PageLimit(None))

    assert [page.number for page in pages] == list(range(1, 10))
    assert [command[0] for command in runner.commands] == ["pdfinfo", "pdftoppm"]


def test_pages_are_sorted_numerically_not_lexically(tmp_path) -> None:
    runner = FakeRunner(pdfinfo="Pages:          10\n", created=0)
    for number in (1, 10, 2):
        (tmp_path / f"page-{number}.png").write_bytes(b"png")

    pages = renderer(runner).render(source("application/pdf", "x.pdf"), tmp_path, PageLimit(None))
    assert [page.path.stem for page in pages] == ["page-1", "page-2", "page-10"]
    assert [page.number for page in pages] == [1, 2, 3]


def test_an_unreadable_page_count_names_the_problem(tmp_path) -> None:
    runner = FakeRunner(pdfinfo="Title: x\n")
    with pytest.raises(UnreadablePageCount, match="could not read the page count from that PDF"):
        renderer(runner).render(source("application/pdf", "x.pdf"), tmp_path, PageLimit(5))


def test_the_render_dpi_is_the_command(tmp_path) -> None:
    runner = FakeRunner(pdfinfo="Pages: 1\n", created=1)
    renderer(runner, dpi=300).render(source("application/pdf", "x.pdf"), tmp_path, PageLimit(1))
    assert runner.commands[1][:4] == ["pdftoppm", "-png", "-r", "300"]


@pytest.mark.parametrize("media_type", ["image/png", "image/jpeg"])
def test_png_and_jpeg_are_parsed_as_they_arrived(tmp_path, media_type) -> None:
    runner = FakeRunner()
    payload = b"\xff\xd8original" if media_type == "image/jpeg" else b"original"

    pages = renderer(runner).render(source(media_type, "upload.img", payload), tmp_path, PageLimit(5))

    assert runner.commands == [], "a PNG/JPEG must not be round-tripped through ImageMagick"
    assert len(pages) == 1 and pages[0].number == 1
    assert pages[0].path.read_bytes() == payload, "the bytes the engine reads must be the upload"


def test_tiff_is_converted_to_a_png_page(tmp_path) -> None:
    runner = FakeRunner()
    pages = renderer(runner).render(source("image/tiff", "upload.tiff"), tmp_path, PageLimit(5))

    assert len(pages) == 1
    assert pages[0].path.name == "page-1.png"
    assert runner.commands[0][:4] == ["convert", str(tmp_path / "upload.tiff"), "-colorspace", "sRGB"]


def test_preview_is_a_resized_jpeg_data_url(tmp_path) -> None:
    runner = FakeRunner(convert_bytes=b"jpegbytes")
    page_path = tmp_path / "page-1.png"
    page_path.write_bytes(b"png")

    from ocr_playground.domain.document import PageImage

    url = ImageMagickPreviewEncoder(runner=runner, max_px=1400, quality=78).encode(
        PageImage(number=1, path=page_path)
    )

    import base64

    assert url == "data:image/jpeg;base64," + base64.b64encode(b"jpegbytes").decode()
    assert runner.commands[0] == [
        "convert", str(page_path), "-resize", "1400x1400", "-quality", "78",
        str(tmp_path / "page-1.preview.jpg"),
    ]
