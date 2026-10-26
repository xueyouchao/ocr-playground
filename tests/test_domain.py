"""The domain vocabulary, on its own: no adapter, no socket, no subprocess."""
from __future__ import annotations

import pytest

from ocr_playground.domain.document import PageLimit, SourceDocument
from ocr_playground.domain.engine import EngineRegistry, EngineSpec
from ocr_playground.domain.errors import (
    EmptyDocument,
    PageLimitExceeded,
    UnknownMode,
    UnsupportedMediaType,
)
from ocr_playground.domain.media import MediaType
from ocr_playground.domain.reading import ReadingStats


@pytest.mark.parametrize(
    ("header", "value"),
    [
        ("image/png", "image/png"),
        ("Image/PNG; charset=binary", "image/png"),
        ("  application/pdf  ", "application/pdf"),
        (None, ""),
    ],
)
def test_media_type_is_normalised(header, value) -> None:
    assert MediaType.parse(header).value == value


def test_media_type_knows_the_two_families() -> None:
    assert MediaType.parse("application/x-pdf").is_pdf
    assert MediaType.parse("image/tiff").is_image
    assert not MediaType.parse("text/plain").is_supported
    assert not MediaType.parse("").is_supported


def test_suffix_prefers_the_uploaded_name_then_the_format() -> None:
    assert MediaType.parse("image/png").suffix("scan.tiff") == ".tiff"
    assert MediaType.parse("application/pdf").suffix("upload") == ".pdf"
    assert MediaType.parse("image/webp").suffix("upload") == ".png"


def test_an_unsupported_type_names_itself(png_bytes) -> None:
    with pytest.raises(UnsupportedMediaType, match="text/plain is not a PDF or an image"):
        SourceDocument.accepted(b"x", "text/plain", "notes.txt")


def test_an_unknown_type_reads_as_unknown(png_bytes) -> None:
    with pytest.raises(UnsupportedMediaType, match="unknown type is not a PDF or an image"):
        SourceDocument.accepted(b"x", "", "form.png")


def test_an_empty_upload_is_refused() -> None:
    with pytest.raises(EmptyDocument, match="that upload is empty"):
        SourceDocument.accepted(b"", "image/png", "form.png")


def test_a_source_document_keeps_its_name_but_displays_a_basename(png_bytes) -> None:
    source = SourceDocument.accepted(png_bytes, "image/png", "a/b/form.png")
    assert source.name == "a/b/form.png"
    assert source.display_name == "form.png"
    assert SourceDocument.accepted(png_bytes, "image/png", "").display_name == "upload"


def test_page_limit_allows_its_own_maximum() -> None:
    PageLimit(5).check(5)
    with pytest.raises(PageLimitExceeded, match=r"that PDF has 6 pages; this demo reads at most 5"):
        PageLimit(5).check(6)


def test_no_limit_allows_any_length() -> None:
    PageLimit(None).check(500)


def test_a_nonsense_page_limit_fails_at_construction() -> None:
    with pytest.raises(ValueError, match="at least 1"):
        PageLimit(0)


def test_registry_names_modes_in_order_and_rejects_others() -> None:
    registry = EngineRegistry({"glm": _spec("glm"), "fast": _spec("fast")}, "glm")

    assert registry.modes == ("glm", "fast")
    assert registry.default_mode == "glm"
    assert registry.spec("fast").id == "fast"
    assert "fast" in registry
    assert registry.alternative_to("glm") == "fast"

    with pytest.raises(UnknownMode, match="'nope' is not an extraction mode; use glm or fast"):
        registry.spec("nope")


def test_registry_refuses_a_default_that_does_not_exist() -> None:
    with pytest.raises(ValueError, match="default mode 'nope' is not in"):
        EngineRegistry({"glm": _spec("glm")}, "nope")


def test_engine_spec_renders_the_six_evidence_fields() -> None:
    rendered = _spec("glm").as_dict()
    assert list(rendered) == ["id", "name", "kind", "pipeline", "device", "needs"]


def test_counts_are_characters_tokens_and_numeric_share() -> None:
    text = "abc 12 34.5 1/2"
    stats = ReadingStats.from_text(text)
    assert stats == ReadingStats(chars=len(text), tokens=4, numeric_share=0.75)
    assert stats.as_dict() == {"chars": 15, "tokens": 4, "numeric_share": 0.75}


def test_tags_are_replaced_by_a_space_not_dropped() -> None:
    assert ReadingStats.from_text("<b>x</b>y").tokens == 2


def test_empty_text_has_no_numeric_share() -> None:
    assert ReadingStats.from_text("") == ReadingStats(chars=0, tokens=0, numeric_share=0.0)


def test_numeric_share_is_rounded_to_three_places() -> None:
    assert ReadingStats.from_text("abc 12 34.5").numeric_share == 0.667


def _spec(mode: str) -> EngineSpec:
    return EngineSpec(
        id=mode, name=mode.upper(), kind="vlm", pipeline="p", device="CPU", needs="nothing",
    )
