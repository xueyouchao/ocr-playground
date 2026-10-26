"""The use case: rendering, reading, previews and the document budget.

The engines, renderer and clock are fakes, so what is under test is the orchestration that the API and
the indexer both depend on.
"""
from __future__ import annotations

import pytest

from fakes import build_rig
from ocr_playground.application.errors import EngineFailure, EngineTimeout, EngineUnavailable
from ocr_playground.domain.document import PageLimit, SourceDocument
from ocr_playground.domain.errors import UnknownMode
from ocr_playground.interfaces.evidence import document_evidence


def source(png_bytes: bytes, name: str = "form.png") -> SourceDocument:
    return SourceDocument.accepted(png_bytes, "image/png", name)


def test_reads_every_page_in_order_with_the_engine_attribution(tmp_path, png_bytes) -> None:
    rig = build_rig(tmp_path, pages=2)
    document = rig.service.execute(source(png_bytes), "fast")

    assert document.page_count == 2
    assert [page.number for page in document.pages] == [1, 2]
    assert all(page.reading.engine_id == "liteparse-tesseract" for page in document.pages)
    assert [page.reading.text for page in document.pages] == ["text of page 1", "text of page 2"]
    assert document.engine.id == "liteparse-tesseract"
    assert document.name == "form.png"


def test_the_note_defaults_to_the_engine_name(tmp_path, png_bytes) -> None:
    rig = build_rig(tmp_path)
    assert rig.service.execute(source(png_bytes), "fast").note == "read by LiteParse + Tesseract"
    assert rig.service.execute(source(png_bytes), "fast", note="a note").note == "a note"


def test_the_default_mode_is_glm_and_reads_through_glm(tmp_path, png_bytes) -> None:
    rig = build_rig(tmp_path, pages=1)
    rig.service.execute(source(png_bytes), "glm")
    assert len(rig.glm.reads) == 1
    assert rig.fast.reads == []


def test_one_deadline_is_shared_by_the_whole_document(tmp_path, png_bytes) -> None:
    rig = build_rig(tmp_path, pages=3, budget=123)
    rig.service.execute(source(png_bytes), "glm")

    deadlines = [deadline for _, deadline in rig.glm.reads]
    assert deadlines == [rig.clock.now + 123] * 3


def test_previews_are_produced_from_the_rendered_pages(tmp_path, png_bytes) -> None:
    rig = build_rig(tmp_path, pages=2)
    document = rig.service.execute(source(png_bytes), "fast")

    assert [page.preview for page in document.pages] == ["data:page-1", "data:page-2"]
    assert rig.previews.encoded == [1, 2]


def test_previews_can_be_skipped_for_a_corpus_run(tmp_path, png_bytes) -> None:
    rig = build_rig(tmp_path)
    document = rig.service.execute(source(png_bytes), "fast", previews=False)

    assert all(page.preview is None for page in document.pages)
    assert rig.previews.encoded == []


def test_previews_can_be_forced_per_call(tmp_path, png_bytes) -> None:
    rig = build_rig(tmp_path, previews=False)
    document = rig.service.execute(source(png_bytes), "fast", previews=True)
    assert document.pages[0].preview == "data:page-1"


def test_the_upload_cap_is_the_default_limit(tmp_path, png_bytes) -> None:
    rig = build_rig(tmp_path, max_pages=7)
    rig.service.execute(source(png_bytes), "fast")
    _, _, limit = rig.renderer.calls[0]
    assert limit == PageLimit(7)


def test_an_explicit_limit_overrides_the_upload_cap(tmp_path, png_bytes) -> None:
    rig = build_rig(tmp_path, max_pages=5)
    rig.service.execute(source(png_bytes), "fast", limit=PageLimit(None))
    _, _, limit = rig.renderer.calls[0]
    assert limit == PageLimit(None)


def test_the_renderer_works_inside_the_workspace(tmp_path, png_bytes) -> None:
    rig = build_rig(tmp_path)
    rig.service.execute(source(png_bytes), "fast")
    rendered_source, scratch, _ = rig.renderer.calls[0]
    assert rendered_source.payload == png_bytes
    assert scratch == rig.workspace


def test_an_unknown_mode_is_named_with_the_choices(tmp_path, png_bytes) -> None:
    rig = build_rig(tmp_path)
    with pytest.raises(UnknownMode, match="'nope' is not an extraction mode; use glm or fast"):
        rig.service.execute(source(png_bytes), "nope")


def test_a_mode_without_an_endpoint_offers_the_other_path(tmp_path, png_bytes) -> None:
    rig = build_rig(tmp_path, glm_available=False)
    with pytest.raises(EngineUnavailable) as caught:
        rig.service.execute(source(png_bytes), "glm")
    assert str(caught.value) == (
        "GLM-OCR is not configured on this server; use the fast option (LiteParse + Tesseract)"
    )


def test_an_engine_failure_propagates_untouched(tmp_path, png_bytes) -> None:
    rig = build_rig(tmp_path, glm_error=EngineTimeout("the GPU took too long"))
    with pytest.raises(EngineTimeout, match="the GPU took too long"):
        rig.service.execute(source(png_bytes), "glm")


def test_a_failed_page_stops_the_document(tmp_path, png_bytes) -> None:
    rig = build_rig(tmp_path, pages=3, fast_error=EngineFailure("bad page"))
    with pytest.raises(EngineFailure):
        rig.service.execute(source(png_bytes), "fast")
    assert len(rig.fast.reads) == 1, "a failure is not retried page after page"


def test_the_service_refuses_engines_that_do_not_match_the_registry(tmp_path) -> None:
    from fakes import FakeEngine, FakePreviewEncoder, FakeRenderer, FakeWorkspace, registry
    from ocr_playground.application.read_document import ReadDocumentService

    with pytest.raises(ValueError, match="do not match the registry"):
        ReadDocumentService(
            engines={"glm": FakeEngine(registry().spec("glm"))},
            registry=registry(),
            renderer=FakeRenderer(),
            preview_encoder=FakePreviewEncoder(),
            workspace=FakeWorkspace(tmp_path),
            clock=None,
            budget_seconds=1,
            max_pages=5,
        )


def test_a_comparison_reads_the_same_pages_with_every_engine(tmp_path, png_bytes) -> None:
    rig = build_rig(tmp_path, pages=2)
    document = rig.service.execute(source(png_bytes), "glm,fast")

    assert [engine.id for engine in document.engines] == [
        "glm-ocr", "liteparse-tesseract",
    ]
    assert document.engine.id == "glm-ocr", "the first engine asked for is the primary"
    assert document.note == "read side by side by GLM-OCR + LiteParse + Tesseract"
    for page in document.pages:
        assert {reading.engine_id for reading in page.readings} == {
            "glm-ocr", "liteparse-tesseract",
        }
    assert rig.glm.reads and rig.fast.reads, "both engines read, not just the primary"


def test_the_engines_run_concurrently_rather_than_one_after_another(tmp_path, png_bytes) -> None:
    """Each engine is its own endpoint, so overlapping them costs the same and waits once."""
    import threading

    rig = build_rig(tmp_path, pages=1)
    both_running = threading.Barrier(2, timeout=5)
    entered: list[str] = []

    def rendezvous(name: str, text: str) -> None:
        entered.append(name)
        both_running.wait()  # only releases if the other engine is inside read() too
        return text

    rig.glm._text = lambda page: rendezvous("glm", f"glm page {page}")
    rig.fast._text = lambda page: rendezvous("fast", f"fast page {page}")

    document = rig.service.execute(source(png_bytes), "glm,fast")
    assert sorted(entered) == ["fast", "glm"]
    assert {reading.engine_id for reading in document.pages[0].readings} == {
        "glm-ocr", "liteparse-tesseract",
    }


def test_one_engine_failing_leaves_the_comparison_standing(tmp_path, png_bytes) -> None:
    rig = build_rig(tmp_path, pages=2, fast_error=EngineTimeout("the endpoint did not answer"))
    document = rig.service.execute(source(png_bytes), "glm,fast")

    assert [engine.id for engine in document.engines] == ["glm-ocr", "liteparse-tesseract"]
    for page in document.pages:
        assert page.reading_by("glm-ocr") is not None
        assert page.reading_by("liteparse-tesseract") is None
        assert page.failure_for("liteparse-tesseract") == "the endpoint did not answer"

    evidence = document_evidence(document)
    cells = evidence["pages"][0]["cells"]
    assert cells["glm-ocr"]["chars"] > 0
    assert cells["liteparse-tesseract"]["chars"] is None
    assert cells["liteparse-tesseract"]["error"] == "the endpoint did not answer"
    assert [system["id"] for system in evidence["systems"]] == ["glm-ocr", "liteparse-tesseract"]


def test_a_document_that_no_engine_could_read_is_an_error(tmp_path, png_bytes) -> None:
    rig = build_rig(
        tmp_path, glm_error=EngineTimeout("glm timed out"), fast_error=EngineFailure("fast broke"),
    )
    with pytest.raises(EngineTimeout):
        rig.service.execute(source(png_bytes), "glm,fast")


def test_a_failed_engine_is_not_retried_on_every_page(tmp_path, png_bytes) -> None:
    rig = build_rig(tmp_path, pages=3, fast_error=EngineFailure("fast broke"))
    rig.service.execute(source(png_bytes), "glm,fast")
    assert len(rig.fast.reads) == 1, "one failure, not one per page"


def test_a_comparison_resolves_every_engine_before_reading(tmp_path, png_bytes) -> None:
    rig = build_rig(tmp_path, fast_available=False)
    with pytest.raises(EngineUnavailable):
        rig.service.execute(source(png_bytes), "glm,fast")
    assert not rig.glm.reads, "nothing is read when one engine in the list cannot run"


def test_a_mode_list_is_deduplicated_and_order_preserved(tmp_path, png_bytes) -> None:
    rig = build_rig(tmp_path, pages=1)
    document = rig.service.execute(source(png_bytes), " fast , glm , fast ")
    assert [engine.id for engine in document.engines] == ["liteparse-tesseract", "glm-ocr"]
