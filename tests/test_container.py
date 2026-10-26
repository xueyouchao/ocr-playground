"""The container: settings from configuration, one instance per node, and overrides that work.

The point of the container is that an entry point asks for a service, not that the service imports its
collaborators. These tests hold it to that.
"""
from __future__ import annotations

from fakes import (
    FAST,
    GLM,
    FakeClock,
    FakeEngine,
    FakePreviewEncoder,
    FakeRenderer,
    FakeWorkspace,
    registry,
)
from ocr_playground.composition.container import Container, build_container
from ocr_playground.domain.document import PageLimit, SourceDocument
from ocr_playground.infrastructure.settings import Settings


def test_defaults_resolve_without_bootstrapping() -> None:
    container = Container()
    settings = container.settings()

    assert settings.default_mode == "glm"
    assert settings.max_pages == 5
    assert settings.max_upload_bytes == 20 * 1024 * 1024
    assert settings.glm_base_url == ""


def test_configuration_overrides_settings() -> None:
    container = Container()
    container.config.from_dict({
        "glm_base_url": "https://gpu.example/v1/",
        "glm_model": "glm-ocr-v2",
        "max_pages": 3,
        "max_upload_bytes": 1024,
        "default_mode": "fast",
        "parse_slots": 4,
        "parse_timeout_seconds": 9,
    })
    settings = container.settings()

    assert settings.glm_base_url == "https://gpu.example/v1", "trailing slash is formatting"
    assert settings.glm_model == "glm-ocr-v2"
    assert settings.max_pages == 3
    assert settings.max_upload_bytes == 1024
    assert settings.default_mode == "fast"
    assert settings.parse_slots == 4
    assert container.process().timeout == 9


def test_environment_seeds_the_container() -> None:
    container = build_container({
        "GLM_OCR_BASE_URL": "https://gpu.example/v1/",
        "GLM_OCR_MODEL": "ocr-large",
        "GLM_OCR_PROMPT": "Read this.",
        "GLM_OCR_BUDGET_SECONDS": "42",
        "LIT_CLI": "/opt/lit",
    })
    settings = container.settings()

    assert settings.glm_base_url == "https://gpu.example/v1"
    assert settings.glm_model == "ocr-large"
    assert settings.glm_prompt == "Read this."
    assert settings.glm_budget_seconds == 42
    assert settings.lit_path == "/opt/lit"
    assert container.glm().available is True


def test_the_registry_is_derived_from_the_wired_engines() -> None:
    container = build_container({"GLM_OCR_BASE_URL": "https://gpu.example/v1"})
    registry_ = container.registry()

    assert registry_.modes == ("glm", "fast")
    assert registry_.spec("glm").id == "glm-ocr"
    assert registry_.spec("glm").name == "GLM-OCR"
    assert registry_.default_mode == "glm"


def test_singletons_are_shared_and_a_use_case_is_one_instance() -> None:
    container = build_container()
    assert container.read_document() is container.read_document()
    assert container.engines()["glm"] is container.glm()
    assert container.engines()["fast"] is container.fast()


def test_a_node_can_be_replaced_without_touching_the_service() -> None:
    container = Container()
    container.glm.override(FakeEngine(GLM, available=False))
    assert container.glm().available is False
    container.glm.reset_override()
    assert container.glm().available is False, "no endpoint in the default configuration"


def test_the_use_case_takes_its_budget_and_page_cap_from_configuration(tmp_path) -> None:
    container = Container()
    container.config.from_dict({"glm_budget_seconds": 42, "max_pages": 3})
    clock = FakeClock(now=100)
    renderer = FakeRenderer(pages=1)
    glm = FakeEngine(GLM)

    container.clock.override(clock)
    container.renderer.override(renderer)
    container.preview_encoder.override(FakePreviewEncoder())
    container.workspace.override(FakeWorkspace(tmp_path))
    container.engines.override({"glm": glm, "fast": FakeEngine(FAST)})
    container.registry.override(registry("glm"))

    document = container.read_document().execute(
        SourceDocument.accepted(b"x", "image/png", "form.png"), "glm"
    )

    assert document.pages[0].reading.text == "text of page 1"
    assert renderer.calls[0][2] == PageLimit(3), "the configured upload cap reached the renderer"
    assert glm.reads == [(1, 142)], "the configured budget reached the engine"


def test_settings_default_to_the_deployment_values() -> None:
    assert Settings().max_pages == 5
    assert Settings().preview_max_px == 1400
    assert Settings.from_mapping({}).default_mode == "glm"
