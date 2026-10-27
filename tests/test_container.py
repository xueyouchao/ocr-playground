"""The container: settings from configuration, one instance per node, and overrides that work.

The point of the container is that an entry point asks for a service, not that the service imports its
collaborators. These tests hold it to that.
"""
from __future__ import annotations

import re
from pathlib import Path

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
from ocr_playground.application.warming import WarmStatus
from ocr_playground.composition.container import Container, build_container
from ocr_playground.domain.ports import PreWarmable
from ocr_playground.domain.document import PageLimit, SourceDocument
from ocr_playground.infrastructure.settings import Settings

ROOT = Path(__file__).resolve().parents[1]


def test_defaults_resolve_without_bootstrapping() -> None:
    container = Container()
    settings = container.settings()

    assert settings.default_mode == "chandra"
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

    assert registry_.modes == (
        "glm", "fast", "chandra", "nemotron", "unlimited", "mineru", "paddle",
    )
    assert registry_.spec("glm").id == "glm-ocr"
    assert registry_.spec("glm").name == "GLM-OCR"
    assert registry_.default_mode == "chandra"


def test_each_vlm_endpoint_is_its_own_mode() -> None:
    container = build_container({
        "GLM_OCR_BASE_URL": "https://glm.example/v1",
        "CHANDRA_OCR_BASE_URL": "https://chandra.example/v1",
        "NEMOTRON_PARSE_BASE_URL": "https://nemotron.example/v1",
        "UNLIMITED_OCR_BASE_URL": "https://unlimited.example/v1",
        "MINERU_BASE_URL": "https://mineru.example/v1",
        "PADDLE_OCR_BASE_URL": "https://paddle.example/v1",
    })
    engines = container.engines()

    assert {mode: engine.available for mode, engine in engines.items()} == {
        "glm": True, "fast": True, "chandra": True, "nemotron": True, "unlimited": True,
        "mineru": True, "paddle": True,
    }
    assert engines["chandra"].spec.id == "chandra-2"
    assert engines["nemotron"].spec.id == "nemotron-parse"
    assert engines["unlimited"].spec.id == "unlimited-ocr"
    assert engines["mineru"].spec.id == "mineru-gpu"
    assert engines["paddle"].spec.id == "paddleocr-gpu"
    # The fast path never needs an endpoint, and the GPU modes never run without one.
    assert build_container({}).engines()["chandra"].available is False


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
    assert Settings.from_mapping({}).default_mode == "chandra"


# --- pre-warming the GPU ---------------------------------------------------------------


def test_the_warm_policy_comes_from_configuration() -> None:
    container = Container()
    container.config.from_dict({
        "warm_enabled": True,
        "warm_min_interval_seconds": 60,
        "warm_daily_cap": 7,
        "warm_timeout_seconds": 30,
    })
    policy = container.warm_policy()

    assert (policy.enabled, policy.min_interval_seconds, policy.daily_cap, policy.timeout_seconds) == (
        True,
        60,
        7,
        30,
    )


def test_warming_is_throttled_by_default_and_can_be_switched_off() -> None:
    assert Container().warm_policy().min_interval_seconds == 120
    assert Container().warm_policy().daily_cap == 40

    off = build_container({"GLM_OCR_WARM": "0", "GLM_OCR_WARM_DAILY_CAP": "3"})
    assert off.settings().warm_enabled is False
    assert off.settings().warm_daily_cap == 3
    assert off.warmer().request("glm") == WarmStatus.DISABLED


def test_the_warm_interval_fits_inside_the_warm_window() -> None:
    """An engine that may only be woken after it scaled down is cold and un-warmable at once.

    The warm window is `scaledown_window` in each Modal app: the GPU stays up that long after the
    last request. Waiting longer than that to allow the next warm hands the visitor the cold start
    the path exists to remove — which is how 300 s of throttle against a 120 s window behaved.
    """
    windows: list[int] = []
    for app in sorted((ROOT / "deploy" / "modal").glob("serve_*.py")):
        declared = re.search(r"scaledown_window=(\d+)", app.read_text())
        if declared:
            windows.append(int(declared.group(1)))

    assert len(windows) == 6, "the six GPU apps each declare their warm window"
    assert Container().warm_policy().min_interval_seconds <= min(windows)


def test_only_the_remote_engine_can_be_warmed() -> None:
    container = build_container({"GLM_OCR_BASE_URL": "https://gpu.example/v1"})
    assert isinstance(container.engines()["glm"], PreWarmable)
    assert not isinstance(container.engines()["fast"], PreWarmable)


def test_a_host_with_no_endpoint_reports_the_gpu_as_unavailable() -> None:
    assert Container().warmer().request("glm") == WarmStatus.UNAVAILABLE
