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
    FakeWarmEngine,
    FakeEngine,
    FakePreviewEncoder,
    FakeRenderer,
    FakeWorkspace,
    registry,
)
from ocr_playground.application.warming import (
    ONE_DAY,
    EngineWarmer,
    WarmPolicy,
    WarmStatus,
)
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
    assert Container().warm_policy().min_interval_seconds == 90
    assert Container().warm_policy().daily_cap == 100

    off = build_container({"GLM_OCR_WARM": "0", "GLM_OCR_WARM_DAILY_CAP": "3"})
    assert off.settings().warm_enabled is False
    assert off.settings().warm_daily_cap == 3
    assert off.warmer().request("glm") == WarmStatus.DISABLED


def test_the_warm_interval_fits_inside_the_warm_window() -> None:
    """An engine that may only be woken after it scaled down is cold and un-warmable at once.

    The warm window is `scaledown_window` in each Modal app: the GPU stays up that long after the
    last request. Waiting longer than that to allow the next warm hands the visitor the cold start
    the path exists to remove — which is how 300 s of throttle against a 120 s window behaved. Waiting
    exactly that long is worse than useless: it is the same race, decided by a second.
    """
    windows: list[int] = []
    for app in sorted((ROOT / "deploy" / "modal").glob("serve_*.py")):
        declared = re.search(r"scaledown_window=(\d+)", app.read_text())
        if declared:
            windows.append(int(declared.group(1)))

    assert len(windows) == 6, "the six GPU apps each declare their warm window"
    interval = Container().warm_policy().min_interval_seconds
    assert interval < min(windows), (
        f"a warm every {interval}s against a {min(windows)}s window is a tie, and a tie is decided by "
        "whichever side of the second the request lands on"
    )


def test_the_warm_budget_can_be_read_without_spending_it() -> None:
    """`/api/health` answers with this: how many warms are left and when the next one is allowed."""
    clock = FakeClock()
    jobs: list = []
    warmer = EngineWarmer(
        engines={"glm": FakeWarmEngine(GLM), "fast": FakeEngine(FAST)},
        registry=registry(),
        clock=clock,
        policy=WarmPolicy(min_interval_seconds=90, daily_cap=2),
        submit=jobs.append,
    )

    before = warmer.report()
    assert before["glm"] == {
        "used_today": 0, "daily_cap": 2, "interval_seconds": 90,
        "next_warm_in": 0, "next_warm": "started",  # nothing is in the way yet
    }
    assert jobs == [], "reading the budget must not start a warm"

    assert warmer.request("glm") == WarmStatus.STARTED
    clock.now += 30
    assert warmer.report()["glm"]["used_today"] == 1
    assert warmer.report()["glm"]["next_warm_in"] == 60

    assert warmer.report()["glm"]["next_warm"] == "throttled"
    assert warmer.request("glm") == WarmStatus.THROTTLED, "30 s is inside the 90 s window"
    clock.now += 61
    assert warmer.request("glm") == WarmStatus.STARTED, "past the window, and the cap allows two"

    # The cap is now the binding limit, and the wait is until the oldest start leaves the rolling day —
    # not zero, which is what a report built only from the interval said.
    spent = warmer.report()["glm"]
    assert spent["next_warm"] == "capped"
    assert spent["used_today"] == 2
    assert spent["next_warm_in"] == round(ONE_DAY - 91, 1)
    assert warmer.request("glm") == WarmStatus.CAPPED


def test_a_zero_cap_is_a_refusal_not_a_crash() -> None:
    """`GLM_OCR_WARM_DAILY_CAP=0` switches warming off through the budget; it may not raise."""
    warmer = EngineWarmer(
        engines={"glm": FakeWarmEngine(GLM)},
        registry=registry(),
        clock=FakeClock(),
        policy=WarmPolicy(min_interval_seconds=90, daily_cap=0),
        submit=lambda job: None,
    )

    assert warmer.request("glm") == WarmStatus.CAPPED
    budget = warmer.report()["glm"]
    assert budget["next_warm"] == "capped"
    assert budget["next_warm_in"] is None, "no wait helps when the cap is zero"


def test_the_budget_says_what_a_request_would_say() -> None:
    """The report is a promise about `request()`: off and unavailable are not `started`."""
    off = EngineWarmer(
        engines={"glm": FakeWarmEngine(GLM)},
        registry=registry(),
        clock=FakeClock(),
        policy=WarmPolicy(enabled=False),
        submit=lambda job: None,
    )
    assert off.report()["glm"]["next_warm"] == "disabled"
    assert off.report()["glm"]["next_warm_in"] is None

    cold = EngineWarmer(
        engines={"glm": FakeWarmEngine(GLM, available=False)},
        registry=registry(),
        clock=FakeClock(),
        policy=WarmPolicy(),
        submit=lambda job: None,
    )
    assert cold.report()["glm"]["next_warm"] == "unavailable"
    assert cold.report()["glm"]["next_warm_in"] is None


def test_only_the_remote_engine_can_be_warmed() -> None:
    container = build_container({"GLM_OCR_BASE_URL": "https://gpu.example/v1"})
    assert isinstance(container.engines()["glm"], PreWarmable)
    assert not isinstance(container.engines()["fast"], PreWarmable)


def test_a_host_with_no_endpoint_reports_the_gpu_as_unavailable() -> None:
    assert Container().warmer().request("glm") == WarmStatus.UNAVAILABLE


def test_every_l4_app_can_fall_back_to_a10g() -> None:
    """L4 is what this account prefers and often cannot get; A10G is what it can.

    Modal schedules a list in order of preference, so the app asks for both instead of waiting out a
    drought: the L4 pool was empty for the forty minutes the playground spent showing nothing.
    """
    l4_apps = ["chandra", "glm_ocr", "mineru", "nemotron", "unlimited"]
    misses = {
        name: re.findall(r'gpu=(.*)', (ROOT / "deploy" / "modal" / f"serve_{name}.py").read_text())[0]
        for name in l4_apps
    }
    for name, request in misses.items():
        assert '"L4"' in request and '"A10G"' in request, f"{name} cannot fall back: {request}"
        assert request.index('"L4"') < request.index('"A10G"'), f"{name} prefers A10G over L4"


def test_every_modal_app_is_capped_at_one_container() -> None:
    """One upload asks each engine exactly once, and a client that gives up asks again.

    A second concurrent request otherwise buys a second copy of a model that is already loading on the
    GPU — the playground got two mineru containers that way, both of them booting for uploads the
    browser had already abandoned. The cap makes the retry queue instead.
    """
    apps = sorted((ROOT / "deploy" / "modal").glob("serve_*.py"))
    assert len(apps) == 6, "one file per engine"
    uncapped = [app.name for app in apps if not re.search(r"max_containers\s*=\s*1", app.read_text())]
    assert uncapped == [], f"these apps can be cloned by a retry: {uncapped}"
