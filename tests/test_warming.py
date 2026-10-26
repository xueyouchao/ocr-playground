"""The pre-warm policy: it buys a cold start back without renting a GPU all day.

Every rule here is about money as much as latency. A warm container bills GPU time from boot until
`scaledown_window` after the last request, so the tests pin three things: one boot at a time, one per
interval, and a hard ceiling per day.
"""
from __future__ import annotations

import pytest

from fakes import FAST, GLM, FakeClock, FakeEngine, FakeWarmEngine, registry
from ocr_playground.application.warming import EngineWarmer, WarmPolicy, WarmStatus
from ocr_playground.domain.errors import UnknownMode


class Harness:
    """A warmer whose jobs run only when the test asks, so start and finish stay separable."""

    def __init__(self, engine, **policy: object) -> None:
        self.engine = engine
        self.clock = FakeClock()
        self.jobs: list = []
        self.warmer = EngineWarmer(
            engines={"glm": engine, "fast": FakeEngine(FAST)},
            registry=registry(),
            clock=self.clock,
            policy=WarmPolicy(**policy),  # type: ignore[arg-type]
            submit=self.jobs.append,
        )

    def run(self) -> None:
        while self.jobs:
            self.jobs.pop(0)()


def test_an_unknown_mode_is_the_client_s_error_before_anything_boots():
    harness = Harness(FakeWarmEngine(GLM))
    with pytest.raises(UnknownMode):
        harness.warmer.request("magic")
    assert harness.jobs == []


def test_a_local_engine_has_nothing_to_warm():
    harness = Harness(FakeWarmEngine(GLM))
    assert harness.warmer.request("fast") == WarmStatus.NOT_WARMABLE
    assert harness.jobs == []


def test_an_engine_with_no_endpoint_is_not_warmed():
    harness = Harness(FakeWarmEngine(GLM, available=False))
    assert harness.warmer.request("glm") == WarmStatus.UNAVAILABLE
    assert harness.jobs == []


def test_warming_can_be_turned_off_for_a_host_that_pays_for_its_own_gpu():
    harness = Harness(FakeWarmEngine(GLM), enabled=False)
    assert harness.warmer.request("glm") == WarmStatus.DISABLED
    assert harness.jobs == []


def test_the_first_request_starts_exactly_one_boot():
    engine = FakeWarmEngine(GLM)
    harness = Harness(engine)
    assert harness.warmer.request("glm") == WarmStatus.STARTED
    assert len(harness.jobs) == 1
    assert engine.warm_calls == []  # the decision does not block on the endpoint
    harness.run()
    assert engine.warm_calls == [120.0]


def test_a_second_request_inside_the_interval_is_throttled():
    harness = Harness(FakeWarmEngine(GLM), min_interval_seconds=300)
    harness.warmer.request("glm")
    harness.clock.now += 299
    assert harness.warmer.request("glm") == WarmStatus.THROTTLED
    assert len(harness.jobs) == 1


def test_the_interval_lets_a_later_visitor_warm_again():
    harness = Harness(FakeWarmEngine(GLM), min_interval_seconds=300)
    harness.warmer.request("glm")
    harness.clock.now += 300
    assert harness.warmer.request("glm") == WarmStatus.STARTED
    assert len(harness.jobs) == 2


def test_the_day_has_a_hard_ceiling():
    harness = Harness(FakeWarmEngine(GLM), min_interval_seconds=0, daily_cap=2)
    assert [harness.warmer.request("glm") for _ in range(3)] == [
        WarmStatus.STARTED,
        WarmStatus.STARTED,
        WarmStatus.CAPPED,
    ]
    assert len(harness.jobs) == 2


def test_the_ceiling_counts_any_24_hours_not_a_calendar_day():
    """A counter that resets at a fixed boundary allows twice the ceiling across it."""
    harness = Harness(FakeWarmEngine(GLM), min_interval_seconds=0, daily_cap=2)
    assert harness.warmer.request("glm") == WarmStatus.STARTED  # t0
    harness.clock.now += 86_000
    assert harness.warmer.request("glm") == WarmStatus.STARTED  # t0 + 23h20m, still inside the day
    assert harness.warmer.request("glm") == WarmStatus.CAPPED

    harness.clock.now += 401  # t0 has aged out; t1 has not
    assert harness.warmer.request("glm") == WarmStatus.STARTED
    assert harness.warmer.request("glm") == WarmStatus.CAPPED


def test_an_idle_day_clears_the_ceiling_completely():
    harness = Harness(FakeWarmEngine(GLM), min_interval_seconds=0, daily_cap=1)
    harness.warmer.request("glm")
    assert harness.warmer.request("glm") == WarmStatus.CAPPED
    harness.clock.now += 86_400
    assert harness.warmer.request("glm") == WarmStatus.STARTED


def test_the_policy_s_timeout_reaches_the_engine():
    engine = FakeWarmEngine(GLM)
    harness = Harness(engine, timeout_seconds=5)
    harness.warmer.request("glm")
    harness.run()
    assert engine.warm_calls == [5.0]


def test_a_warm_that_fails_never_raises(capsys):
    engine = FakeWarmEngine(GLM, answers=False)
    harness = Harness(engine)
    harness.warmer.request("glm")
    harness.run()
    assert "did not answer in time" in capsys.readouterr().err

    class Broken(FakeWarmEngine):
        def warm(self, timeout: float = 120) -> bool:
            raise OSError("connection refused")

    broken = Harness(Broken(GLM))
    broken.warmer.request("glm")
    broken.run()  # a dead endpoint must not become a failed upload
    assert "connection refused" in capsys.readouterr().err
