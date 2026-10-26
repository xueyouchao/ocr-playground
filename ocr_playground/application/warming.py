"""Waking a scale-to-zero engine before the upload that needs it.

GLM-OCR lives on a Modal L4 that scales to zero: the first request after idle waits ~60 s for the
container to boot and vLLM to come up. That wait cannot be removed, but it does not have to happen
*after* somebody picks a file — this module starts it when they look like they are about to.

It is also where the money is. Modal bills GPU time from boot until `scaledown_window` after the last
request, so an unthrottled warm on every page view turns a trickle of traffic into a rented GPU. The
policy bounds exactly that: one warm per interval, and a capped number per rolling day.
"""
from __future__ import annotations

import sys
import threading
from collections import deque
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from enum import StrEnum

from ..domain.engine import EngineRegistry
from ..domain.ports import Clock, PageOcr, PreWarmable

ONE_DAY = 86_400.0


class WarmStatus(StrEnum):
    """Why a warm did or did not start. Serialised into the HTTP reply, so the page can explain."""

    STARTED = "started"
    THROTTLED = "throttled"
    CAPPED = "capped"
    UNAVAILABLE = "unavailable"
    NOT_WARMABLE = "not_warmable"
    DISABLED = "disabled"


@dataclass(frozen=True, slots=True)
class WarmPolicy:
    enabled: bool = True
    min_interval_seconds: float = 300
    daily_cap: int = 40
    timeout_seconds: float = 120


def _in_background(job: Callable[[], None]) -> None:
    threading.Thread(target=job, name="warm-engine", daemon=True).start()


class EngineWarmer:
    """Starts a lazy engine booting: once per interval, at most `daily_cap` times in any 24 hours.

    The cap keeps the starts themselves, not a counter, so a start at midnight and a start a second
    before it are both still counted later in the day; a counter that resets at a fixed boundary
    would allow twice the ceiling across that boundary.

    State is per process and survives for its lifetime; a restart forgets the starts, which is the
    right failure direction for a demo (it can only warm once more than intended, never fewer).
    """

    def __init__(
        self,
        *,
        engines: Mapping[str, PageOcr],
        registry: EngineRegistry,
        clock: Clock,
        policy: WarmPolicy,
        submit: Callable[[Callable[[], None]], None] = _in_background,
    ) -> None:
        self._engines = engines
        self._registry = registry
        self._clock = clock
        self._policy = policy
        self._submit = submit
        self._last_started: float | None = None
        self._starts: deque[float] = deque()

    def request(self, mode: str) -> WarmStatus:
        """Decide whether to warm `mode` now, and start it. Never blocks on the endpoint."""
        self._registry.spec(mode)  # an unknown mode is the client's error, before any side effect
        if not self._policy.enabled:
            return WarmStatus.DISABLED
        engine = self._engines[mode]
        if not engine.available:
            return WarmStatus.UNAVAILABLE
        if not isinstance(engine, PreWarmable):
            return WarmStatus.NOT_WARMABLE

        now = self._clock.monotonic()
        while self._starts and now - self._starts[0] >= ONE_DAY:
            self._starts.popleft()
        if len(self._starts) >= self._policy.daily_cap:
            return WarmStatus.CAPPED
        if self._last_started is not None and now - self._last_started < self._policy.min_interval_seconds:
            return WarmStatus.THROTTLED

        self._last_started = now
        self._starts.append(now)
        self._submit(lambda: self._warm(engine))
        return WarmStatus.STARTED

    def _warm(self, engine: PreWarmable) -> None:
        try:
            warmed = engine.warm(self._policy.timeout_seconds)
        except Exception as error:  # noqa: BLE001 - a failed warm must not become a failed upload
            sys.stderr.write(f"warming {engine.spec.id} failed: {error}\n")
            return
        if not warmed:
            sys.stderr.write(f"warming {engine.spec.id} did not answer in time\n")
