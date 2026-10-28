"""Waking a scale-to-zero engine before the upload that needs it.

GLM-OCR lives on a Modal L4 that scales to zero: the first request after idle waits ~60 s for the
container to boot and vLLM to come up. That wait cannot be removed, but it does not have to happen
*after* somebody picks a file — this module starts it when they look like they are about to.

It is also where the money is. Modal bills GPU time from boot until `scaledown_window` after the last
request, so an unthrottled warm on every page view turns a trickle of traffic into a rented GPU. The
policy bounds exactly that: one warm per engine per interval, and a capped number per rolling day.

Both bounds are keyed by *engine*, not by process. The page warms every parser it has selected, one
`POST /api/warm` each, and six independent scale-to-zero GPUs share this one process; a single
process-wide slot boots whichever request wins the race and leaves the other five cold. That is what
a shared counter did once the picker learned to ask for several parsers at once.
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
    """What one engine may spend on being warm.

    `min_interval_seconds` has to sit *strictly* inside `scaledown_window` (120 s in
    `deploy/modal/serve_*.py`). Equal is a tie, and a tie is decided by whichever side of the second
    the request lands on: the container is allowed to die at the same instant the next warm is
    allowed, so the warm path becomes a coin flip. An engine that may only be woken after it scaled
    down is cold and un-warmable at the same time — the cold window this whole path exists to remove.
    `timeout_seconds` has to outlast the boot it waits for (2-3 minutes), or a successful warm is
    reported as a failure.
    """

    enabled: bool = True
    min_interval_seconds: float = 90
    daily_cap: int = 40
    timeout_seconds: float = 300


def _in_background(job: Callable[[], None]) -> None:
    threading.Thread(target=job, name="warm-engine", daemon=True).start()


class EngineWarmer:
    """Starts a lazy engine booting: one boot per engine per interval, `daily_cap` per engine a day.

    The windows are keyed by mode because the page warms every parser it has selected, one request
    each, and the six GPUs boot and bill independently. One process-wide slot would hand the boot to
    whichever request won the race and leave the rest cold — which is what a shared counter did once
    the picker learned to ask for several parsers at once.

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
        self._last_started: dict[str, float] = {}
        self._starts: dict[str, deque[float]] = {}
        # One page can ping every parser at once, so the window is read and closed under a lock:
        # without it two requests for the same engine both see a closed window and both boot.
        self._lock = threading.Lock()

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

        with self._lock:
            now = self._clock.monotonic()
            starts = self._starts.setdefault(mode, deque())
            while starts and now - starts[0] >= ONE_DAY:
                starts.popleft()
            if len(starts) >= self._policy.daily_cap:
                return WarmStatus.CAPPED
            last = self._last_started.get(mode)
            if last is not None and now - last < self._policy.min_interval_seconds:
                return WarmStatus.THROTTLED
            self._last_started[mode] = now
            starts.append(now)

        self._submit(lambda: self._warm(engine))  # outside the lock: it starts a thread
        return WarmStatus.STARTED

    def report(self) -> dict[str, dict[str, object]]:
        """What each engine's warm budget looks like, without starting anything.

        `/api/health` answers with this, so "is the warm path working" is one request rather than an
        investigation of containers, scale-down windows and a capped counter nobody can see.
        """
        with self._lock:
            now = self._clock.monotonic()
            report: dict[str, dict[str, object]] = {}
            for mode in self._registry.modes:
                if not isinstance(self._engines.get(mode), PreWarmable):
                    continue  # a native parser has no GPU to keep warm
                starts = self._starts.setdefault(mode, deque())
                while starts and now - starts[0] >= ONE_DAY:
                    starts.popleft()
                last = self._last_started.get(mode)
                wait = self._policy.min_interval_seconds if last is None else max(
                    0.0, self._policy.min_interval_seconds - (now - last)
                )
                report[mode] = {
                    "used_today": len(starts),
                    "daily_cap": self._policy.daily_cap,
                    "interval_seconds": self._policy.min_interval_seconds,
                    "next_warm_in": round(wait, 1),
                }
            return report

    def _warm(self, engine: PreWarmable) -> None:
        try:
            warmed = engine.warm(self._policy.timeout_seconds)
        except Exception as error:  # noqa: BLE001 - a failed warm must not become a failed upload
            sys.stderr.write(f"warming {engine.spec.id} failed: {error}\n")
            return
        if not warmed:
            sys.stderr.write(f"warming {engine.spec.id} did not answer in time\n")
