"""The bound on concurrent parses.

The box has four cores and the fast path is CPU-bound; a GLM upload holds its slot while it waits on
the GPU, so two cold starts cannot stampede the endpoint. Two at a time is the whole policy; this
class is the only place it lives.
"""
from __future__ import annotations

import threading
from collections.abc import Iterator
from contextlib import contextmanager


class SlotsBusy(RuntimeError):
    """Every parse slot stayed taken for the whole wait."""

    def __init__(self, wait_seconds: float) -> None:
        super().__init__(f"no parse slot within {wait_seconds} s")


class ParseSlots:
    """A bounded gate around "read the body, then read the document"."""

    def __init__(self, size: int, wait_seconds: float = 30) -> None:
        if size < 1:
            raise ValueError("at least one parse slot is needed")
        self._slots = threading.BoundedSemaphore(size)
        self._wait_seconds = wait_seconds

    @contextmanager
    def hold(self) -> Iterator[None]:
        if not self._slots.acquire(timeout=self._wait_seconds):
            raise SlotsBusy(self._wait_seconds)
        try:
            yield
        finally:
            self._slots.release()
