"""The two system services a use case needs: time and scratch space."""
from __future__ import annotations

import tempfile
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path


class SystemClock:
    def monotonic(self) -> float:
        return time.monotonic()


class TempWorkspace:
    """A temporary directory that is deleted before the response is sent."""

    def __init__(self, prefix: str = "ocr-playground-") -> None:
        self._prefix = prefix

    @contextmanager
    def create(self) -> Iterator[Path]:
        with tempfile.TemporaryDirectory(prefix=self._prefix) as name:
            yield Path(name)
