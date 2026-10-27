"""Ports: the seams the application is allowed to talk to.

Protocols, not base classes, so an adapter is any object with the right shape and a fake in a test is
five lines. `domain` depends on none of the implementations; `composition` decides which one is
plugged in.
"""
from __future__ import annotations

from collections.abc import Callable, Sequence
from contextlib import AbstractContextManager
from pathlib import Path
from typing import Protocol, runtime_checkable

from .document import PageImage, PageLimit, SourceDocument
from .engine import EngineSpec

Cancel = Callable[[], bool]
"""True once nobody is waiting for this reading any more — the browser closed the connection.

Polled where abandoning the work is still free: before a page, before an upload, between attempts.
It cannot interrupt a request already on the wire, so an engine treats it as a checkpoint rather
than an abort, and a read that notices it raises `ClientGone` instead of returning text nobody
will read.
"""


class Workspace(Protocol):
    """A throwaway directory for rendered pages and previews."""

    def create(self) -> AbstractContextManager[Path]:
        ...


class PageRenderer(Protocol):
    """Turn a source document into page images in reading order, honouring the limit."""

    def render(self, source: SourceDocument, scratch: Path, limit: PageLimit) -> Sequence[PageImage]:
        ...


class PreviewEncoder(Protocol):
    """A small, self-contained image a browser can draw without a second request."""

    def encode(self, page: PageImage) -> str:
        ...


class PageOcr(Protocol):
    """One engine: how it is described, whether it can run, and how it reads a page."""

    spec: EngineSpec

    @property
    def available(self) -> bool:
        """Can this engine run at all right now? No network, no work."""

    def identity(self) -> str:
        """Everything that changes what it emits, for the indexer's memo key."""

    def read(
        self, page: PageImage, deadline: float | None, cancel: Cancel | None = None
    ) -> str:
        """The page's text. `deadline` is the monotonic time the document must finish by.

        `cancel` is asked at each point where stopping costs nothing; a parked engine that sees it
        raises `ClientGone` so the caller can stop paying for a page nobody is waiting for.
        """


@runtime_checkable
class PreWarmable(Protocol):
    """An engine whose backend boots lazily, and can be woken ahead of the first page.

    Separate from `PageOcr` because only a remote, scale-to-zero engine has anything to warm: the
    local OCR stack is ready the moment its subprocess is.
    """

    def warm(self, timeout: float) -> bool:
        """Ask the backend to boot. True if it answered inside `timeout`."""


class Clock(Protocol):
    def monotonic(self) -> float:
        """Seconds on a monotonic clock; injected so a budget can be tested without waiting."""
