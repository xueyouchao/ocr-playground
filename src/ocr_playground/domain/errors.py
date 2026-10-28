"""Domain-layer exceptions.

These are the failures a caller can act on: the bytes are not something we read, the document is too
long, the mode does not exist. They carry a finished sentence because the phrase is part of the
contract — the upload API returns it verbatim to the browser — and keeping it here means the wording
does not drift between the HTTP adapter, the indexer and the tests.
"""
from __future__ import annotations

from collections.abc import Iterable


class DomainError(Exception):
    """A request that cannot be honoured, whichever adapters are plugged in."""


class UnsupportedMediaType(DomainError):
    def __init__(self, media_type: str) -> None:
        self.media_type = media_type
        super().__init__(f"{media_type or 'unknown type'} is not a PDF or an image")


class EmptyDocument(DomainError):
    def __init__(self) -> None:
        super().__init__("that upload is empty")


class PageLimitExceeded(DomainError):
    def __init__(self, pages: int, limit: int) -> None:
        self.pages = pages
        self.limit = limit
        super().__init__(f"that PDF has {pages} pages; this demo reads at most {limit}")


class UnreadablePageCount(DomainError):
    def __init__(self) -> None:
        super().__init__("could not read the page count from that PDF")


class TooManyModes(DomainError):
    """A comparison asks several engines to read the same page; the picker is meant to stay small."""

    def __init__(self, limit: int) -> None:
        self.limit = limit
        super().__init__(f"a comparison reads at most {limit} parsers at once")


class UnknownMode(DomainError):
    def __init__(self, mode: str, modes: Iterable[str]) -> None:
        self.mode = mode
        self.modes = tuple(modes)
        super().__init__(f"{mode!r} is not an extraction mode; use {' or '.join(self.modes)}")
