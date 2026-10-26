"""The document aggregate: bytes in, one reading per page out.

`SourceDocument` is what an uploader handed us. `ReadDocument` is the finished aggregate every
consumer renders — the API, the indexer and the static build all speak it. Neither knows the evidence
JSON shape; that translation is an interface concern (`interfaces/evidence.py`).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .engine import EngineSpec
from .errors import EmptyDocument, PageLimitExceeded, UnsupportedMediaType
from .media import MediaType
from .reading import PageReading


@dataclass(frozen=True, slots=True)
class SourceDocument:
    """The bytes to read, with the name and type they arrived under."""

    name: str
    media_type: MediaType
    payload: bytes

    @classmethod
    def accepted(cls, payload: bytes, media_type: str | MediaType, name: str = "") -> SourceDocument:
        """Validate an upload once, at the boundary, so nothing downstream re-checks it."""
        media = media_type if isinstance(media_type, MediaType) else MediaType.parse(media_type)
        if not payload:
            raise EmptyDocument()
        if not media.is_supported:
            raise UnsupportedMediaType(media.value)
        return cls(name=name, media_type=media, payload=payload)

    @property
    def display_name(self) -> str:
        """The name a browser shows: the basename, never empty."""
        return Path(self.name).name or "upload"

    @property
    def suffix(self) -> str:
        return self.media_type.suffix(self.name)


@dataclass(frozen=True, slots=True)
class PageImage:
    """One rendered page, numbered from 1 in reading order."""

    number: int
    path: Path


@dataclass(frozen=True, slots=True)
class PageLimit:
    """How many pages a run may read. `None` means every page, which is the indexer's default."""

    maximum: int | None = None

    def __post_init__(self) -> None:
        if self.maximum is not None and self.maximum < 1:
            raise ValueError("a page limit must be at least 1, or None for every page")

    def check(self, pages: int) -> None:
        if self.maximum is not None and pages > self.maximum:
            raise PageLimitExceeded(pages, self.maximum)


@dataclass(frozen=True, slots=True)
class DocumentPage:
    """One page of the reading, with the preview only when the caller asked for one."""

    number: int
    reading: PageReading
    preview: str | None = None


@dataclass(frozen=True, slots=True)
class ReadDocument:
    """The aggregate root: one document, read once, by one engine."""

    name: str
    note: str
    engine: EngineSpec
    pages: tuple[DocumentPage, ...]

    @property
    def page_count(self) -> int:
        return len(self.pages)

    @property
    def display_name(self) -> str:
        return Path(self.name).name or "upload"

    def __post_init__(self) -> None:
        if not self.pages:
            raise ValueError("a read document needs at least one page")
