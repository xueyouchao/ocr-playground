"""The vocabulary of the reading, usable without an adapter.

Importing from `ocr_playground.domain` is the short form for the names an interface actually needs;
anything more specific belongs in its own module.
"""
from .document import DocumentPage, PageImage, PageLimit, ReadDocument, SourceDocument
from .engine import EngineRegistry, EngineSpec
from .errors import (
    DomainError,
    EmptyDocument,
    PageLimitExceeded,
    UnknownMode,
    UnreadablePageCount,
    UnsupportedMediaType,
)
from .media import IMAGE_TYPES, PDF_TYPES, SUPPORTED_TYPES, MediaType
from .reading import PageReading, ReadingStats

__all__ = [
    "IMAGE_TYPES",
    "PDF_TYPES",
    "SUPPORTED_TYPES",
    "DocumentPage",
    "DomainError",
    "EmptyDocument",
    "EngineRegistry",
    "EngineSpec",
    "MediaType",
    "PageImage",
    "PageLimit",
    "PageLimitExceeded",
    "PageReading",
    "ReadDocument",
    "ReadingStats",
    "SourceDocument",
    "UnknownMode",
    "UnreadablePageCount",
    "UnsupportedMediaType",
]
