"""What may be read at all: PDF and image bytes, and nothing else.

A `MediaType` is a value object, not a string. The upload API lowercases and strips what a client
sent, but everything downstream asks `is_pdf` / `is_image` instead of comparing to a list, so adding a
format is one line here rather than a hunt through the readers.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .errors import UnsupportedMediaType

PDF_TYPES = frozenset({"application/pdf", "application/x-pdf"})
IMAGE_TYPES = frozenset({"image/png", "image/jpeg", "image/tiff", "image/webp"})
SUPPORTED_TYPES = PDF_TYPES | IMAGE_TYPES


@dataclass(frozen=True, slots=True)
class MediaType:
    """The media type of a payload, normalised the way a request header is."""

    value: str

    @classmethod
    def parse(cls, header: str | None) -> MediaType:
        """`"image/png; charset=binary"` -> `image/png`; an empty header stays empty."""
        return cls((header or "").split(";")[0].strip().lower())

    @property
    def is_pdf(self) -> bool:
        return self.value in PDF_TYPES

    @property
    def is_image(self) -> bool:
        return self.value in IMAGE_TYPES

    @property
    def is_supported(self) -> bool:
        return self.value in SUPPORTED_TYPES

    def suffix(self, name: str) -> str:
        """The extension for a scratch copy: the uploaded one if it has one, else the format's."""
        return Path(name).suffix or (".pdf" if self.is_pdf else ".png")

    def require_supported(self) -> None:
        if not self.is_supported:
            raise UnsupportedMediaType(self.value)

    def __str__(self) -> str:
        return self.value
