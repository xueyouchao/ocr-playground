"""What the page counts show: characters, tokens, and the share that are numbers.

This is the one measurement in the repository. It used to exist twice — once in `pipeline.py` for an
upload, once in `collect.py` for the static build — which is exactly the kind of duplication that lets
two halves of a page disagree. The extraction is a domain concern because it is a statement about the
reading, not about HTTP or files.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

# A tag is closed by a space, not deleted: "<b>x</b>y" is two tokens ("x", "y"), not one ("xy").
_TAG = re.compile(r"<[^>]+>")
_TOKEN = re.compile(r"\S+")
_NUMERIC = re.compile(r"[\d.,:%/\-+()]+")


@dataclass(frozen=True, slots=True)
class ReadingStats:
    """The counts every pane shows beside its text."""

    chars: int
    tokens: int
    numeric_share: float

    @classmethod
    def from_text(cls, text: str) -> ReadingStats:
        stripped = _TAG.sub(" ", text)
        tokens = _TOKEN.findall(stripped)
        numeric = [token for token in tokens if _NUMERIC.fullmatch(token)]
        return cls(
            chars=len(text),
            tokens=len(tokens),
            numeric_share=round(len(numeric) / len(tokens), 3) if tokens else 0.0,
        )

    def as_dict(self) -> dict[str, int | float]:
        return {"chars": self.chars, "tokens": self.tokens, "numeric_share": self.numeric_share}


@dataclass(frozen=True, slots=True)
class PageReading:
    """One page's text and its measurement, attributed to the engine that produced it."""

    engine_id: str
    text: str
    stats: ReadingStats

    @classmethod
    def of(cls, engine_id: str, text: str) -> PageReading:
        return cls(engine_id=engine_id, text=text, stats=ReadingStats.from_text(text))
