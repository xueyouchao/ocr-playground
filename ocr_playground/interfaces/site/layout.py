"""Where the repository keeps things, named once.

`data/` is what a run produced — the scan, the roster that describes each parser, the records each
parser left behind. `site/` is what the build publishes — the template, the page images it points
at, and the two files the build generates from them. Nothing else spells either directory out, so
moving one is a change to this file and not to a hunt through string literals.
"""
from __future__ import annotations

from pathlib import Path

DATA = "data"
SITE = "site"


def data(root: Path) -> Path:
    """The scan, the roster and the recorded runs."""
    return root / DATA


def site(root: Path) -> Path:
    """The template, its page images, and what `make site` writes."""
    return root / SITE
