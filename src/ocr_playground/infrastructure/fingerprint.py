"""A fingerprint of the reading code, for the indexer's memo key.

CocoIndex memoises the indexer's own AST, not the functions it calls, so without this a change to an
engine or to the use case would serve text the old reader produced. Hashing the package makes every
change to what the reader emits invalidate the memo — the safe direction to be wrong in.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

PACKAGE = Path(__file__).resolve().parents[1]


def reading_fingerprint(root: Path | None = None) -> str:
    """SHA-256 over every Python file under `root` (default: this package), truncated for readability."""
    base = PACKAGE if root is None else Path(root)
    digest = hashlib.sha256()
    for path in sorted(base.rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        digest.update(path.relative_to(base).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()[:12]
