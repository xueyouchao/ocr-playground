"""Read a tool's version without letting its absence break a run.

The indexer puts these strings in its memo key: a new Tesseract must re-read the corpus, and a machine
without `tesseract` must still be able to run the GLM path.
"""
from __future__ import annotations

import subprocess
from collections.abc import Sequence


def first_line(command: Sequence[str], fallback: str = "unknown") -> str:
    try:
        output = subprocess.run(command, capture_output=True, text=True, timeout=20).stdout
    except (OSError, subprocess.SubprocessError):
        return fallback
    stripped = output.strip()
    return stripped.splitlines()[0] if stripped else fallback
