"""Shared fixtures.

The suite is deliberately offline: no GPU endpoint, no LiteParse, no poppler. Anything that would
shell out is replaced at a port, which is what the fakes in `tests/fakes.py` are for.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
# The importable tree lives under src/; the repository root holds data, site and scripts.
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

# A 1x1 transparent PNG; enough for "is this bytes an image" without shipping binary fixtures.
PNG_1PX = bytes.fromhex(
    "89504e470d0a1a0a0000000d494844520000000100000001080600000"
    "01f15c4890000000a49444154789c6360000002000100ffff03000006"
    "0005574b3c9a0000000049454e44ae426082"
)


@pytest.fixture
def png_bytes() -> bytes:
    return PNG_1PX


@pytest.fixture
def repo_root() -> Path:
    return ROOT
