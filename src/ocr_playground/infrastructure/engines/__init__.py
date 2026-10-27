"""The live readers, each one adapter behind the same port."""
from .liteparse import FAST_SPEC, LiteParseEngine
from .vlm import (
    CHANDRA_SPEC,
    GLM_SPEC,
    MINERU_SPEC,
    NEMOTRON_SPEC,
    PADDLE_SPEC,
    UNLIMITED_SPEC,
    VlmEngine,
)

__all__ = [
    "CHANDRA_SPEC",
    "FAST_SPEC",
    "GLM_SPEC",
    "MINERU_SPEC",
    "NEMOTRON_SPEC",
    "PADDLE_SPEC",
    "UNLIMITED_SPEC",
    "LiteParseEngine",
    "VlmEngine",
]
