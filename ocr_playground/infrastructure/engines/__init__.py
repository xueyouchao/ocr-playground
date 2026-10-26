"""The two live readers, each one adapter behind the same port."""
from .glm_ocr import GLM_SPEC, GlmOcrEngine
from .liteparse import FAST_SPEC, LiteParseEngine

__all__ = ["FAST_SPEC", "GLM_SPEC", "GlmOcrEngine", "LiteParseEngine"]
