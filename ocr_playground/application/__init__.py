"""Use cases, expressed in domain terms.

An application service orchestrates ports: render this document, read each page with this engine,
attach a preview if asked. It does not know whether the renderer is poppler or a fake, whether the
engine is a subprocess or a GPU endpoint, or that an evidence JSON even exists.
"""

from .capacity import ParseSlots, SlotsBusy
from .errors import EngineFailure, EngineTimeout, EngineUnavailable
from .read_document import ReadDocumentService
from .warming import EngineWarmer, WarmPolicy, WarmStatus

__all__ = [
    "EngineFailure",
    "EngineTimeout",
    "EngineUnavailable",
    "EngineWarmer",
    "ParseSlots",
    "ReadDocumentService",
    "SlotsBusy",
    "WarmPolicy",
    "WarmStatus",
]
