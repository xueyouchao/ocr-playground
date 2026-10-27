"""The engines the roster can run, and the registry that names them by mode.

An `EngineSpec` is the row the page renders under each pane; `systems.json` carries the same six
fields for the eight recorded systems. `EngineRegistry` is the one place that maps a request's
`glm`/`fast` to a spec, so an unknown mode has a single, testable error.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

from .errors import UnknownMode

EVIDENCE_FIELDS = ("id", "name", "kind", "pipeline", "device", "needs")


@dataclass(frozen=True, slots=True)
class EngineSpec:
    """What the page says about a parser: who it is and how it reads."""

    id: str
    name: str
    kind: str
    pipeline: str
    device: str
    needs: str

    def as_dict(self) -> dict[str, str]:
        """The six fields `collect.py` copies out of `systems.json`, in the same order."""
        return {field: getattr(self, field) for field in EVIDENCE_FIELDS}


@dataclass(frozen=True, slots=True)
class EngineRegistry:
    """Mode -> spec, plus the default. Frozen: a request cannot invent an engine."""

    specs: Mapping[str, EngineSpec]
    default_mode: str

    def __post_init__(self) -> None:
        frozen = MappingProxyType(dict(self.specs))
        object.__setattr__(self, "specs", frozen)
        if self.default_mode not in frozen:
            raise ValueError(f"default mode {self.default_mode!r} is not in {tuple(frozen)!r}")

    @property
    def modes(self) -> tuple[str, ...]:
        return tuple(self.specs)

    def spec(self, mode: str) -> EngineSpec:
        try:
            return self.specs[mode]
        except KeyError:
            raise UnknownMode(mode, self.modes) from None

    def alternative_to(self, mode: str) -> str | None:
        """The mode to offer when `mode` is unavailable — the other CPU/GPU path."""
        return next((candidate for candidate in self.specs if candidate != mode), None)

    def __contains__(self, mode: object) -> bool:
        return mode in self.specs
