"""Read one document: the use case the upload API and the indexer share.

The order is deliberate. The engine is resolved and checked *before* any work, so an unknown mode or
an unconfigured GPU does not queue behind two real uploads. The page budget starts after rendering,
so slow poppler time is not charged to the model. The preview is produced from the same rendered page
the engine read, so what the browser shows is what was parsed.
"""
from __future__ import annotations

from collections.abc import Mapping

from ..domain.document import (
    DocumentPage,
    PageLimit,
    ReadDocument,
    SourceDocument,
)
from ..domain.engine import EngineRegistry, EngineSpec
from ..domain.ports import Clock, PageOcr, PageRenderer, PreviewEncoder, Workspace
from ..domain.reading import PageReading
from .errors import EngineUnavailable


class ReadDocumentService:
    """One use case, one instance: the engines and ports are wired once and shared."""

    def __init__(
        self,
        *,
        engines: Mapping[str, PageOcr],
        registry: EngineRegistry,
        renderer: PageRenderer,
        preview_encoder: PreviewEncoder,
        workspace: Workspace,
        clock: Clock,
        budget_seconds: float,
        max_pages: int | None,
        previews: bool = True,
    ) -> None:
        self._engines = dict(engines)
        if set(self._engines) != set(registry.specs):
            raise ValueError(
                f"engines {sorted(self._engines)} do not match the registry {sorted(registry.specs)}"
            )
        self.registry = registry
        self._renderer = renderer
        self._preview_encoder = preview_encoder
        self._workspace = workspace
        self._clock = clock
        self._budget_seconds = budget_seconds
        self._max_pages = max_pages
        self._default_previews = previews

    def require_mode(self, mode: str) -> PageOcr:
        """Resolve a mode to a runnable engine, or raise the error the caller should report."""
        spec = self.registry.spec(mode)  # UnknownMode for a mode nobody registered
        engine = self._engines[mode]
        if not engine.available:
            raise EngineUnavailable(self._unavailable_message(mode, spec))
        return engine

    def _unavailable_message(self, mode: str, spec: EngineSpec) -> str:
        # Prefer a different *kind* of engine: a GPU parser that is not configured should point at
        # the CPU path, which is ready, rather than at another GPU that may be just as cold.
        runnable = [
            candidate for candidate in self.registry.modes
            if candidate != mode and self._engines[candidate].available
        ]
        alternative = next(
            (candidate for candidate in runnable
             if self.registry.spec(candidate).kind != spec.kind),
            runnable[0] if runnable else None,
        )
        if alternative is None:
            return f"{spec.name} is not configured on this server"
        other = self.registry.spec(alternative)
        return (
            f"{spec.name} is not configured on this server; "
            f"use the {alternative} option ({other.name})"
        )

    def execute(
        self,
        source: SourceDocument,
        mode: str,
        *,
        previews: bool | None = None,
        note: str | None = None,
        limit: PageLimit | None = None,
    ) -> ReadDocument:
        """Read `source` with `mode`. `limit` defaults to the configured upload cap."""
        engine = self.require_mode(mode)
        page_limit = PageLimit(self._max_pages) if limit is None else limit
        with_previews = self._default_previews if previews is None else previews

        pages: list[DocumentPage] = []
        with self._workspace.create() as scratch:
            images = self._renderer.render(source, scratch, page_limit)
            # One budget for the whole document: the cold start is paid on the first page and the
            # rest inherit what is left of it.
            deadline = self._clock.monotonic() + self._budget_seconds
            for image in images:
                text = engine.read(image, deadline)
                preview = self._preview_encoder.encode(image) if with_previews else None
                pages.append(DocumentPage(
                    number=image.number,
                    reading=PageReading.of(engine.spec.id, text),
                    preview=preview,
                ))

        return ReadDocument(
            name=source.name,
            note=note or f"read by {engine.spec.name}",
            engine=engine.spec,
            pages=tuple(pages),
        )
