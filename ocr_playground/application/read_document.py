"""Read one document: the use case the upload API and the indexer share.

The order is deliberate. The engine is resolved and checked *before* any work, so an unknown mode or
an unconfigured GPU does not queue behind two real uploads. The page budget starts after rendering,
so slow poppler time is not charged to the model. The preview is produced from the same rendered page
the engine read, so what the browser shows is what was parsed.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor

from ..domain.document import (
    DocumentPage,
    PageFailure,
    PageImage,
    PageLimit,
    ReadDocument,
    SourceDocument,
)
from ..domain.engine import EngineRegistry, EngineSpec
from ..domain.ports import Clock, PageOcr, PageRenderer, PreviewEncoder, Workspace
from ..domain.reading import PageReading
from ..domain.errors import UnknownMode
from .errors import ApplicationError, EngineFailure, EngineUnavailable


def reading_note(engines: Sequence[PageOcr]) -> str:
    """How a reading names itself: one engine, or the comparison the visitor asked for."""
    if len(engines) == 1:
        return f"read by {engines[0].spec.name}"
    return "read side by side by " + " + ".join(engine.spec.name for engine in engines)


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
        modes: str | Sequence[str],
        *,
        previews: bool | None = None,
        note: str | None = None,
        limit: PageLimit | None = None,
    ) -> ReadDocument:
        """Read `source` with one mode, or with several at once to compare them.

        Every engine is resolved before any work starts, so an unknown mode or an unconfigured GPU
        does not queue behind a real upload. The engines then read the same rendered page
        concurrently: each is its own HTTP endpoint, so overlapping them costs the same GPU-minutes
        as running them one after another and takes the time of the slowest instead of the sum.

        One engine failing does not sink the comparison — the pages record what answered and name
        what did not. Only a document that produced no reading at all is an error.
        """
        requested = self._requested(modes)
        engines = [self.require_mode(mode) for mode in requested]
        page_limit = PageLimit(self._max_pages) if limit is None else limit
        with_previews = self._default_previews if previews is None else previews
        # One budget per engine, not per document: a comparison may be waiting on four cold starts.
        budgets = {
            engine.spec.id: self._clock.monotonic() + self._budget_seconds for engine in engines
        }

        pages: list[DocumentPage] = []
        failed: dict[str, Exception] = {}
        with self._workspace.create() as scratch:
            images = self._renderer.render(source, scratch, page_limit)
            for image in images:
                readings, failures = self._read_page(engines, image, budgets, failed)
                preview = self._preview_encoder.encode(image) if with_previews else None
                pages.append(DocumentPage(
                    number=image.number,
                    readings=tuple(readings),
                    preview=preview,
                    failures=tuple(failures),
                ))

        if not any(page.readings for page in pages):
            raise self._first_failure(engines, failed)

        return ReadDocument(
            name=source.name,
            note=note or reading_note(engines),
            engines=tuple(engine.spec for engine in engines),
            pages=tuple(pages),
        )

    def _requested(self, modes: str | Sequence[str]) -> tuple[str, ...]:
        """One mode or a comma-separated comparison, without duplicates and in the order given."""
        if isinstance(modes, str):
            modes = [part for part in modes.replace(" ", ",").split(",") if part]
        ordered: list[str] = []
        for mode in modes:
            if mode not in ordered:
                ordered.append(mode)
        if not ordered:
            raise UnknownMode("")
        return tuple(ordered)

    def _read_page(
        self,
        engines: Sequence[PageOcr],
        image: PageImage,
        budgets: dict[str, float],
        failed: dict[str, Exception],
    ) -> tuple[list[PageReading], list[PageFailure]]:
        """Read one page with every engine that is still working, and keep the ones that answer."""
        pending = [engine for engine in engines if engine.spec.id not in failed]
        readings: list[PageReading] = []
        if pending:
            with ThreadPoolExecutor(max_workers=len(pending)) as pool:
                running = {
                    engine.spec.id: pool.submit(engine.read, image, budgets[engine.spec.id])
                    for engine in pending
                }
                for engine in pending:
                    try:
                        text = running[engine.spec.id].result()
                    except Exception as error:  # noqa: BLE001 — see the docstring: partial is the point
                        failed[engine.spec.id] = error
                    else:
                        readings.append(PageReading.of(engine.spec.id, text))

        # An engine that failed stays failed: later pages name the same reason rather than retrying
        # a cold endpoint for every page of the document.
        return readings, [
            PageFailure(engine.spec.id, self._reason(failed[engine.spec.id]))
            for engine in engines
            if engine.spec.id in failed
        ]

    def _first_failure(self, engines: Sequence[PageOcr], failed: Mapping[str, Exception]) -> Exception:
        """Nothing answered: report the primary engine's failure, which is the one that was asked for."""
        for engine in engines:
            error = failed.get(engine.spec.id)
            if isinstance(error, ApplicationError):
                return error
            if error is not None:
                return EngineFailure(f"{engine.spec.name} failed: {self._reason(error)}")
        return EngineFailure("no engine produced a reading")

    @staticmethod
    def _reason(error: Exception) -> str:
        text = str(error).strip()
        return text or type(error).__name__

