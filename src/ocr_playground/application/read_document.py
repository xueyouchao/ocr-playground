"""Read one document: the use case the upload API and the indexer share.

The order is deliberate. The engine is resolved and checked *before* any work, so an unknown mode or
an unconfigured GPU does not queue behind two real uploads. The page budget starts after rendering,
so slow poppler time is not charged to the model. The preview is produced from the same rendered page
the engine read, so what the browser shows is what was parsed.
"""
from __future__ import annotations

import random
import time
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor, wait

from ..domain.document import (
    DocumentPage,
    PageFailure,
    PageImage,
    PageLimit,
    ReadDocument,
    SourceDocument,
)
from ..domain.engine import EngineRegistry, EngineSpec
from ..domain.ports import Cancel, Clock, PageOcr, PageRenderer, PreviewEncoder, Workspace
from ..domain.reading import PageReading
from ..domain.errors import UnknownMode
from .errors import (
    ApplicationError,
    ClientGone,
    EngineFailure,
    EngineTimeout,
    EngineUnavailable,
)

_CANCEL_POLL = 0.25
"""How often the wait asks whether the client is still there.

Short enough that closing the tab releases the parse slot promptly, long enough that a comparison
of four engines does not spend the page spinning on a socket probe.
"""


_COMPARISON_BUDGET = 240.0
"""How long a comparison may go without *anything* to show.

A single engine keeps the document budget — one engine is the whole page, and waiting is the only way
to get it. A comparison is different: four rows that all say nothing is not a page, and a cold boot
longer than four minutes is cheaper to retry than to sit through. Observed boots run 150–450 s.
"""

_STAGGER_LOW = 0.2
_STAGGER_HIGH = 0.4
"""How much later each row starts than the one before it.

Four models means four apps asked to boot. Asked in the same instant they race one GPU and three queue;
a fraction of a second apart they arrive as separate requests. The cost is under a second and a half
per comparison against boots measured in minutes.
"""

_STALL_GRACE = 90.0
"""Once one row has answered, how long the others get before the page goes out without them.

Enough for a container that is nearly up, short enough that one dead row cannot hold three good ones.
"""


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
        # An unavailable parser should point at one that is ready now: the path running on this host
        # answers in seconds, while another GPU may be just as cold or missing. Rank by *where* it
        # runs — `device` — not by `kind`, which says how a page is read; ranking by kind sends a
        # missing MinerU to a cold GLM once both are no longer the same kind.
        runnable = [
            candidate for candidate in self.registry.modes
            if candidate != mode and self._engines[candidate].available
        ]
        alternative = (
            next((c for c in runnable if self.registry.spec(c).device.startswith("native")), None)
            or next((c for c in runnable if self.registry.spec(c).kind != spec.kind), None)
            or (runnable[0] if runnable else None)
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
        cancel: Cancel | None = None,
    ) -> ReadDocument:
        """Read `source` with one mode, or with several at once to compare them.

        Every engine is resolved before any work starts, so an unknown mode or an unconfigured GPU
        does not queue behind a real upload. The engines then read the same rendered page
        concurrently: each is its own HTTP endpoint, so overlapping them costs the same GPU-minutes
        as running them one after another and takes the time of the slowest instead of the sum.

        One engine failing does not sink the comparison — the pages record what answered and name
        what did not. Only a document that produced no reading at all is an error.

        `cancel` is asked before each page and handed to every engine: a client that hung up stops the
        document before the next page, and a parked engine stops waiting instead of holding the slot.
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
                if cancel is not None and cancel():
                    raise ClientGone()
                readings, failures = self._read_page(engines, image, budgets, failed, cancel)
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
        cancel: Cancel | None = None,
    ) -> tuple[list[PageReading], list[PageFailure]]:
        """Read one page with every engine that is still working, and keep the ones that answer."""
        pending = [engine for engine in engines if engine.spec.id not in failed]
        readings: list[PageReading] = []
        if pending:
            pool = ThreadPoolExecutor(max_workers=len(pending))
            try:
                running = {
                    engine.spec.id: pool.submit(
                        self._read_in_turn, index, engine, image, budgets[engine.spec.id], cancel
                    )
                    for index, engine in enumerate(pending)
                }
                # Waiting in request order would hide a cancellation behind the first engine: a row
                # parked in a request it cannot interrupt (an upload already on the wire) never sees
                # the token, so the loop would sit on it for the whole budget while another row has
                # already said the client is gone. So completion is harvested in arrival order and
                # the token is asked while waiting; the readings are put back in request order below.
                answers: dict[str, str] = {}
                waiting = set(running.values())
                # The watchdog runs on the wall clock, not the injected one: a frozen clock must not be
                # able to disarm the thing that stops a request from hanging.
                started = time.monotonic()
                # A comparison that has produced nothing by now is not going to; one that has something
                # only gives the stragglers a grace period, so a dead row cannot hold the others.
                ceiling = started + _COMPARISON_BUDGET if len(engines) > 1 else None
                answered_at: float | None = None
                while waiting:
                    waiting = set(wait(waiting, timeout=_CANCEL_POLL).not_done)
                    for engine in pending:
                        future, engine_id = running[engine.spec.id], engine.spec.id
                        if engine_id in answers or engine_id in failed or not future.done():
                            continue
                        try:
                            answers[engine_id] = future.result()
                        except ClientGone:
                            raise  # nobody wants this page: stop the document, not a named failure
                        except Exception as error:  # noqa: BLE001 — partial is the point
                            failed[engine_id] = error
                    now = time.monotonic()
                    if answers and answered_at is None:
                        answered_at = now
                    if cancel is not None and cancel():
                        raise ClientGone()
                    if ceiling is not None and now > ceiling:
                        break
                    if answered_at is not None and now - answered_at > _STALL_GRACE:
                        break
                # Whatever is still running is reported by name instead of held open: the page shows
                # the rows that answered and says which ones did not, now rather than in fifteen minutes.
                for engine in pending:
                    if engine.spec.id not in answers and engine.spec.id not in failed:
                        failed[engine.spec.id] = EngineTimeout(
                            f"{engine.spec.name} did not answer in time — try again; the GPU is "
                            "booting or busy"
                        )
                readings.extend(
                    PageReading.of(engine.spec.id, answers[engine.spec.id])
                    for engine in pending
                    if engine.spec.id in answers
                )
            except BaseException:
                # A parked engine must not hold the request open after the client has gone; its own
                # thread ends when its socket does, and nobody waits for it here.
                pool.shutdown(wait=False, cancel_futures=True)
                raise
            if any(not future.done() for future in running.values()):
                pool.shutdown(wait=False, cancel_futures=True)  # stopped waiting; do not join it
            else:
                pool.shutdown()

        # An engine that failed stays failed: later pages name the same reason rather than retrying
        # a cold endpoint for every page of the document.
        return readings, [
            PageFailure(engine.spec.id, self._reason(failed[engine.spec.id]))
            for engine in engines
            if engine.spec.id in failed
        ]

    @staticmethod
    def _read_in_turn(
        index: int, engine: PageOcr, image: PageImage, deadline: float, cancel: Cancel | None
    ) -> str:
        """Start this engine after its turn, so a comparison does not ask every app to boot at once."""
        if index:
            time.sleep(index * random.uniform(_STAGGER_LOW, _STAGGER_HIGH))
        return engine.read(image, deadline, cancel)

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

