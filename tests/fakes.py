"""Doubles for the ports, plus a wired use case.

Everything the application needs is behind a protocol, so a test gets a real `ReadDocumentService`
with fake engines and a fake filesystem instead of monkeypatching module globals. The service under
test is the production class — only its collaborators are fake.
"""
from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ocr_playground.application.capacity import SlotsBusy
from ocr_playground.application.read_document import ReadDocumentService
from ocr_playground.domain.document import (
    DocumentPage,
    PageImage,
    PageLimit,
    PageReading,
    ReadDocument,
    SourceDocument,
)
from ocr_playground.domain.engine import EngineRegistry, EngineSpec

FAST = EngineSpec(
    id="liteparse-tesseract",
    name="LiteParse + Tesseract",
    kind="baseline",
    pipeline="raster + bundled Tesseract, parsed live on upload",
    device="this server (CPU)",
    needs="nothing",
)
GLM = EngineSpec(
    id="glm-ocr",
    name="GLM-OCR",
    kind="vlm",
    pipeline="0.9B VLM on a Modal L4 (vLLM) — cold starts can take minutes",
    device="GPU (Modal L4)",
    needs="GLM_OCR_BASE_URL",
)
# The page's default parser, and a second warmable engine: one page can ask for several at once.
CHANDRA = EngineSpec(
    id="chandra-2",
    name="Chandra OCR 2",
    kind="vlm",
    pipeline="Chandra 2 on a Modal L4 — layout-preserving markdown",
    device="GPU (Modal L4)",
    needs="CHANDRA_OCR_BASE_URL",
)


def registry(default_mode: str = "glm") -> EngineRegistry:
    return EngineRegistry({"glm": GLM, "fast": FAST}, default_mode)


class FakeClock:
    """A monotonic clock that only moves when a test moves it."""

    def __init__(self, now: float = 1000.0) -> None:
        self.now = now

    def monotonic(self) -> float:
        return self.now


class _ExistingDirectory:
    def __init__(self, path: Path) -> None:
        self._path = path

    def __enter__(self) -> Path:
        return self._path

    def __exit__(self, *exc: object) -> bool:
        return False


class FakeWorkspace:
    """Hands back a directory a test owns, instead of creating and deleting a temp one."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root)

    def create(self):
        self.root.mkdir(parents=True, exist_ok=True)
        return _ExistingDirectory(self.root)


class FakeRenderer:
    def __init__(self, pages: int = 2) -> None:
        self.pages = pages
        self.calls: list[tuple[SourceDocument, Path, PageLimit]] = []

    def render(self, source: SourceDocument, scratch: Path, limit: PageLimit) -> list[PageImage]:
        self.calls.append((source, scratch, limit))
        return [PageImage(number, Path(scratch) / f"page-{number}.png")
                for number in range(1, self.pages + 1)]


class FakePreviewEncoder:
    def __init__(self) -> None:
        self.encoded: list[int] = []

    def encode(self, page: PageImage) -> str:
        self.encoded.append(page.number)
        return f"data:page-{page.number}"


class FakeEngine:
    def __init__(self, spec: EngineSpec, *, available: bool = True, text: Any = None,
                 error: BaseException | None = None) -> None:
        self.spec = spec
        self._available = available
        self._text = text
        self._error = error
        self.reads: list[tuple[int, float | None]] = []

    @property
    def available(self) -> bool:
        return self._available

    def identity(self) -> str:
        return f"{self.spec.id} identity"

    def read(
        self, page: PageImage, deadline: float | None, cancel: object = None
    ) -> str:
        self.reads.append((page.number, deadline))
        if self._error is not None:
            raise self._error
        if callable(self._text):
            return self._text(page.number)
        return self._text if self._text is not None else f"text of page {page.number}"


class FakeWarmEngine(FakeEngine):
    """An engine that can be told to boot before the first page (GLM-OCR on a scale-to-zero GPU)."""

    def __init__(self, spec: EngineSpec, *, available: bool = True, answers: bool = True) -> None:
        super().__init__(spec, available=available)
        self.answers = answers
        self.warm_calls: list[float] = []

    def warm(self, timeout: float = 300) -> bool:
        self.warm_calls.append(timeout)
        return self.answers


@dataclass
class Rig:
    service: ReadDocumentService
    glm: FakeEngine
    fast: FakeEngine
    renderer: FakeRenderer
    previews: FakePreviewEncoder
    clock: FakeClock
    workspace: Path


def build_rig(
    tmp_path: Path,
    *,
    pages: int = 2,
    glm_available: bool = True,
    fast_available: bool = True,
    budget: float = 900,
    max_pages: int | None = 5,
    previews: bool = True,
    glm_error: BaseException | None = None,
    fast_error: BaseException | None = None,
    default_mode: str = "glm",
) -> Rig:
    workspace = Path(tmp_path) / "scratch"
    clock = FakeClock()
    glm = FakeEngine(GLM, available=glm_available, error=glm_error)
    fast = FakeEngine(FAST, available=fast_available, error=fast_error)
    renderer = FakeRenderer(pages)
    preview_encoder = FakePreviewEncoder()
    service = ReadDocumentService(
        engines={"glm": glm, "fast": fast},
        registry=registry(default_mode),
        renderer=renderer,
        preview_encoder=preview_encoder,
        workspace=FakeWorkspace(workspace),
        clock=clock,
        budget_seconds=budget,
        max_pages=max_pages,
        previews=previews,
    )
    return Rig(service, glm, fast, renderer, preview_encoder, clock, workspace)


def sample_document(
    pages: int = 1,
    *,
    text: str = "hello",
    preview: str | None = "data:x",
    engines: tuple[EngineSpec, ...] = (FAST,),
) -> ReadDocument:
    return ReadDocument(
        name="form.png",
        note="a note",
        engines=engines,
        pages=tuple(
            DocumentPage(
                number=number,
                readings=tuple(
                    PageReading.of(engine.id, f"{text} {number}") for engine in engines
                ),
                preview=preview,
            )
            for number in range(1, pages + 1)
        ),
    )


class FakeSlots:
    """Counts acquisitions and releases; `grant=False` models a saturated server."""

    def __init__(self, grant: bool = True) -> None:
        self.grant = grant
        self.acquired = 0
        self.released = 0

    @contextmanager
    def hold(self) -> Iterator[None]:
        self.acquired += 1
        if not self.grant:
            raise SlotsBusy(30)
        try:
            yield
        finally:
            self.released += 1


class StubService:
    """A use case that answers (or fails) immediately, for mapping tests."""

    def __init__(self, *, document: ReadDocument | None = None,
                 error: BaseException | None = None) -> None:
        self.document = document
        self.error = error
        self.calls: list[tuple[SourceDocument, str, dict]] = []
        self.registry = registry()
        self._engines = {
            "glm": type("Engine", (), {"spec": GLM})(),
            "fast": type("Engine", (), {"spec": FAST})(),
        }

    def require_mode(self, mode: str):
        self.registry.spec(mode)
        return self._engines[mode]

    def execute(self, source: SourceDocument, mode: str, **kwargs):
        self.calls.append((source, mode, kwargs))
        if self.error is not None:
            raise self.error
        return self.document
