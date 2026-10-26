"""The object graph, declared once.

python-dependency-injector is used for the thing it is actually good at here: declaring *what depends
on what* in one place, and letting a test or a different entry point replace any node without editing
the code under test.

- `config` is a `Configuration` provider. `build_container()` seeds it from the environment, a test
  seeds it with `from_dict`, and settings are built from whichever arrived — one set of defaults.
- `Singleton` for every node: the engines, renderer and use case are stateless, and rebuilding a
  `ProcessRunner` per request would only add noise.
- Nothing is resolved at import. Importing this module contacts no endpoint and runs no tool; the
  first request (or the indexer's first document) is what pulls the graph into existence.
"""
from __future__ import annotations

from collections.abc import Mapping

from dependency_injector import containers, providers

from ..application.capacity import ParseSlots
from ..application.read_document import ReadDocumentService
from ..domain.engine import EngineRegistry
from ..domain.ports import PageOcr
from ..infrastructure.engines import GlmOcrEngine, LiteParseEngine
from ..infrastructure.fingerprint import reading_fingerprint
from ..infrastructure.previews import ImageMagickPreviewEncoder
from ..infrastructure.process import ProcessRunner
from ..infrastructure.rendering import PopplerPageRenderer
from ..infrastructure.settings import Settings
from ..infrastructure.workspace import SystemClock, TempWorkspace


def build_registry(engines: Mapping[str, PageOcr], default_mode: str) -> EngineRegistry:
    """The roster is derived from the engines that were wired, never maintained twice."""
    return EngineRegistry(
        specs={mode: engine.spec for mode, engine in engines.items()},
        default_mode=default_mode,
    )


class Container(containers.DeclarativeContainer):
    config = providers.Configuration(default=Settings().as_dict())

    settings = providers.Singleton(Settings.from_mapping, config)

    clock = providers.Singleton(SystemClock)
    process = providers.Singleton(ProcessRunner, timeout=settings.provided.parse_timeout_seconds)
    workspace = providers.Singleton(TempWorkspace)

    renderer = providers.Singleton(
        PopplerPageRenderer, runner=process, dpi=settings.provided.render_dpi,
    )
    preview_encoder = providers.Singleton(
        ImageMagickPreviewEncoder,
        runner=process,
        max_px=settings.provided.preview_max_px,
        quality=settings.provided.preview_quality,
    )

    glm = providers.Singleton(
        GlmOcrEngine,
        base_url=settings.provided.glm_base_url,
        model=settings.provided.glm_model,
        prompt=settings.provided.glm_prompt,
        api_key=settings.provided.glm_api_key,
        request_timeout=settings.provided.glm_request_timeout,
    )
    fast = providers.Singleton(
        LiteParseEngine, runner=process, lit_path=settings.provided.lit_path,
    )
    engines = providers.Dict(glm=glm, fast=fast)

    registry = providers.Singleton(
        build_registry, engines=engines, default_mode=settings.provided.default_mode,
    )
    read_document = providers.Singleton(
        ReadDocumentService,
        engines=engines,
        registry=registry,
        renderer=renderer,
        preview_encoder=preview_encoder,
        workspace=workspace,
        clock=clock,
        budget_seconds=settings.provided.glm_budget_seconds,
        max_pages=settings.provided.max_pages,
    )
    parse_slots = providers.Singleton(
        ParseSlots,
        size=settings.provided.parse_slots,
        wait_seconds=settings.provided.slot_wait_seconds,
    )

    fingerprint = providers.Singleton(reading_fingerprint)


def build_container(env: Mapping[str, str] | None = None) -> Container:
    """A container configured from the environment — the only place `os.environ` is read for it."""
    container = Container()
    container.config.from_dict(Settings.from_env(env).as_dict())
    return container
