# Architecture

The reading is the product. Everything else — HTTP, CocoIndex, poppler, the static build — is a way
in or a way out, and the structure says so.

```
ocr_playground/
  domain/            what a document, a page reading and an engine are; what a bad request means
  application/       use cases: read this document, hold a parse slot
  infrastructure/    adapters: poppler, ImageMagick, LiteParse, the OpenAI-compatible GPU
                     endpoints, settings, temp dirs
  interfaces/        ways in: HTTP, the CocoIndex indexer, the static site build
  composition/       the dependency-injection container that wires the four together
```

The dependency rule points inward. `domain` imports only itself. `application` imports `domain` and
nothing concrete: it talks to ports (Protocols in `domain/ports.py`). `infrastructure` implements
those ports and may import `application`/`domain` for the errors it raises. `interfaces` imports
everything below it, and `composition` imports everything — it is the only package allowed to know
the whole graph.

## The ports

| port | what the application asks for | production adapter |
| --- | --- | --- |
| `PageRenderer` | bytes → numbered page images, honouring a page limit | poppler + ImageMagick |
| `PageOcr` | one page → text, plus availability and an identity string | LiteParse + Tesseract; every OpenAI-compatible GPU endpoint (GLM-OCR, Chandra, Nemotron, Unlimited-OCR, MinerU, PP-StructureV3) |
| `PreviewEncoder` | a page image → a data URL | ImageMagick |
| `Workspace` | a scratch directory, deleted on exit | `tempfile.TemporaryDirectory` |
| `Clock` | monotonic seconds, for the document budget | `time.monotonic` |

A test replaces exactly those five things; nothing else knows the difference.

## Why the injector

`composition/container.py` is a `dependency_injector` `DeclarativeContainer`:

- `config` is a `Configuration` provider. `build_container()` seeds it from the environment, a test
  seeds it with `from_dict`, and `Settings.from_mapping` turns either into one frozen dataclass — so
  defaults exist once and overrides are a data change, not a patch of a module global.
- Everything is a `Singleton`. The engines, renderer and use case are stateless; rebuilding a
  `ProcessRunner` per request would add noise and no isolation.
- Nothing resolves at import. Importing the container contacts no endpoint and runs no tool; the first
  request, or the indexer's first document, is what pulls the graph into existence.
- A test replaces a node (`container.read_document.override(fake)`) instead of monkeypatching a
  module. That is what makes the HTTP tests run the real handler against a fake reading.

The trade is honest: the service now depends on a compiled package (`dependency-injector==4.49.1`)
where it used to be stdlib-only. The static build does not: `scripts/collect.py`, `scripts/build.py` and `scripts/prepare.py`
reach `domain` and `interfaces/site` and never touch the container, so `make site` still runs on a
bare Python. If the dependency becomes unwelcome, the container is one file and every constructor is
already explicit — deleting it means writing that file by hand, not untangling imports.

## Two ways in, one use case

```
POST /api/parse ──▶ interfaces/http ──▶ ReadDocumentService ──▶ PageRenderer / PageOcr / PreviewEncoder
cocoindex update ─▶ interfaces/indexer ─┘
```

The HTTP handler owns transport policy (status codes, CORS, body cap, percent-decoded filename) and
turns application errors into responses. The indexer owns its own policy (every page by default,
no previews, memo keys). Both call the same `ReadDocumentService`, so the live pane and the index
cannot drift into two different readers — the original reason the reading was extracted in the first
place.

The aggregate leaves the use case as domain objects. `interfaces/evidence.py` renders those into the
one JSON shape (`document` / `systems` / `pages[].cells`) that the page, the upload response and the
indexer all use. The domain never learns the word "cells".

## Tests

`pytest` runs offline: no GPU, no LiteParse, no poppler. The shape is:

- `tests/test_domain.py`, `tests/test_read_document_service.py` — vocabulary and orchestration, with
  fakes for the ports.
- `tests/test_rendering.py`, `tests/test_glm_ocr.py`, `tests/test_liteparse_engine.py` — adapters,
  with a fake process runner and a scripted local endpoint.
- `tests/test_http.py`, `tests/test_site_build.py`, `tests/test_indexer.py` — the interface contracts.
- `tests/test_container.py` — the wiring itself: configuration reaches the use case, overrides work.
- `tests/test_indexer_run.py` — a real incremental CocoIndex update, skipped when the optional
  toolchain is absent.

`tests/fakes.py` holds the doubles; `make test` runs the suite.
