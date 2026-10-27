# ocr-playground

Eight document parsers, one scanned form, side by side — a throwaway comparison turned into something
you can run again.

Open `index.html` and you get the scanned page on the left and every parser's output on the right, with
a **text** / **preview** toggle per pane: `text` is exactly what the parser emitted, `preview` renders it
(HTML in a sandboxed iframe with scripts blocked, markdown through a small converter).

Every pane also has its own **find bar**. Type a word or phrase and every occurrence is marked in that
pane — in `text` and in `preview` alike — with a match count and `‹` / `›` buttons (or Enter /
Shift+Enter) to step between hits. A query stays with its pane when you change pages, so the same field
can be looked up in every parser's reading of every page. Matches are literal: punctuation is not a
pattern, and a run of spaces in the query matches any whitespace, so a phrase still works across a line
break. In `preview` the rendered text is searched, so a phrase also crosses a `<br>`, a block edge, or a
table cell.

A **night theme** is a click away on the title line: it swaps the palette, remembers the choice, and
follows the OS (which is what makes the page go dark at sunset) until you pick one yourself. The
preview panes are separate documents, so they are re-rendered in the same palette — no white rectangles
left glowing in the middle of a dark page, and find marks keep a readable text colour at night.

You can also **read your own document**: a PDF of up to five pages, or a single image (PNG, JPEG,
TIFF, WebP), shown in the same layout. **Chandra OCR 2** is the default — it returns layout-rich
markdown with tables, at about a minute a page, and it runs on a scale-to-zero GPU, so the first
upload after idle also waits for boot. **GLM-OCR** is the faster warm path (about five seconds a page)
and reads handwriting best. **LiteParse + Tesseract** is the fast option: a couple of seconds per page
on the server's CPU, and visibly weaker on handwriting. The picker also lists every other GPU parser
this deployment has an endpoint for — GLM, Nemotron, Unlimited-OCR, MinerU and
**PP-StructureV3 (GPU)** — and several can be ticked to read the same pages side by side. Nothing is
stored.

The bundled document is
[`corpus/document.pdf`](corpus/document.pdf) — six pages from a demo corpus: one printed page and five
handwritten flight/field forms, read ahead of time by all eight systems. Where the scan came from, and
what that means for redistributing it, is in [`THIRD_PARTY.md`](THIRD_PARTY.md#the-bundled-sample-document).

## Why it exists

The original question was "why does our ingestion pipeline publish garbage for this PDF?" The answer is
in `corpus/document.pdf` itself: its `creator` is `ocrmypdf 13.4.3 / Tesseract OCR-PDF v5.2.0`, so the
"text layer" is a 2023 Tesseract pass over the scan. Anything that trusts a text layer inherits that
pass. Hence this shootout: what do the parsers that actually look at the pixels produce?

## The eight systems

| system | pipeline | runs where |
|---|---|---|
| Chandra OCR 2 (default) | datalab VLM | Modal L4 |
| Nemotron 3.5 Parse | NVIDIA VLM + the repo's own postprocessing | Modal L4 |
| GLM-OCR | 0.9B VLM | Modal L4 (vLLM) |
| Unlimited-OCR | tiled VLM | Modal L4 |
| PP-StructureV3 (GPU) | server models, tables, chart branch | Modal L4 |
| MinerU 2.5 (VLM) | VLM backend, page by page | Modal L4 |
| LiteParse + Tesseract | raster + bundled Tesseract | CPU |
| PaddleOCR PP-StructureV3 | layout → OCR → tables → order, mobile models | CPU |

Each was run the way its project intends: native CLI or API, default prompts from the model cards, and
`records/<id>.json` holds the raw output verbatim.

Every card carries two chips. The first says where it runs — `native` on this host's CPU, or `remote`
on a Modal L4 — and the second says how it reads a page: `vlm` (one vision-language model, one pass,
markdown out), `staged` (separate layout, OCR and table models in sequence, which is PP-StructureV3 at
either size), or `hybrid` (MinerU: a VLM with a staged pipeline behind it as the fallback). The two are
independent — PP-StructureV3 is `staged` on CPU and `staged` on GPU alike.

## Run it

```bash
make pages          # corpus/document.pdf -> /tmp/ocr-playground/pages (200 dpi) + assets/pages (110 dpi)
make site           # records/ -> evidence.json -> index.html
make serve          # http://127.0.0.1:8080
```

`make pages` needs poppler (`pdftoppm`). `make site` needs nothing but Python 3.11+ — the build is
stdlib-only and does not import the service's container.

The upload service is a separate process, wired with
[python-dependency-injector](https://github.com/ets-labs/python-dependency-injector). Create the venv
once, then run it beside the page. Each VLM parser needs its own OpenAI-compatible endpoint; the
ones without an endpoint simply show as unavailable in the page's dropdown:

```bash
make venv                                     # dependency-injector + pytest
GLM_OCR_BASE_URL=https://<glm-app>.modal.run/v1 \
CHANDRA_OCR_BASE_URL=https://<chandra-app>.modal.run/v1 \
NEMOTRON_PARSE_BASE_URL=https://<nemotron-app>.modal.run/v1 \
UNLIMITED_OCR_BASE_URL=https://<unlimited-app>.modal.run/v1 \
MINERU_BASE_URL=https://<mineru-app>.modal.run/v1 \
PADDLE_OCR_BASE_URL=https://<paddle-app>.modal.run/v1 \
make api
make api                                      # no endpoint: fast (LiteParse + Tesseract) only
```

The six GPU apps are `modal deploy modal/serve_{glm_ocr,chandra,nemotron,unlimited,mineru,paddle}.py`,
one GPU each, all scaled to zero when idle. Five run on an L4; PP-StructureV3 runs on an A10G, also
24 GB, because the L4 pool was out of capacity when its wrapper was deployed. The VLM wrappers share
one HF cache volume, so the first boot of each pays for its own weights once; MinerU keeps its models
in its own volume, and PP-StructureV3 in PaddleX's.

`make test` runs the offline suite (no GPU, no LiteParse, no poppler). The fast path needs LiteParse
(`pip install liteparse==2.17.0`, which provides the `lit` CLI) and Tesseract on the host or in the
image.

`GET /api/health` reports which modes this server can run — `glm`, `fast`, `chandra`, `nemotron`,
`unlimited`, `mineru`, `paddle`, each with its name, availability and cold-start note, which is what
the page's picker is built from. `POST /api/parse` takes the file as the body and `X-Mode:
glm|fast|chandra|nemotron|unlimited|mineru|paddle` (or `?mode=`); with neither, it uses Chandra OCR 2.

A comma-separated list asks for a comparison: `X-Mode: chandra,nemotron` reads the same rendered
pages with both, concurrently, and the response carries one system and one cell per engine. An
engine that fails does not sink the comparison — its cells carry the reason and the others are
still rendered — and the request only fails if no engine produced a reading. Each selected parser
is its own endpoint, so a comparison costs the same GPU-minutes as running them one at a time and
takes the time of the slowest.

`POST /api/warm` starts that cold GPU before an upload needs it. The page sends it when somebody
picks GLM or reaches for the uploader, so the boot overlaps with the seconds they spend choosing a
file. A warm container bills GPU time from boot until `scaledown_window` after the last request, so
the server throttles it to one per `GLM_OCR_WARM_INTERVAL` (300 s) and at most `GLM_OCR_WARM_DAILY_CAP`
(40) per rolling day; `GLM_OCR_WARM=0` turns it off and `GLM_OCR_WARM_TIMEOUT` bounds the wait. It is
deliberately not on `GET /api/health`: warming every page view would rent a GPU for a trickle of
traffic.

In production the service runs as its own container behind the same Traefik router
(`deploy/README.md`), which is why the page calls `/api/parse` on its own origin.

### Index a corpus with CocoIndex

The upload API reads one document per request. That reading is a use case
(`ReadDocumentService`), and the indexer is a second way to call it, so a directory can be read
incrementally — a file that did not change is not read again, and changing a file, the mode or the
model re-reads only what that change affects.

```bash
python3 -m venv .venv-index
.venv-index/bin/pip install -r requirements-cocoindex.txt
.venv-index/bin/cocoindex update main.py            # corpus/ -> out/, default mode

OCR_CORPUS=scans OCR_MODE=fast .venv-index/bin/cocoindex update main.py
```

It declares `out/documents/<path>.json` (per-page text and counts, plus the digest it was read
from), `out/pages.jsonl` (one row per page) and `out/summary.json` (what ran, what failed, what was
skipped). Reads are memoised in `cocoindex.db` on content digest + mode + engine identity + page
limit; delete it to force a full re-read. `OCR_MODE` defaults to `chandra` when `CHANDRA_OCR_BASE_URL`
is set, then `glm` when `GLM_OCR_BASE_URL` is set, and `fast` otherwise; `OCR_MAX_PAGES` defaults to
every page (the upload API caps at 5), and `OCR_CORPUS` / `OCR_OUT` / `OCR_DB` name the rest. The fast
path needs `lit`; Chandra and GLM need their endpoints above. The indexer is optional and pinned in
`requirements-cocoindex.txt`, which now also installs
the service's one runtime dependency.

Re-running a parser writes its record; then `make site` picks it up:

```bash
# CPU parsers
LIT_CLI=/path/to/lit python runners/liteparse_system.py
/home/ubuntu/tmp/paddle-venv/bin/python runners/paddle_structure.py

# GPU parsers (Modal; each app scales to zero)
modal deploy modal/serve_glm_ocr.py        # then point the runner at its URL
python runners/vlm_endpoint.py --id glm-ocr --model glm-ocr \
    --base-url https://…/v1 --prompt "Text Recognition:"

modal run modal/paddle_structure_gpu.py --paths /tmp/ocr-playground/pages/page-1.png,…
modal run modal/mineru_gpu.py              # page by page, retries, pipeline fallback
```

`runners/vlm_endpoint.py` speaks any OpenAI-compatible endpoint, so a hosted API works as well as a
Modal deployment. Cold Modal endpoints answer with a `303`; the runner follows it and waits.

## Deploy it

It is a static site: `index.html`, `assets/pages/*.jpg`, nothing else is needed at runtime.

```bash
rsync -av --delete index.html assets user@host:/var/www/ocr-playground/
```

`deploy/nginx.conf` is a minimal server block with cache headers for the page images. Any static host
works — the page has no build step, no frameworks, no external requests.

The upload service is a second container. It keeps the CPU toolchain for the fast path, and needs
`CHANDRA_OCR_BASE_URL` in its environment for the default path; without it, Chandra OCR 2 shows as
unavailable and the page falls back to the first configured parser (GLM-OCR if present, otherwise
LiteParse + Tesseract). `deploy/README.md` has the run command.

## What the numbers are, and are not

The page shows **counts only**: characters, tokens written, and the share that are numbers. There is
deliberately no accuracy score.

An earlier version scored every pane with a dictionary share (how many of its alphabetic tokens a
73k-entry word list knows). It was removed because it measured the wrong thing here: page 4 is 77%
numbers and its 50 remaining words are column abbreviations — `CLD` (×14), `ABV`, `BTN`, `BLW`, `TDME`,
`TRK`, `WD`, `WS`, `SP`, `NOAA` — so a correct reading scored 30%. Across the six pages the denominator
ranged from 2 to 187 words, which made the score meaningful on page 1 and noise everywhere else.

A real accuracy number needs ground truth: one page hand-transcribed and diffed per parser. That has
not been done. Until it is, the comparison is visual, and the character counts are not a ranking —
Chandra's 21,033 characters on page 5 are a table truncated at `max_tokens`, and MinerU's 13,630 are a
verbose pipeline fallback.

## What this prototype taught us

- **A text layer is not evidence of a reading.** This file's layer is a prior OCR pass; the pipeline that
  trusted it published wrong text for five of six pages.
- **Routing needs a content-quality signal, not a damage signal.** The acceptance rule measured Unicode
  damage, and ordinary letters from a bad OCR pass score 0.0 damage.
- **Page-level, not document-level.** MinerU's VLM backend silently dropped page 3 from a whole-document
  run and read it fine alone; PP-StructureV3's server layout model classes that page as a chart and
  emits an empty grid unless the chart branch is on. Both needed per-page attention to produce anything.
- **Structure without content looks like success.** The empty-cell table is the most dangerous output
  here, because a screenshot of it looks parsed.

## How it is put together

The reading is organised as domain → application → infrastructure → interfaces, wired by a
dependency-injection container. The short version:

| layer | holds |
| --- | --- |
| `domain/` | `SourceDocument`, `ReadDocument`, `ReadingStats`, `EngineSpec`, the port Protocols, the errors a bad request deserves |
| `application/` | `ReadDocumentService` (render → read → preview, one budget) and the parse-slot gate |
| `infrastructure/` | poppler, ImageMagick, LiteParse, GLM-OCR, settings, temp dirs, the reading fingerprint |
| `interfaces/` | the HTTP handler, the CocoIndex flow, the static site build, the evidence mapping |
| `composition/` | `container.py`: configuration, singletons, and the only place every layer is known |

[`docs/architecture.md`](docs/architecture.md) explains the boundaries, the ports, why the container
is worth a compiled dependency, and how the suite replaces a port instead of patching a module.

## Layout

```
corpus/document.pdf      the six-page scan (source of truth)
runners/                 one file per CPU parser; _common.py defines the record shape
modal/                   one app per GPU parser, all scale-to-zero
records/<id>.json        {"page-1": "text…", …} — raw output, never hand-edited
systems.json             the roster the page renders (name, pipeline, device, how to run)
template.html            the page: source rail, eight panes, text/preview + a find bar per pane
ocr_playground/
  domain/                documents, readings, engines, ports, domain errors
  application/           the read-document use case and the two-slot gate
  infrastructure/        poppler, ImageMagick, LiteParse, GLM-OCR, settings, fingerprint
  interfaces/            http/, indexer/, site/, evidence.py
  composition/           the dependency-injector container
api/server.py            compatibility entry point -> interfaces/http
main.py                  CocoIndex entry point -> interfaces/indexer
collect.py build.py      compatibility entry points -> interfaces/site
prepare.py               compatibility entry point -> interfaces/site/prepare
docs/architecture.md     the design note behind the structure
tests/                   the offline suite; tests/fakes.py holds the doubles
requirements.txt         dependency-injector, pinned (the upload service only)
requirements-cocoindex.txt  the indexer's extra dependency, pinned
```

## Further reading

[`WRITEUP.md`](WRITEUP.md) is the post draft behind the demo: what it is, the two upload paths, and
what the cold-start caching saved. The code's own design — the ports, the container, the two ways in —
is in [`docs/architecture.md`](docs/architecture.md). The longer engineering write-up that the post
replaced is in git history (`git show 0a304e2:WRITEUP.md`). This README stays the setup guide.

## Credits

Parsers belong to their projects: [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR),
[MinerU](https://github.com/opendatalab/MinerU), [Chandra](https://github.com/datalab-to/chandra),
[GLM-OCR](https://huggingface.co/zai-org/GLM-OCR), [Unlimited-OCR](https://github.com/baidu/Unlimited-OCR),
NVIDIA Nemotron-3.5-Parse, [LiteParse](https://github.com/datalab-to/liteparse). Their licences apply
to their output. [`THIRD_PARTY.md`](THIRD_PARTY.md) says what is vendored, what is only invoked, and
which terms apply to which part.

## Licence

MIT — see [`LICENSE`](LICENSE), copyright 2026 xueyouchao.

One carve-out: `modal/nemotron_helper/` is vendored NVIDIA code under OpenMDW-1.1, with the SPDX
headers and the upstream revision left intact. The agreement is in
[`LICENSES/OpenMDW-1.1.txt`](LICENSES/OpenMDW-1.1.txt); everything outside that directory is MIT.
The bundled sample scan raises a separate provenance question, which [`THIRD_PARTY.md`](THIRD_PARTY.md)
records.
