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

You can also **read your own document**: a PDF of up to five pages, or a single image (PNG, JPEG,
TIFF, WebP), shown in the same layout. **GLM-OCR** is the default — it reads handwriting and tables
best, but it runs on a scale-to-zero GPU, so the first upload after idle can take several minutes
while the model boots. **LiteParse + Tesseract** is the fast option: a couple of seconds per page on
the server's CPU, and visibly weaker on handwriting. Nothing is stored.

The bundled document is
[`corpus/document.pdf`](corpus/document.pdf) — six pages from a demo corpus: one printed page and five
handwritten flight/field forms, read ahead of time by all eight systems.

## Why it exists

The original question was "why does our ingestion pipeline publish garbage for this PDF?" The answer is
in `corpus/document.pdf` itself: its `creator` is `ocrmypdf 13.4.3 / Tesseract OCR-PDF v5.2.0`, so the
"text layer" is a 2023 Tesseract pass over the scan. Anything that trusts a text layer inherits that
pass. Hence this shootout: what do the parsers that actually look at the pixels produce?

## The eight systems

| system | pipeline | runs where |
|---|---|---|
| LiteParse + Tesseract | raster + bundled Tesseract | CPU |
| PaddleOCR PP-StructureV3 | layout → OCR → tables → order, mobile models | CPU |
| PP-StructureV3 (GPU) | server models, tables, chart branch | Modal L4 |
| MinerU 2.5 (VLM) | VLM backend, page by page | Modal L4 |
| GLM-OCR | 0.9B VLM | Modal L4 (vLLM) |
| Chandra OCR 2 | datalab VLM | Modal L4 |
| Unlimited-OCR | tiled VLM | Modal L4 |
| Nemotron 3.5 Parse | NVIDIA VLM + the repo's own postprocessing | Modal L4 |

Each was run the way its project intends: native CLI or API, default prompts from the model cards, and
`records/<id>.json` holds the raw output verbatim.

## Run it

```bash
make pages          # corpus/document.pdf -> /tmp/ocr-playground/pages (200 dpi) + assets/pages (110 dpi)
make site           # records/ -> evidence.json -> index.html
make serve          # http://127.0.0.1:8080
```

`make pages` needs poppler (`pdftoppm`). `make site` needs nothing but Python 3.11+.

The upload service is a separate process; run it beside the page. GLM-OCR is the default and needs an
OpenAI-compatible endpoint; without one the server stays up but only the fast path is available:

```bash
GLM_OCR_BASE_URL=https://<your-modal-app>.modal.run/v1 python api/server.py --port 8090
python api/server.py --port 8090        # no endpoint: fast (LiteParse + Tesseract) only
```

`GET /api/health` reports which modes this server can run. `POST /api/parse` takes the file as the
body and `X-Mode: glm|fast` (or `?mode=fast`); with neither, it uses GLM-OCR. The GLM path is the
`modal deploy modal/serve_glm_ocr.py` app below — cold after idle, so its first read can take
minutes; that is the trade the page's method picker is there to explain.

In production the service runs as its own container behind the same Traefik router
(`deploy/README.md`), which is why the page calls `/api/parse` on its own origin.

### Index a corpus with CocoIndex

The upload API reads one document per request. The same pipeline (`pipeline.py`) is also a CocoIndex
app, so a directory can be read incrementally — a file that did not change is not read again, and
changing a file, the mode or the model re-reads only what that change affects.

```bash
python3 -m venv .venv-index
.venv-index/bin/pip install -r requirements-cocoindex.txt
.venv-index/bin/cocoindex update main.py            # corpus/ -> out/, default mode

OCR_CORPUS=scans OCR_MODE=fast .venv-index/bin/cocoindex update main.py
```

It declares `out/documents/<path>.json` (per-page text and counts, plus the digest it was read
from), `out/pages.jsonl` (one row per page) and `out/summary.json` (what ran, what failed, what was
skipped). Reads are memoised in `cocoindex.db` on content digest + mode + engine identity + page
limit; delete it to force a full re-read. `OCR_MODE` defaults to `glm` when `GLM_OCR_BASE_URL` is set
and `fast` otherwise; `OCR_MAX_PAGES` defaults to every page (the upload API caps at 5), and
`OCR_CORPUS` / `OCR_OUT` / `OCR_DB` name the rest. The fast path needs `lit`; GLM needs the endpoint
above. The indexer is optional — the page and the API stay stdlib-only, and
`requirements-cocoindex.txt` pins the one extra dependency.

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
`GLM_OCR_BASE_URL` in its environment for the default path; without it, GLM-OCR shows as unavailable
and the page falls back to LiteParse + Tesseract. `deploy/README.md` has the run command.

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

## Layout

```
corpus/document.pdf      the six-page scan (source of truth)
prepare.py               poppler -> /tmp/ocr-playground/pages (200 dpi) + assets/pages (110 dpi)
runners/                 one file per CPU parser; _common.py defines the record shape
modal/                   one app per GPU parser, all scale-to-zero
records/<id>.json        {"page-1": "text…", …} — raw output, never hand-edited
systems.json             the roster the page renders (name, pipeline, device, how to run)
api/server.py            the upload service: GLM-OCR by default, LiteParse + Tesseract for speed
pipeline.py              the reading both the API and the indexer call (render + GLM-OCR / LiteParse)
main.py                  CocoIndex entry point: index a corpus into out/ incrementally
indexer/flow.py          walk OCR_CORPUS, memoise a read per digest+mode+identity, declare out/
requirements-cocoindex.txt  the indexer's one optional dependency, pinned
collect.py               records + roster -> evidence.json
build.py                 template.html + evidence.json -> index.html
template.html            the page: source rail, eight panes, text/preview + a find bar per pane
```

## Further reading

[`WRITEUP.md`](WRITEUP.md) is the longer story: what the artifacts are, how the build is wired, what
the runs showed, and the trade-offs behind the decisions. This README stays the setup guide.

## Credits

Parsers belong to their projects: [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR),
[MinerU](https://github.com/opendatalab/MinerU), [Chandra](https://github.com/datalab-to/chandra),
[Unlimited-OCR](https://github.com/baidu/Unlimited-OCR), NVIDIA Nemotron-3.5-Parse,
[LiteParse](https://github.com/datalab-to/liteparse). Their licences apply to their output.
