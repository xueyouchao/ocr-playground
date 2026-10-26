# OCR playground — a write-up

Eight document parsers, one scanned form, side by side. This is the longer story behind the page:
what the artifacts actually are, how the pipeline is wired, what the runs showed, and why a few
obvious next steps were deliberately not taken.

- Live: <https://ocr-playground.srv1567269.hstgr.cloud>
- Code: <https://github.com/xueyouchao/ocr-playground>
- Setup and commands: [`README.md`](README.md)
- Status: prototype, October 2026 — 8 systems × 6 pages, all 48 cells populated

---

## 1. The question it started from

The project began as an incident, not a benchmark: *why does our ingestion pipeline publish garbage
for this PDF?*

The answer was in the file's own metadata. `corpus/document.pdf` was assembled with
`ocrmypdf 13.4.3`, whose text layer is a `Tesseract OCR-PDF v5.2.0` pass over the scan (created
2023-10-18, modified 2023-11-13). So the document ships a *reading* already, and everything that
trusts the text layer inherits that 2023 reading — including its mistakes, in a form that looks like
clean, extractable text.

That reframes the comparison. The interesting question is not "which parser is best" but **what do
systems that actually look at the pixels produce, on a document whose text layer is a prior OCR
pass?** The corpus makes that question concrete: six pages, one printed and five handwritten
flight/field forms, heavy on tables and column abbreviations.

## 2. What the artifact is

A single static page. Left: the scanned page. Right: eight panes, one per parser, each showing either
the exact text that parser emitted or a rendered preview of it. Each pane carries its own find bar:
matching text is marked and counted in both views, and the query survives a page switch so the same
field can be looked for in every parser's reading. Above: six page chips, each annotated with how many
parsers produced text for that page. Below the fold: counts per pane, and provenance.

It supports two document sources through one renderer:

| source | parsed | parsers | shape |
| --- | --- | --- | --- |
| the bundled sample | ahead of time | all eight | `evidence.json` |
| your upload | live, per request | GLM-OCR (default) or LiteParse + Tesseract | API response |

The page has exactly one branch for this — `currentDoc()` returns `state.uploaded || DATA` — because
the upload service returns the **same evidence shape** the static build produces. That is the central
design decision of the whole project, and section 3 is about why it pays off.

## 3. The data contract

Everything hangs off one small, boring shape.

`records/<id>.json` is the raw output of one parser, verbatim, never hand-edited:

```json
{ "page-1": "…markdown or HTML…", "page-2": "…", "page-3": "…" }
```

`evidence.json` is the join of those records against `systems.json`:

```json
{
  "document": { "name": "efda8b60…a8a9.pdf", "pages": 6, "note": "…" },
  "systems":  [ { "id": "glm-ocr", "name": "GLM-OCR", "kind": "vlm",
                  "pipeline": "…", "device": "GPU (Modal L4)", "needs": "…" } ],
  "pages":    [ { "page": 1, "cells": { "glm-ocr": { "text": "…", "chars": 1234,
                                                      "tokens": 189, "numeric_share": 0.19 } } } ]
}
```

The API returns the same three keys. `build.py` inlines `evidence.json` into `template.html` and
writes `index.html`; the placeholder substitution also rewrites `<` to `\u003c`, because parser
output is untrusted and a stray `</script>` inside the inlined JSON would otherwise close the tag it
lives in.

Consequences worth stating plainly:

- **The renderer has no idea where a document came from.** Text/preview toggles, counts, page strip,
  footer — all identical for a bundled record and a live upload. Adding a parser to the *static* page
  is a record plus a roster entry; nothing in the view changes.
- **The records are the audit trail.** `collect.py` only measures; it never rewrites text. If a
  number looks wrong, the file it came from is one `cat` away, in the exact form the parser emitted.
- **There is no runtime database or fetch.** The published page is one HTML file with its data in it,
  so it works from `file://`, from a static host, and behind the API without CORS.

## 4. How a page gets built

```
corpus/document.pdf
      │  prepare.py   (poppler pdftoppm)
      ├─→ /tmp/ocr-playground/pages/page-N.png     200 dpi, what the parsers read
      └─→ assets/pages/page-N.jpg                  110 dpi q78, what the page shows
                │
      ┌─────────┴──────────┐
      │ runners/*.py       │  CPU: LiteParse, PP-StructureV3
      │ modal/*.py         │  GPU: PP-StructureV3 server models, MinerU, 4 VLMs
      └─────────┬──────────┘
                ↓
        records/<id>.json      raw output, one key per page
                │  collect.py  (+ systems.json, counts)
                ↓
          evidence.json
                │  build.py   (inline into template.html)
                ↓
           index.html          static site, ~310 KB, no runtime dependencies
```

Two details that matter in practice:

- **Two render densities, on purpose.** Parsers read a 200 dpi PNG; the rail shows a 110 dpi JPEG
  (1.2 MB for all six). Rendering the page twice is cheaper than shipping 200 dpi images to every
  visitor, and it keeps the comparison honest: the preview is a legible reference, not the exact
  bytes fed to the models.
- **`records/` is the stable seam.** Every runner — CPU, Modal, or any OpenAI-compatible endpoint —
  is just a program that writes that one JSON shape. Re-running one parser is
  `python runners/… && make site`. Nothing else in the pipeline needs to know which parser ran.

## 5. The eight systems

Each was run the way its own project intends: native CLI or API, the model card's own prompt,
temperature 0 for the VLMs, one page per call.

| system | approach | runs on |
| --- | --- | --- |
| LiteParse + Tesseract | PDFium raster + bundled Tesseract, layout by grid projection | CPU ×4 |
| PaddleOCR PP-StructureV3 | mobile layout/OCR/table models, reading order | CPU ×4 |
| PP-StructureV3 (GPU) | PP-DocLayout-L + PP-OCRv5 server + table/formula/chart branches | Modal L4 |
| MinerU 2.5 (VLM) | MinerU2.5-1.2B + layout/table models | Modal L4 |
| GLM-OCR | 0.9B VLM, 128K context | Modal L4 (vLLM) |
| Chandra OCR 2 | datalab VLM, layout-rich HTML out | Modal L4 |
| Unlimited-OCR | tiled 640 px inference for long pages | Modal L4 |
| Nemotron 3.5 Parse | NVIDIA VLM + the model repo's own postprocessing | Modal L4 |

The GPU apps all scale to zero and cache models in Modal volumes; the VLM runner
(`runners/vlm_endpoint.py`) speaks plain OpenAI-compatible chat completions, so a hosted API
substitutes for a deployment without code changes. Cold Modal endpoints answer with a `303` carrying
an attempt token; following that `Location` *is* the cold-start wait, so the runner does it by hand
rather than taking a Modal SDK dependency.

## 6. The upload path

The sample is parsed ahead of time; an upload is parsed live. Those are different jobs, so they are
different processes:

- **The page** is static files — no server needed, and its images never change, so they cache for a
  year.
- **The API** is a small Python service wired with a dependency-injection container. It renders
  uploads with poppler, then parses them through one of two paths, in its own container behind the
  same Traefik router at `/api`.

| mode | parser | where | character |
| --- | --- | --- | --- |
| `glm` (default) | GLM-OCR, a 0.9B VLM | Modal L4, scale-to-zero | best on handwriting and tables; cold start of minutes after idle |
| `fast` | LiteParse + Tesseract | this server's CPU | ~2 s/page; visibly weaker on the same pages |

A warm GLM page took about six seconds in testing, the fast path about two, and the first request
after idle paid roughly ninety seconds before vLLM answered. The budget is deliberately generous: the
GPU may need minutes, and a request that gives up early is worse than one that explains itself while
it waits. The page keeps a running status line with exactly that explanation, and offers the fast
path beside it.

Contract: `POST /api/parse` with the file as the request body, a `Content-Type`, an `X-Filename`
header (percent-encoded — header values are Latin-1, so `扫描.png` cannot ride in one verbatim), and
`X-Mode: glm|fast` (or `?mode=fast`). With no mode it is GLM-OCR. The response is the evidence shape
above, with one system. `GET /api/health` reports which modes this server can actually run and the
limits; the page reads it to disable GLM-OCR when the deployment has no endpoint, and to select the
fast path rather than fail the first upload.

**The same reading, indexed.** Reading is a use case (`application/read_document.py`); the HTTP
handler is a shell around it and the indexer is a second caller.
`interfaces/indexer/flow.py` is a CocoIndex app: walk a directory, memoise each document on its
content digest, mode, engine identity, page limit and a fingerprint of the reading code, and declare
`out/documents/<path>.json`, `out/pages.jsonl` and `out/summary.json`. A second run with nothing changed reads nothing; editing one
file, switching modes or changing the model re-reads exactly the documents that change affects. Across
the six-page sample plus a PNG, the fast path read seven pages in 21.6 s, and a switch to GLM re-read
one page after its 156.8 s cold start, producing the same 1,234 characters the upload path shows.

**Why not all eight live?** The other GPU-class models need their own deployments and multi-minute
cold starts, and the CPU PaddleOCR pipeline is too heavy to answer inside a request on a four-core
box. Two paths cover the range the page is making a point about — a VLM that reads handwriting, and a
classic OCR pass that is fast — without turning a demo into a queue. The upload pane is therefore a
single pane either way.

The security model is "bounds, and nothing on disk":

| bound | value | why |
| --- | --- | --- |
| request body | 20 MiB | rejects before reading |
| pages | 5 | poppler render time is linear |
| concurrent parses | 2 | the fast path is CPU-bound; a GLM upload holds its slot while it waits |
| socket timeout | 30 s | a silent client cannot own a thread |
| fast parse timeout | 120 s | per subprocess |
| GLM budget | 900 s | cold start plus five pages; the request is held, not dropped |
| storage | none | temp dir deleted before the response is sent |

The subtle one: the parse slot is acquired **before** the body is read. Otherwise an upload that
declares a length and then goes quiet holds a thread for free; with the slot first, stalled uploads
queue behind the same two-slot bound as real work and time out into a `408`.

Another: PNG and JPEG uploads are parsed **as uploaded**. The first version round-tripped every
upload through ImageMagick, which rewrote an 8-bit RGB scan as an 8-bit grayscale palette image and
cost more than half the text — 111 characters instead of 407 on the sample form. Only TIFF and WebP,
which LiteParse cannot open, take the conversion path.

A third, specific to the VLM path: the image data URL carries the real media type, sniffed from the
first bytes rather than the filename, because an upload keeps its own bytes and a mislabelled JPEG is
a `400` from the endpoint rather than a reading.

## 7. What the runs showed

Four findings, in the order they hurt.

**1. A text layer is not evidence of a reading.** This file's layer is a prior OCR pass, so the
pipeline that trusted it published wrong text for five of six pages. The comparison starts by
discarding the layer entirely and handing every parser pixels.

**2. Routing needs a content-quality signal, not a damage signal.** The ingestion acceptance rule
measured Unicode damage — replacement characters, control codepoints, encoding artifacts. Ordinary
letters from a bad OCR pass score 0.0 damage. A page can be confidently, cleanly wrong.

**3. Page-level, not document-level.** MinerU's VLM backend silently skipped page 3 in a
whole-document run and read it fine alone; the runner now splits one-page PDFs and retries, with a
pipeline-backend fallback per page. PP-StructureV3's server layout model classifies page 3 as a
chart and emits a placeholder with no text unless the chart branch is on. Both needed per-page
attention to produce anything at all — a document-level pass hides exactly this failure.

**4. Structure without content looks like success.** On page 3, PP-StructureV3 (GPU) emits a tidy
two-column table — and its values are wrong: `FLT ID | 95.0626`, `PressURE | 01(2, 2/0)3.5`. The
format is plausible; the reading is not. This is the most dangerous output class in the whole
shootout, because a screenshot of it looks parsed.

The counts make the same point from a different angle. On the table-heavy page 5, the
HTML-emitting parsers produce 85–99% markup; Chandra's 21,033 characters there are 262 characters of
actual text and a grid of empty cells. **A character count on a table page measures markup, not
reading.** That is why counts are shown and never ranked.

## 8. Why there is no accuracy score

An earlier version scored every pane with a dictionary share: how many of its alphabetic tokens a
73k-entry word list recognised. It was removed because it measured the wrong thing.

Page 4 is roughly half to four-fifths numbers depending on the parser, and its remaining words are
column abbreviations — `CLD` (×14), `ABV`, `BTN`, `BLW`, `TDME`, `TRK`, `WD`, `WS`, `SP`, `NOAA`.
A *correct* reading scored 30%. Across the six pages the denominator ranged from 2 to 187 words, so
the number was meaningful on page 1 and noise everywhere else. A score that wrong is worse than no
score.

The honest path is ground truth: hand-transcribe one page, diff it against every parser (CER/WER),
then rank. That has not been done. Until it is, the comparison is visual, and the counts are
descriptive statistics — not a leaderboard.

## 9. Decisions and trade-offs

- **Verbatim records over normalised text.** Normalising would make parsers comparable and destroy
  provenance. The preview layer is where messy output gets rendered, not rewritten.
- **Inlined JSON over a fetched `evidence.json`.** One file to publish, works offline, no CORS. Cost:
  a ~310 KB page and a rebuild whenever a record changes.
- **Template + 10-line build script over a framework.** The page is 504 lines of HTML/CSS/JS with no
  dependencies. Cost: a hand-rolled markdown converter and a preview sandbox, both small and
  inspectable.
- **Parser output is untrusted.** HTML renders only inside a sandboxed iframe (`sandbox=""`, own
  styles); everything else goes through a converter that escapes before it formats. `srcdoc` is a
  double-quoted attribute, so quotes are escaped too. Extracted figures point at files that were never
  shipped, so `<img>` becomes an "image not shipped" chip instead of a broken-image icon.
- **GLM-OCR by default, with its cost on the label.** GLM-OCR reads this corpus's handwriting far
  better than Tesseract, so it is the default path — but it is a scale-to-zero GPU, and the first read
  after idle waits minutes. Rather than hide that behind a spinner or silently fall back, the panel
  names both paths, states the trade, and lets the picker choose fast. A silent fallback would hand
  back a reading while claiming a model that never ran.
- **CocoIndex for the batch path, not the request path.** Indexing a directory wants memoisation,
  retraction of documents that disappeared, and a run that can be restarted; an upload wants an answer
  inside one request. Both call the same use case, so the index cannot quietly read a document with a
  different parser than the demo shows.
- **Domain/application/infrastructure, wired by a container.** The reading — document, page, reading
  stats, engine identity, the errors a bad request deserves — is a domain that imports nothing
  outside itself; the use case talks to ports, and poppler, ImageMagick, LiteParse and GLM-OCR are
  adapters behind them. `composition/container.py` is the only file that knows the whole graph, so a
  test replaces a port and the HTTP handler still runs for real. Cost: the service now needs a
  compiled package (`dependency-injector`), where before it needed nothing but the stdlib. The static
  build does not: `collect.py`, `build.py` and `prepare.py` never import the container, so the page
  still builds on a bare Python — that boundary is now an invariant, not an accident.
- **Mixed-format cells are handled as mixtures.** Nemotron emits `##` headings around raw `<table>`
  blocks; Chandra emits whole HTML documents. Picking one format per cell mangles the other, so
  block-level HTML is preserved and the Markdown between blocks is converted.
- **The static mirror was deleted.** GitHub Pages has no server behind `/api`, so the upload panel
  could only apologise there. One deployable is better than two with different capabilities.

## 10. Limits, and what would come next

- **No ground truth, therefore no accuracy.** The single highest-value next step: transcribe one page
  by hand, add a per-parser diff, and let the ranking be earned rather than implied.
- **The upload API is public and unauthenticated, and so is the GPU endpoint behind it.** The caps
  are the whole defence on this side; the Modal app accepts anyone who knows its URL. Rate limiting,
  and a proxy-auth or key on the endpoint, are prerequisites before this points at anything that
  matters.
- **Uploads are not reproducible.** Nothing is stored — a privacy choice that also means a user's
  result cannot be audited or shared by link later.
- **Handwriting is still hard, and the failures are structural.** The interesting errors here are
  dropped pages, misclassified layouts, and confident empty tables, not misread characters.
- **The roster is data.** Adding a ninth system is a record file and a `systems.json` entry; the page
  and the pipeline do not change.

## 11. Repository map

```
corpus/document.pdf      the six-page scan (source of truth)
prepare.py               poppler → 200 dpi pages + 110 dpi previews
runners/                 one file per CPU parser; _common.py defines the record shape
modal/                   one app per GPU parser, all scale-to-zero
records/<id>.json        {"page-1": "text…"} — raw output, never hand-edited
systems.json             the roster the page renders
collect.py               records + roster → evidence.json
build.py                 template.html + evidence.json → index.html
template.html            the page: rail, page strip, eight panes with find bars, upload panel
ocr_playground/domain/    documents, readings, engine specs, ports, domain errors
ocr_playground/application/  the read-document use case; the two-slot gate
ocr_playground/infrastructure/  poppler, ImageMagick, LiteParse, GLM-OCR, settings, fingerprint
ocr_playground/interfaces/   http/, indexer/, site/, evidence.py
ocr_playground/composition/  the dependency-injector container
api/server.py            entry point -> interfaces/http (compatibility shim)
main.py                  entry point -> interfaces/indexer (CocoIndex needs a file here)
collect.py build.py prepare.py  entry points -> interfaces/site (compatibility shims)
docs/architecture.md     the design note: layers, ports, why the container
tests/                   the offline suite; tests/fakes.py holds the doubles
requirements.txt         dependency-injector, pinned — the service only
requirements-cocoindex.txt  the indexer's extra dependency, pinned
deploy/                  Traefik/nginx notes for this box and any plain host
```

```bash
make pages    # PDF → /tmp/ocr-playground/pages + assets/pages
make site     # records/ → evidence.json → index.html
make serve    # http://127.0.0.1:8080
make api      # upload service on :8090
make index    # read OCR_CORPUS into out/ incrementally (cocoindex)
```

The full per-parser commands — including the Modal deploys and the endpoint runner — are in the
[`README`](README.md), because they are a moving target and this document should not drift with them.

## 12. Credit where it is due

The parsers belong to their projects: [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR),
[MinerU](https://github.com/opendatalab/MinerU),
[Chandra](https://github.com/datalab-to/chandra),
[Unlimited-OCR](https://github.com/baidu/Unlimited-OCR), NVIDIA Nemotron-3.5-Parse, and
[LiteParse](https://github.com/datalab-to/liteparse). Their licences apply to their output. This
repository's contribution is the harness: one corpus, eight runs, one shape, and the honesty to show
the numbers without inventing a score.
