# Spec: show each pane as it answers

Status: proposed, not implemented. Context: `docs/architecture.md`, and the 2026-10-27 incident where
three panes were ready and the page showed none of them.

## The problem

`POST /api/parse` with a comparison (`X-Mode: chandra,nemotron,paddleocr-gpu,mineru`) answers **once**,
after every engine has finished or burned its budget. The budget is `glm_budget_seconds = 900`, and the
per-request socket timeout is 900 s too, so one row that never answers costs the whole page a quarter of
an hour — even when a different row answered at second 30.

Measured on 2026-10-27 20:46: four engines, two of which never produced a container. Paddle answered
inside a minute. The page showed nothing for fifteen minutes. Because nothing came back, the upload was
retried four more times; each retry held another request per engine, and the uncapped apps grew a
container per request (two mineru containers, both booting for uploads the browser had abandoned).

Cancellation (`cancel`, in this same PR) stops the *work* once the client hangs up. It does nothing for
the client that is still there, watching a spinner, while three of four panes are ready.

## Goals

- Each pane renders the moment **its** engine answers. A row that never answers costs its own pane.
- The page keeps its current look and its `systems` / `pages` / `cells` vocabulary.
- No new server state to expire, and no change to the extraction engines or warming.

Non-goal: making cold starts shorter (see `deploy/modal/*.py`, `scaledown_window`, warming).

## Options

1. **Shorten the budget.** One line. But a cold boot legitimately takes 2–7 minutes; a short budget is
   exactly the change that produced blank panes before (the "did not answer: Broken pipe" incident).
2. **Job API.** `POST /api/parse` → `202 {"job": id}`, then `GET /api/jobs/{id}` until done. No proxy
   buffering risk, survives a page reload, and lets a second tab watch. Cost: server-side job state
   (in-memory dict, TTL, GC on a process that is restarted by every deploy), plus a polling loop in the
   page.
3. **Stream the response.** Keep one `POST`; answer `application/x-ndjson` and write events as they
   happen. No server state, results in order of arrival. Risk: an intermediary that buffers the
   response would defeat it entirely.
4. **Server-Sent Events.** A `GET` stream after a `POST` handshake. Same buffering risk, plus two phases
   and CORS pre-flight on the streaming endpoint.

**Recommendation: 3**, with 2 as the fallback if Traefik turns out to buffer. It is the smallest change
that fixes the symptom, and the page's render path barely moves.

## Wire contract

The request does not change. The response does, and only for clients that ask for it:

```http
POST /api/parse
Accept: application/x-ndjson          ← opt-in; without it the server answers today's JSON
X-Mode: chandra,nemotron,paddleocr-gpu,mineru
```

Body: one JSON object per line, in this order, for as long as the work runs.

```jsonc
{"type":"start","systems":[{"id":"chandra","name":"Chandra OCR 2","cold_start":true}, …],"pages":2}
{"type":"page","index":0,"preview":"data:image/webp;base64,…"}      // rendered, before any reading
{"type":"cell","page":0,"engine":"paddleocr-gpu","chars":1840,"text":"…"}   // as each engine answers
{"type":"cell","page":0,"engine":"chandra","error":"…did not finish within the time limit…"}
{"type":"done","note":"read by Chandra OCR 2 + …","elapsed":147.2}
```

Rules:

- **Status before the first byte.** Everything the handler can reject (unknown mode, over the size
  limit, busy slots) is answered *before* streaming starts, as today: a normal 4xx with a JSON body.
  Once the first event is written the status is `200` and failures become events.
- **One line, one event.** `\n`-terminated, `ensure_ascii=False`, no embedded newlines (JSON escapes
  them). `wfile.flush()` after each line.
- **Chunked encoding.** `Transfer-Encoding: chunked` on HTTP/1.1; the handler must not set
  `Content-Length`, and must close the connection at the end (`close_connection = True`) so a truncated
  stream is detectable.
- **Keep-alive ping.** `{"type":"ping"}` every 15 s of silence — a 900 s stream with a proxy in front
  needs traffic, and the page can show "still waiting on 2 engines" instead of freezing.
- **Ordering.** `start`, then per page: `page` (preview) → its `cell`s in whatever order engines finish
  → `done`. Events for page N+1 may not start before page N's cells are all emitted, so the page can
  keep the current "one page at a time" mental model. (Engines are already fanned out per page, not
  across pages.)
- **Cancellation.** The server side is already in place: `ClientGone` stops the fan-out. The client side
  is the `AbortController` the page now creates per upload (see *Page changes*), and a stream inherits
  it: `abort()` closes the socket, the handler's probe sees EOF, the stream ends with no `done` event,
  which is how the page tells a cancelled stream from a finished one.

## Server changes

- `interfaces/http/server.py`: after the pre-flight checks, if `Accept` contains
  `application/x-ndjson`, write the status and headers and hand the writer to the use case; otherwise
  today's path is untouched.
- `application/read_document.py`: `execute(..., sink=None)`. With a sink, `_read_page` emits each
  reading/failure as it resolves instead of only at the end, and the returned `ReadDocument` stays the
  same object — so `document_evidence` (and the indexer, and the CLI) keep working unchanged.
- `interfaces/evidence.py`: `document_evidence` gains a per-cell serializer so the stream and the JSON
  body cannot drift.

## Page changes

`site/template.html` (built into `site/index.html` by `make site`). The first item is already done:

- **The `AbortController` and its `signal` are a page change, not an existing feature.** Until this
  change the upload `fetch` had no `signal`, so nothing the page did could close the request; only the
  browser closing the tab did, and only when it got round to it. The page now creates one controller per
  upload, aborts the previous upload when a new one starts, and aborts on `pagehide`. This is what makes
  the server-side cancellation reachable at all — and it is what stops a retry from queueing behind the
  upload it replaced.
- Add `Accept: application/x-ndjson` to the upload `fetch`, then read `response.body.getReader()` and
  split on `\n`, feeding events into the same render path that consumes `body.systems` /
  `body.pages[*].cells` today.
- A pane with no `cell` yet renders the existing "waiting" state with the engine's `cold_start` note,
  which the page already knows how to show.
- On `error`/`done`, finalise; on stream abort, show the per-pane state that arrived.

## Risks

| Risk | Why it matters | Mitigation |
|---|---|---|
| Traefik buffers the response | The whole point is lost, silently — everything arrives at once | Verify first: `curl -N` against the container, then against `https://…/api/parse`, printing timestamps per line. If Traefik buffers, add `buffering` to the router (or `X-Accel-Buffering: no`), or fall back to option 2. |
| A proxy idle timeout cuts a long stream | Panes hang mid-page | 15 s `ping` events |
| Doubled code paths (JSON and NDJSON) | The two drift apart | One serializer in `evidence.py`, and the existing handler tests run parameterised over both `Accept` values |
| Page reload loses the partial result | Old behaviour returns | Accepted for now; that is the job API's advantage, and the reason it stays the fallback |
| A cancelled stream looks like a short one | The page could render a half-filled comparison as complete | No `done` event means cancelled: the page keeps the panes it has and says the rest was abandoned |
| `parse_slots` held for the whole stream | Capacity semantics change quietly | Slot lifetime is already "the whole request"; make it explicit in the slot test |

## Tests

1. **Handler, streaming:** a fake service that yields cells with delays — assert the first `cell` line
   reaches the client *before* the slowest engine finishes. This is the regression for "the page shows
   nothing while three panes are ready".
2. **Handler, compatibility:** the same request without the `Accept` header returns one JSON body, byte
   for byte what today's tests expect.
3. **Handler, pre-flight:** an unknown mode still answers 400 with JSON, never a half-open stream.
4. **Use case:** a sink receives events in the order they resolve, and the returned document equals the
   non-streamed one.
5. **Page:** replay a recorded NDJSON fixture through the template's parse function and assert three
   panes with text and one with a named failure. (No JS test harness exists; either add one or assert on
   the built page with a scripted `fetch`.)

## Rollout

1. Ship the server side first — it is opt-in, so the deployed page is unaffected.
2. `make site`, publish, and deploy the API together; keep `Accept` off in the page until the API is live.
3. Watch one 4-engine upload: panes should fill in the order the engines answer.
4. Rollback: drop the `Accept` header from the page; nothing else changes.
