"""CocoIndex app: index a directory of documents with the same pipeline the page uploads through.

`cocoindex update main.py` walks `OCR_CORPUS`, reads every PDF or image with the configured mode, and
declares the results under `OCR_OUT`:

    documents/<relative path>.json   one document: per-page text and counts, plus its digest
    pages.jsonl                     one row per page — the indexable corpus
    summary.json                    what this run declared, and what it could not read

Reading is memoised on the content digest, the mode, the engine identity and the page limit, so a
second run with nothing changed reads nothing, and changing a file, a mode, the model or the limit
re-reads exactly the documents that change affects. That is the whole reason to put this behind
CocoIndex instead of a `for` loop.
"""
from __future__ import annotations

import hashlib
import json
import os
import pathlib
import subprocess
from dataclasses import dataclass

import cocoindex as coco
from cocoindex.connectors import localfs

import pipeline
from pipeline import (
    GLM_BASE_URL,
    GLM_MODEL,
    GLM_PROMPT,
    LIT,
    SYSTEMS,
    parse_document,
)

SUFFIX_TYPES = {
    ".pdf": "application/pdf",
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".tif": "image/tiff",
    ".tiff": "image/tiff",
    ".webp": "image/webp",
}

CORPUS = pathlib.Path(os.environ.get("OCR_CORPUS", "corpus")).resolve()
OUT = pathlib.Path(os.environ.get("OCR_OUT", "out"))
DB = pathlib.Path(os.environ.get("OCR_DB", "cocoindex.db"))
# A run with no endpoint named stays on the free CPU path; naming glm without one is an error.
MODE = os.environ.get("OCR_MODE", "").strip().lower() or ("glm" if GLM_BASE_URL else "fast")
# 0 means every page: an index that silently dropped pages would be worse than a slow run.
MAX_PAGES = int(os.environ.get("OCR_MAX_PAGES", "0"))
IDENTITY = ""

# CocoIndex memoises this module's own AST, not the code it calls, so a change to pipeline.py would
# otherwise serve text the old reader produced. The fingerprint makes the identity name the recipe too.
PIPELINE_FINGERPRINT = hashlib.sha256(pathlib.Path(pipeline.__file__).read_bytes()).hexdigest()[:12]


class ConfigError(RuntimeError):
    """The environment names a run this checkout cannot perform."""


def _version(command: list[str], fallback: str) -> str:
    try:
        output = subprocess.run(command, capture_output=True, text=True, timeout=20).stdout
    except (OSError, subprocess.SubprocessError):
        return fallback
    return output.strip().splitlines()[0] if output.strip() else fallback


def engine_identity(mode: str) -> str:
    """Names everything that changes what the reader emits, so the memo key can carry it."""
    if mode == "glm":
        return (
            f"glm-ocr model={GLM_MODEL} prompt={GLM_PROMPT!r} endpoint={GLM_BASE_URL} "
            f"pipeline={PIPELINE_FINGERPRINT}"
        )
    lit = _version([LIT, "--version"], "unknown")
    tesseract = _version(["tesseract", "--version"], "unknown")
    return f"liteparse-tesseract lit={lit} tesseract={tesseract} pipeline={PIPELINE_FINGERPRINT}"


@dataclass(frozen=True, slots=True)
class SourceFile:
    name: str
    media_type: str
    digest: str


@dataclass(frozen=True, slots=True)
class Indexed:
    source: SourceFile | None
    document: dict | None = None
    error: str = ""


@coco.lifespan
def coco_lifespan(builder: coco.EnvironmentBuilder):
    builder.settings.db_path = DB
    yield


@coco.fn(memo=True)
def read_document(digest: str, name: str, media_type: str, mode: str, identity: str, max_pages: int) -> str:
    """Read one document with one engine.

    `identity` and `max_pages` are arguments for the memo key's sake: a different model, prompt or
    endpoint changes the identity, and a different page limit changes what is read at all. The file is
    checked against the digest it was discovered under, so a corpus that moved under the run fails
    visibly.
    """
    payload = (CORPUS / name).read_bytes()
    if hashlib.sha256(payload).hexdigest() != digest:
        raise ValueError(f"{name} does not hold the bytes its digest names")
    document = parse_document(payload, media_type, name, mode, previews=False, max_pages=max_pages or None)
    document["document"]["path"] = name
    document["document"]["digest"] = digest
    document["document"]["engine_identity"] = identity
    return json.dumps(document, ensure_ascii=False)


@coco.fn
async def index_one(source: SourceFile) -> Indexed:
    """Read one document. A file that cannot be read is that document's failure, not the run's."""
    try:
        payload = await coco.use_mount(
            coco.component_subpath("read", source.name),
            read_document,
            source.digest,
            source.name,
            source.media_type,
            MODE,
            IDENTITY,
            MAX_PAGES,
        )
    except Exception as error:  # noqa: BLE001 - one bad file must not stop the corpus
        return Indexed(source=source, error=f"{type(error).__name__}: {error}")
    return Indexed(source=source, document=json.loads(payload))


def _page_rows(results: list[Indexed]) -> list[dict]:
    rows = []
    for result in results:
        if result.document is None:
            continue
        document = result.document
        system = document["systems"][0]
        for page in document["pages"]:
            cell = page["cells"][system["id"]]
            rows.append({
                "document": document["document"]["path"],
                "digest": document["document"]["digest"],
                "page": page["page"],
                "engine": system["id"],
                "chars": cell["chars"],
                "tokens": cell["tokens"],
                "numeric_share": cell["numeric_share"],
                "text": cell["text"],
            })
    rows.sort(key=lambda row: (row["document"], row["page"]))
    return rows


def _payload(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"


@coco.fn
async def app_main() -> None:
    global IDENTITY
    if MODE not in SYSTEMS:
        raise ConfigError(f"OCR_MODE={MODE!r} is not an extraction mode; use glm or fast")
    if MODE == "glm" and not GLM_BASE_URL:
        raise ConfigError("OCR_MODE=glm needs GLM_OCR_BASE_URL; leave the mode unset to use the CPU path")
    if not CORPUS.is_dir():
        raise ConfigError(f"OCR_CORPUS={str(CORPUS)!r} is not a directory")
    if MAX_PAGES < 0:
        raise ConfigError("OCR_MAX_PAGES must be 0 (every page) or a positive number")
    IDENTITY = engine_identity(MODE)

    results: list[Indexed] = []
    sources: list[SourceFile] = []
    skipped: list[str] = []
    async for file in localfs.walk_dir(CORPUS, recursive=True):
        path = file.file_path.path
        name = path.relative_to(CORPUS).as_posix()
        media_type = SUFFIX_TYPES.get(path.suffix.lower())
        if media_type is None:
            skipped.append(name)
            continue
        try:
            payload = await file.read()
        except OSError as error:
            # Discovery reads bytes to identify the document. A file that cannot be read here is still
            # a discovered document, and its failure belongs in the summary beside the others.
            results.append(Indexed(
                source=SourceFile(name=name, media_type=media_type, digest=""),
                error=f"{type(error).__name__}: {error}",
            ))
            continue
        sources.append(SourceFile(name=name, media_type=media_type, digest=hashlib.sha256(payload).hexdigest()))
    sources.sort(key=lambda source: source.name)
    skipped.sort()

    results.extend(await coco.map(index_one, sources))

    for result in sorted(results, key=lambda item: item.source.name):
        assert result.source is not None
        target = OUT / "documents" / f"{result.source.name}.json"
        content = (
            _payload({"document": {"path": result.source.name, "digest": result.source.digest},
                      "error": result.error})
            if result.document is None
            else _payload(result.document)
        )
        localfs.declare_file(target, content, create_parent_dirs=True)

    rows = _page_rows(results)
    failures = sorted(
        ({"document": result.source.name, "error": result.error}
         for result in results if result.document is None),
        key=lambda item: item["document"],
    )
    localfs.declare_file(
        OUT / "pages.jsonl",
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        create_parent_dirs=True,
    )
    localfs.declare_file(
        OUT / "summary.json",
        _payload({
            "corpus": CORPUS.name,
            "mode": MODE,
            "engine": IDENTITY,
            "documents": len(results),
            "read": sum(1 for result in results if result.document is not None),
            "failed": len(failures),
            "pages": len(rows),
            "failures": failures,
            "skipped": skipped,
        }),
        create_parent_dirs=True,
    )


def build_app() -> coco.App:
    return coco.App(coco.AppConfig(name="ocr-playground-indexer"), app_main)
