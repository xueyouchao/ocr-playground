"""CocoIndex app: index a directory of documents with the same pipeline the page uploads through.

`cocoindex update main.py` walks `OCR_CORPUS`, reads every PDF or image with the configured mode, and
declares the results under `OCR_OUT`:

    documents/<relative path>.json   one document: per-page text and counts, plus its digest
    pages.jsonl                     one row per page — the indexable corpus
    summary.json                    what this run declared, and what it could not read

Reading is memoised on the content digest, the mode and the engine identity, so a second run with
nothing changed reads nothing, and changing a file, a mode or the model re-reads exactly the documents
that change affects. That is the whole reason to put this behind CocoIndex instead of a `for` loop.
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
MAX_PAGES = int(os.environ.get("OCR_MAX_PAGES", "0")) or None
IDENTITY = ""


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
        return f"glm-ocr model={GLM_MODEL} prompt={GLM_PROMPT!r} endpoint={GLM_BASE_URL}"
    lit = _version([LIT, "--version"], "unknown")
    tesseract = _version(["tesseract", "--version"], "unknown")
    return f"liteparse-tesseract lit={lit} tesseract={tesseract}"


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
def read_document(digest: str, name: str, media_type: str, mode: str, identity: str) -> str:
    """Read one document with one engine.

    `identity` is an argument for the memo key's sake alone: a different model, prompt or endpoint
    changes it, so text read by one reader cannot be served for another. The file is checked against
    the digest it was discovered under, so a corpus that moved under the run fails visibly.
    """
    payload = (CORPUS / name).read_bytes()
    if hashlib.sha256(payload).hexdigest() != digest:
        raise ValueError(f"{name} does not hold the bytes its digest names")
    document = parse_document(payload, media_type, name, mode, previews=False, max_pages=MAX_PAGES)
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
    IDENTITY = engine_identity(MODE)

    sources: list[SourceFile] = []
    skipped: list[str] = []
    async for file in localfs.walk_dir(CORPUS, recursive=True):
        path = file.file_path.path
        name = path.relative_to(CORPUS).as_posix()
        media_type = SUFFIX_TYPES.get(path.suffix.lower())
        if media_type is None:
            skipped.append(name)
            continue
        payload = await file.read()
        sources.append(SourceFile(name=name, media_type=media_type, digest=hashlib.sha256(payload).hexdigest()))
    sources.sort(key=lambda source: source.name)
    skipped.sort()

    results = await coco.map(index_one, sources)

    for result in results:
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
            "documents": len(sources),
            "read": sum(1 for result in results if result.document is not None),
            "failed": sum(1 for result in results if result.document is None),
            "pages": len(rows),
            "failures": [
                {"document": result.source.name, "error": result.error}
                for result in results if result.document is None
            ],
            "skipped": skipped,
        }),
        create_parent_dirs=True,
    )


def build_app() -> coco.App:
    return coco.App(coco.AppConfig(name="ocr-playground-indexer"), app_main)
