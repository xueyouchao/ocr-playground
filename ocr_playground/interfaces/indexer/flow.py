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
from dataclasses import dataclass

import cocoindex as coco
from cocoindex.connectors import localfs

from ...composition.container import build_container
from ...domain.document import PageLimit, SourceDocument
from ..evidence import document_evidence
from .config import SUFFIX_TYPES, ConfigError, IndexerConfig
from .projection import page_rows, payload

# The graph is built once per process. `build_container` reads the environment; nothing is contacted
# until a document is actually read.
CONFIG = IndexerConfig.from_env()
CONTAINER = build_container()
SERVICE = CONTAINER.read_document()

IDENTITY = ""


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
    builder.settings.db_path = CONFIG.db
    yield


@coco.fn(memo=True)
def read_document(digest: str, name: str, media_type: str, mode: str, identity: str, max_pages: int) -> str:
    """Read one document with one engine.

    `identity` and `max_pages` are arguments for the memo key's sake: a different model, prompt or
    endpoint changes the identity, and a different page limit changes what is read at all. The file is
    checked against the digest it was discovered under, so a corpus that moved under the run fails
    visibly.
    """
    content = (CONFIG.corpus / name).read_bytes()
    if hashlib.sha256(content).hexdigest() != digest:
        raise ValueError(f"{name} does not hold the bytes its digest names")
    document = SERVICE.execute(
        SourceDocument.accepted(content, media_type, name),
        mode,
        previews=False,
        limit=PageLimit(max_pages or None),
    )
    evidence = document_evidence(document)
    evidence["document"]["path"] = name
    evidence["document"]["digest"] = digest
    evidence["document"]["engine_identity"] = identity
    return json.dumps(evidence, ensure_ascii=False)


@coco.fn
async def index_one(source: SourceFile) -> Indexed:
    """Read one document. A file that cannot be read is that document's failure, not the run's."""
    try:
        encoded = await coco.use_mount(
            coco.component_subpath("read", source.name),
            read_document,
            source.digest,
            source.name,
            source.media_type,
            CONFIG.mode,
            IDENTITY,
            CONFIG.max_pages,
        )
    except Exception as error:  # noqa: BLE001 - one bad file must not stop the corpus
        return Indexed(source=source, error=f"{type(error).__name__}: {error}")
    return Indexed(source=source, document=json.loads(encoded))


@coco.fn
async def app_main() -> None:
    global IDENTITY
    CONFIG.validate(SERVICE.registry, CONTAINER.engines())
    IDENTITY = (
        f"{SERVICE.require_mode(CONFIG.mode).identity()} pipeline={CONTAINER.fingerprint()}"
    )

    results: list[Indexed] = []
    sources: list[SourceFile] = []
    skipped: list[str] = []
    async for file in localfs.walk_dir(CONFIG.corpus, recursive=True):
        path = file.file_path.path
        name = path.relative_to(CONFIG.corpus).as_posix()
        media_type = SUFFIX_TYPES.get(path.suffix.lower())
        if media_type is None:
            skipped.append(name)
            continue
        try:
            content = await file.read()
        except OSError as error:
            # Discovery reads bytes to identify the document. A file that cannot be read here is still
            # a discovered document, and its failure belongs in the summary beside the others.
            results.append(Indexed(
                source=SourceFile(name=name, media_type=media_type, digest=""),
                error=f"{type(error).__name__}: {error}",
            ))
            continue
        sources.append(SourceFile(
            name=name,
            media_type=media_type,
            digest=hashlib.sha256(content).hexdigest(),
        ))
    sources.sort(key=lambda source: source.name)
    skipped.sort()

    results.extend(await coco.map(index_one, sources))

    for result in sorted(results, key=lambda item: item.source.name):
        assert result.source is not None
        target = CONFIG.out / "documents" / f"{result.source.name}.json"
        content = (
            payload({"document": {"path": result.source.name, "digest": result.source.digest},
                     "error": result.error})
            if result.document is None
            else payload(result.document)
        )
        localfs.declare_file(target, content, create_parent_dirs=True)

    rows = page_rows(
        result.document for result in results if result.document is not None
    )
    failures = sorted(
        ({"document": result.source.name, "error": result.error}
         for result in results if result.document is None),
        key=lambda item: item["document"],
    )
    localfs.declare_file(
        CONFIG.out / "pages.jsonl",
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        create_parent_dirs=True,
    )
    localfs.declare_file(
        CONFIG.out / "summary.json",
        payload({
            "corpus": CONFIG.corpus.name,
            "mode": CONFIG.mode,
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


__all__ = ["ConfigError", "build_app"]
