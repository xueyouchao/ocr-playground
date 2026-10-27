# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Prepared image batches with bounded, overlapping HTTP concurrency and CPU rendering."""
from concurrent.futures import (
    ThreadPoolExecutor,
    ProcessPoolExecutor,
    wait,
    FIRST_COMPLETED,
)
from dataclasses import dataclass, asdict
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace
import base64
import collections
import multiprocessing
import time

from PIL import Image
from .engine import geometry
from .engine.regions import select_union_regions
from .postprocessing import clean_raw_output, postprocess_document


@dataclass
class PreparedPage:
    image: Image.Image
    data_url: str
    source: str | None


@dataclass
class PreparedBatch:
    pages: list[PreparedPage]
    preprocessing_seconds: float


@dataclass
class BatchResult:
    documents: list[dict]
    metrics: dict


def _url(data, mime):
    return f"data:{mime};base64," + base64.b64encode(data).decode("ascii")


def prepare_images(images, *, workers=8):
    if workers < 1:
        raise ValueError("workers must be positive")
    start = time.perf_counter()

    def one(value):
        if isinstance(value, Image.Image):
            image = value.convert("RGB")
            stream = BytesIO()
            image.save(stream, format="PNG")
            return PreparedPage(image, _url(stream.getvalue(), "image/png"), None)
        data = Path(value).read_bytes()
        with Image.open(BytesIO(data)) as opened:
            mime = {"JPEG": "image/jpeg", "PNG": "image/png", "WEBP": "image/webp"}.get(
                opened.format
            )
            keep_encoded = mime is not None and opened.mode == "RGB"
            image = opened.convert("RGB")
        if not keep_encoded:
            stream = BytesIO()
            image.save(stream, format="PNG")
            data, mime = stream.getvalue(), "image/png"
        return PreparedPage(image, _url(data, mime), str(value))

    with ThreadPoolExecutor(max_workers=workers) as pool:
        pages = list(pool.map(one, images))
    if not pages:
        raise ValueError("No page images provided")
    return PreparedBatch(pages, time.perf_counter() - start)


def _finish(task):
    raw, crops, size, options, merge_policy = task
    from .inference import merge_crops

    raw = clean_raw_output(raw, deduplicate=options.deduplicate)
    if crops:
        raw = merge_crops(raw, crops, policy=merge_policy)
    return postprocess_document(raw, image_size=size, options=options)


def _crop_preparer(parser, pages, triple_regions):
    args = SimpleNamespace(
        include_detector_labels="table,chart",
        detector_order="model",
        detector_score_min=0.0,
        max_elements_per_page=parser.max_regions,
        model_coordinate_frame="content_normalized",
        canvas_width=1664.0,
        canvas_height=2048.0,
        pad_position="top-left",
    )

    def page_crops(index, direct_response, detection_response=None, start_region=0):
        page = pages[index]
        page_spec = geometry.PageImage(
            "page", "image.png", Path("image.png"), *page.image.size
        )

        def boxes(response):
            return geometry.parse_model_detection_predictions(
                args,
                [page_spec],
                [{"key": "page", "raw_out": clean_raw_output(response["raw_out"])}],
            )

        selected = (
            boxes(direct_response)
            if direct_response is not None
            else [{"key": "page", "detections": []}]
        )
        if detection_response is not None:
            detected = boxes(detection_response)
            selected = (
                select_union_regions(detected, selected, max_regions=parser.max_regions)
                if triple_regions == "union"
                else detected
            )
        boxes_for_page = selected[0]["detections"]
        items, metadata = [], []
        for region_index, region in enumerate(
            boxes_for_page[start_region:], start_region
        ):
            pixels = geometry.crop_bbox_pixels(
                tuple(region["bbox_norm"]),
                *page.image.size,
                margin_px=parser.crop_margin_px,
                margin_ratio=parser.crop_margin_ratio,
            )
            if pixels[2] - pixels[0] < 8 or pixels[3] - pixels[1] < 8:
                continue
            stream = BytesIO()
            page.image.crop(pixels).save(stream, format="JPEG", quality=95)
            data = stream.getvalue()
            # HTTP sends the prepared crop directly; custom backends receive decoded pixels.
            image = None
            if parser.backend is not None:
                with Image.open(BytesIO(data)) as decoded:
                    image = decoded.convert("RGB")
            items.append((image, _url(data, "image/jpeg")))
            metadata.append(
                {
                    "page_index": index,
                    "crop_id": f"page__{region_index:03d}",
                    "page_key": "page",
                    "detector_label": region["label"],
                    "detector_score": region["score"],
                    "detector_bbox_norm": region["bbox_norm"],
                    "crop_bbox_pixels": pixels,
                    "page_size": page.image.size,
                }
            )
        return index, boxes_for_page, items, metadata

    return page_crops


def _inference_timing(spans):
    """Wall time across model calls, including gaps needed to prepare later calls."""
    elapsed = max(end for begin, end in spans.values()) - min(
        begin for begin, end in spans.values()
    )
    return {
        "inference_seconds": elapsed,
        "inference_timing_includes": [
            "all HTTP/model calls, server queueing and response transfer",
            "gaps between inference calls, including dependent crop preparation",
        ],
        "inference_timing_excludes": [
            "initial image preparation",
            "server startup",
            "warmup requests",
            "merge/render work after the final model response",
            "output serialization",
        ],
        "inference_timing_definition": "first model request start to last model response",
    }


def _pipeline(parser, pages, mode, concurrency, workers, region_policy, allow_partial):
    """Keep one bounded HTTP window full; dependent crops take the next free slot.

    Detection regions form the stable prefix of the union. They can be submitted
    before full-page parsing finishes; parser-only additions follow in union order.
    CPU crop encoding and document rendering never occupy an HTTP worker.
    """
    from .inference import GenerationError

    count = len(pages)
    direct = [None] * count
    detection = [None] * count if mode == "triple" else None
    regions = [[] for _ in pages]
    crops = [{} for _ in pages]
    documents = [None] * count
    prep_pending = [0] * count
    crop_pending = [0] * count
    sealed = [mode == "single"] * count
    render_started = set()
    ready = collections.deque()
    initial = collections.deque(
        (i, stage, (page.image, page.data_url), None)
        for i, page in enumerate(pages)
        for stage in (["fullpage", "detection"] if mode == "triple" else ["fullpage"])
    )
    futures = {}
    http_active = 0
    peak = 0
    spans = {}
    service = collections.Counter()
    started = time.perf_counter()
    page_crops = _crop_preparer(parser, pages, region_policy)

    def generate(stage, item):
        begin = time.perf_counter()
        try:
            result = parser._generate(item[0], stage, data_url=item[1])
        except GenerationError as exc:
            if not (
                allow_partial
                and exc.response.get("finish_reason") == "length"
                and exc.response.get("raw_out")
            ):
                raise
            result = dict(exc.response)
        end = time.perf_counter()
        return dict(result, latency_s=end - begin), begin, end

    # Small batches avoid process startup; threaded CPU work still cannot block HTTP.
    render_executor = (
        ProcessPoolExecutor(
            max_workers=workers, mp_context=multiprocessing.get_context("spawn")
        )
        if workers > 1 and count >= 16
        else ThreadPoolExecutor(max_workers=1)
    )
    with ThreadPoolExecutor(max_workers=concurrency) as http, ThreadPoolExecutor(
        max_workers=min(8, count)
    ) as prep, render_executor as render:

        def submit_prep(i, first, second, offset=0):
            prep_pending[i] += 1
            futures[prep.submit(page_crops, i, first, second, offset)] = (
                "prep",
                i,
                None,
            )

        def maybe_render(i):
            if (
                i in render_started
                or direct[i] is None
                or not sealed[i]
                or prep_pending[i]
                or crop_pending[i]
            ):
                return
            render_started.add(i)
            rows = [crops[i][j] for j in sorted(crops[i])]
            task = (
                direct[i]["raw_out"],
                [r for r in rows if r.get("finish_reason") != "length"],
                pages[i].image.size,
                parser.options,
                parser.merge_policy,
            )
            futures[render.submit(_finish, task)] = ("render", i, None)

        while initial or ready or futures:
            while http_active < concurrency and (ready or initial):
                i, stage, item, metadata = (
                    ready.popleft() if ready else initial.popleft()
                )
                futures[http.submit(generate, stage, item)] = (stage, i, metadata)
                http_active += 1
                peak = max(peak, http_active)
            if not futures:
                break
            done, _ = wait(futures, return_when=FIRST_COMPLETED)
            for future in done:
                kind, i, metadata = futures.pop(future)
                value = future.result()
                if kind == "prep":
                    prep_pending[i] -= 1
                    _, selected, items, rows = value
                    # Two prep jobs can finish out of order; retain the full union.
                    if len(selected) > len(regions[i]):
                        regions[i] = selected
                    for item, row in zip(items, rows):
                        ready.append((i, "crop", item, row))
                        crop_pending[i] += 1
                elif kind == "render":
                    documents[i] = value
                else:
                    http_active -= 1
                    response, begin, end = value
                    span = spans.setdefault(kind, [begin, end])
                    span[0], span[1] = min(span[0], begin), max(span[1], end)
                    service[kind] += end - begin
                    if kind == "crop":
                        crop_pending[i] -= 1
                        crops[i][metadata["crop_id"]] = {**metadata, **response}
                    elif kind == "fullpage":
                        direct[i] = response
                        if mode == "double":
                            sealed[i] = True
                            submit_prep(i, response, None)
                        elif detection is not None and detection[i] is not None:
                            sealed[i] = True
                            if region_policy == "union":
                                # The detector prefix is already queued. Only prepare additions.
                                detector_count = len(
                                    page_crops(
                                        i, None, detection[i], parser.max_regions
                                    )[1]
                                )
                                submit_prep(i, response, detection[i], detector_count)
                    else:
                        detection[i] = response
                        submit_prep(i, None, response)
                        if region_policy == "detector":
                            sealed[i] = True
                        elif direct[i] is not None:
                            sealed[i] = True
                            detector_count = len(
                                page_crops(i, None, response, parser.max_regions)[1]
                            )
                            submit_prep(i, direct[i], response, detector_count)
                maybe_render(i)
    assert all(document is not None for document in documents)
    return (
        direct,
        detection,
        regions,
        [[rows[j] for j in sorted(rows)] for rows in crops],
        documents,
        {
            **_inference_timing(spans),
            "peak_inflight_requests": peak,
            "stage_times": {
                stage + "_span_seconds": end - begin
                for stage, (begin, end) in spans.items()
            },
            "request_service_seconds": dict(service),
            "stage_first_request_seconds": {
                stage: span[0] - started for stage, span in spans.items()
            },
            "overlapping_stage_times": True,
        },
    )


def parse_prepared(
    parser,
    prepared,
    *,
    mode="single",
    concurrency=128,
    postprocess_workers=8,
    triple_regions="union",
    allow_partial=False,
    schedule="pipelined",
):
    from .inference import GenerationError

    if mode not in ("single", "double", "triple") or triple_regions not in (
        "detector",
        "union",
    ):
        raise ValueError("Invalid pipeline mode or region policy")
    if concurrency < 1 or postprocess_workers < 1:
        raise ValueError("Worker counts must be positive")
    if not isinstance(prepared, PreparedBatch) or not prepared.pages:
        raise ValueError("Use parser.prepare() to prepare a nonempty image batch")
    if schedule not in ("staged", "pipelined"):
        raise ValueError("schedule must be staged or pipelined")
    pages = prepared.pages
    start = time.perf_counter()
    stage_times = {}
    request_spans = {}

    def requests(items, stage):
        stage_start = time.perf_counter()

        def one(item):
            image, url = item
            request_start = time.perf_counter()
            try:
                response = parser._generate(image, stage, data_url=url)
            except GenerationError as exc:
                # Opt-in partial output is explicit and never counted as complete.
                if (
                    not allow_partial
                    or exc.response.get("finish_reason") != "length"
                    or not exc.response.get("raw_out")
                ):
                    raise
                response = dict(exc.response)
            request_end = time.perf_counter()
            response = dict(response, latency_s=request_end - request_start)
            return response, request_start, request_end

        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            result = list(pool.map(one, items))
        stage_times[stage + "_seconds"] = time.perf_counter() - stage_start
        request_spans[stage] = (
            min(begin for response, begin, end in result),
            max(end for response, begin, end in result),
        )
        return [response for response, begin, end in result]

    if schedule == "pipelined":
        direct, detection, regions, crops_by_page, documents, stream_metrics = (
            _pipeline(
                parser,
                pages,
                mode,
                concurrency,
                postprocess_workers,
                triple_regions,
                allow_partial,
            )
        )
        stage_times.update(stream_metrics.pop("stage_times"))
        crop_responses = [r for crops in crops_by_page for r in crops]
    else:
        direct = requests([(p.image, p.data_url) for p in pages], "fullpage")
        detection = (
            requests([(p.image, p.data_url) for p in pages], "detection")
            if mode == "triple"
            else None
        )
        regions = [[] for _ in pages]
        crop_metadata = []
        crop_items = []
        crop_responses = []
        if mode != "single":
            prep_start = time.perf_counter()
            page_crops = _crop_preparer(parser, pages, triple_regions)
            with ThreadPoolExecutor(max_workers=min(8, len(pages))) as pool:
                for index, selected, items, metadata in pool.map(
                    lambda i: page_crops(
                        i, direct[i], detection[i] if detection is not None else None
                    ),
                    range(len(pages)),
                ):
                    regions[index] = selected
                    crop_items.extend(items)
                    crop_metadata.extend(metadata)
            stage_times["crop_preprocessing_seconds"] = time.perf_counter() - prep_start
            crop_responses = requests(crop_items, "crop") if crop_items else []
        crops_by_page = [[] for _ in pages]
        for metadata, response in zip(crop_metadata, crop_responses):
            crops_by_page[metadata["page_index"]].append({**metadata, **response})
        post_start = time.perf_counter()
        tasks = [
            (
                response["raw_out"],
                [r for r in crops if r.get("finish_reason") != "length"],
                page.image.size,
                parser.options,
                parser.merge_policy,
            )
            for page, response, crops in zip(pages, direct, crops_by_page)
        ]
        if postprocess_workers == 1 or len(tasks) < 16:
            documents = list(map(_finish, tasks))
        else:
            with ProcessPoolExecutor(
                max_workers=postprocess_workers,
                mp_context=multiprocessing.get_context("spawn"),
            ) as pool:
                documents = list(pool.map(_finish, tasks, chunksize=8))
        stage_times["postprocessing_seconds"] = time.perf_counter() - post_start
        stream_metrics = _inference_timing(request_spans)
    capped = collections.Counter()
    completed = 0
    for index, document in enumerate(documents):
        stages = {"fullpage": direct[index], "crops": crops_by_page[index]}
        if detection is not None:
            stages["detection"] = detection[index]
        errors = []
        for stage in ["fullpage", "detection"]:
            if stage in stages and stages[stage].get("finish_reason") == "length":
                capped[stage] += 1
                errors.append(stage + " reached its token limit")
        for crop in stages["crops"]:
            if crop.get("finish_reason") == "length":
                capped["crop"] += 1
                errors.append(
                    "crop "
                    + crop["crop_id"]
                    + " reached its token limit; refinement omitted"
                )
        document["complete"] = not errors
        document["generation_warnings"] = errors
        document["stages"] = stages
        document["source"] = pages[index].source
        document["inference"] = {
            "mode": mode,
            "merge_policy": parser.merge_policy,
            "triple_regions": triple_regions if mode == "triple" else None,
            "model": parser.model,
            "coordinate_frame": "content_normalized",
            "regions": regions[index],
            "sampling": {
                name: asdict(value) for name, value in parser.settings.items()
            },
            "postprocessing": asdict(parser.options),
            "crop_jpeg_quality": 95,
            "crop_margin_ratio": parser.crop_margin_ratio,
            "crop_margin_px": parser.crop_margin_px,
        }
        completed += not errors
    elapsed = time.perf_counter() - start
    all_responses = direct + (detection or []) + crop_responses
    metrics = {
        "mode": mode,
        "merge_policy": parser.merge_policy,
        "schedule": schedule,
        **stream_metrics,
        "pages": len(pages),
        "complete_pages": completed,
        "incomplete_pages": len(pages) - completed,
        "concurrency": concurrency,
        "postprocess_workers": postprocess_workers,
        "preprocessing_seconds": prepared.preprocessing_seconds,
        "inference_pages_per_second": (
            len(pages) / stream_metrics["inference_seconds"]
            if stream_metrics["inference_seconds"] > 0
            else None
        ),
        "complete_inference_pages_per_second": (
            completed / stream_metrics["inference_seconds"]
            if stream_metrics["inference_seconds"] > 0
            else None
        ),
        "pipeline_seconds": elapsed,
        "processed_pages_per_second": len(pages) / elapsed,
        "complete_pages_per_second": completed / elapsed,
        "stage_times": stage_times,
        "requests": {
            "fullpage": len(direct),
            "detection": len(detection or []),
            "crop": len(crop_responses),
        },
        "token_limited_responses": dict(capped),
        "completion_tokens": sum(
            int(r.get("usage", {}).get("completion_tokens") or 0) for r in all_responses
        ),
        "postprocessing": asdict(parser.options),
        "allow_partial": allow_partial,
        "timing_excludes": [
            "initial image preparation",
            "output serialization",
            "server startup",
            "warmup requests",
        ],
        "timing_includes": [
            "all mode inference stages",
            "predicted-region crop preparation",
            "merge and document rendering",
        ],
    }
    return BatchResult(documents, metrics)
