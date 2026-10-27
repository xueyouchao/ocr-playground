# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Image-based single/double/triple parsing with a vLLM endpoint or custom backend."""
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace
import base64
import math
import os
import re

from PIL import Image

from .engine import geometry, merge, policies
from .engine.regions import select_union_regions
from .postprocessing import PostprocessOptions, clean_raw_output, postprocess_document

PROMPT = "</s><s><predict_bbox><predict_classes><output_markdown><predict_text_in_pic>"
DETECT_PROMPT = (
    "</s><s><predict_bbox><predict_classes><output_no_text><predict_no_text_in_pic>"
)


@dataclass(frozen=True)
class GenerationSettings:
    temperature: float = 0.2
    repetition_penalty: float = 1.05
    max_tokens: int = 8744
    top_k: int = 50
    top_p: float = 1.0
    seed: int = 0

    def __post_init__(self):
        if (
            self.max_tokens <= 0
            or not math.isfinite(self.temperature)
            or self.temperature < 0
        ):
            raise ValueError("Invalid generation temperature or token limit")
        if not math.isfinite(self.repetition_penalty) or self.repetition_penalty <= 0:
            raise ValueError("repetition_penalty must be positive and finite")
        if not 0 < self.top_p <= 1 or self.top_k == 0 or self.top_k < -1:
            raise ValueError("Invalid top_p or top_k")


class GenerationError(RuntimeError):
    """A failed or incomplete generation; raw output remains available on response."""

    def __init__(self, stage, response):
        self.stage = stage
        self.response = response
        super().__init__(
            f'{stage} generation failed or is incomplete (finish_reason={response.get("finish_reason")!r})'
        )


class TransformersBackend:
    """Adapt an already loaded Transformers model/processor to DocumentParser.

    Calls are serialized because a model instance and its RNG are shared. This
    adapter changes no model weights and does not require the serving plugin.
    """

    def __init__(self, model, processor, generation_config=None):
        from threading import Lock

        self.model = model
        self.processor = processor
        self.generation_config = generation_config or model.generation_config
        self._lock = Lock()

    def __call__(self, image, *, prompt, stage, settings):
        import torch
        from copy import deepcopy

        with self._lock:
            inputs = self.processor(
                images=[image],
                text=prompt,
                return_tensors="pt",
                add_special_tokens=False,
            ).to(self.model.device)
            config = deepcopy(self.generation_config)
            config.max_new_tokens = settings.max_tokens
            config.do_sample = settings.temperature > 0
            config.repetition_penalty = settings.repetition_penalty
            config.top_k = settings.top_k if settings.top_k > 0 else 0
            config.top_p = settings.top_p
            config.temperature = (
                settings.temperature if settings.temperature > 0 else 1.0
            )
            device = torch.device(self.model.device)
            devices = (
                [
                    (
                        device.index
                        if device.index is not None
                        else torch.cuda.current_device()
                    )
                ]
                if device.type == "cuda"
                else []
            )
            with torch.inference_mode(), torch.random.fork_rng(devices=devices):
                torch.random.default_generator.manual_seed(settings.seed)
                if devices:
                    with torch.cuda.device(devices[0]):
                        torch.cuda.manual_seed(settings.seed)
                outputs = self.model.generate(
                    **inputs, generation_config=config, return_dict_in_generate=True
                )
            sequence = outputs.sequences[0]
            prefix = inputs.get("decoder_input_ids")
            if prefix is None:
                prefix = inputs.get("input_ids")
            prefix_length = 1
            if (
                prefix is not None
                and len(sequence) >= prefix.shape[-1]
                and torch.equal(sequence[: prefix.shape[-1]], prefix[0])
            ):
                prefix_length = prefix.shape[-1]
            count = len(sequence) - prefix_length
            return {
                "raw_out": self.processor.batch_decode(
                    outputs.sequences, skip_special_tokens=False
                )[0],
                "finish_reason": "length" if count >= settings.max_tokens else "stop",
                "usage": {"completion_tokens": count},
            }


def _merge_args():
    return SimpleNamespace(
        structural_classes="Table,Chart",
        include_detector_labels="chart,table",
        min_structural_area=0.0,
        crop_coordinate_frame="content_normalized",
        crop_pad_position="top-left",
        prediction_pad_position="top-left",
        output_coordinate_frame="content_normalized",
        canvas_width=1664.0,
        canvas_height=2048.0,
        structural_source="crop",
        dedupe_overlap_threshold=0.85,
        dedupe_text_threshold=0.8,
        replacement_text_threshold=0.8,
    )


def merge_crops(raw, crop_rows, *, policy="alternatives"):
    """Map crop-local geometry and merge with alternatives or guarded replacement."""
    if policy not in {"alternatives", "guarded"}:
        raise ValueError("merge policy must be alternatives or guarded")
    args = _merge_args()
    candidates = []
    for row in crop_rows:
        if row.get("error") or row.get("finish_reason") in ("length", "content_filter"):
            raise GenerationError("crop", row)
        for block in merge.rebuild_crop_row_blocks(row, args):
            if block.cls in {"Table", "Chart"}:
                candidates.append(block)
    candidates.sort(key=merge.reading_order_key)
    if not candidates:
        return raw
    if policy == "alternatives":
        # Preserve differing structural hypotheses before guarded table refinement.
        # This operates only on predicted content and geometry, without a scorer.
        originals = merge.parse_blocks(raw)
        if not originals:
            return raw
        additions = []
        for candidate in candidates:
            if not merge.is_duplicate(candidate, originals + additions, args):
                additions.append(candidate)
        joined = merge.rebuild_blocks(
            merge.insert_additions_preserving_direct_order(originals, additions)
        )
        return policies.replace_guarded(
            joined, candidates, merge, guard="columns", repair=True, emit_repair=False
        )
    # Refine existing objects first. Rejected replacements must not reappear
    # as an additional overlapping object when their text differs.
    refined = policies.replace_guarded(
        raw, candidates, merge, guard="columns", repair=True, emit_repair=False
    )
    originals = merge.parse_blocks(refined)
    used = set()
    for index, original in enumerate(originals):
        if original.cls != "Chart":
            continue
        eligible = []
        for candidate_index, candidate in enumerate(candidates):
            if candidate_index in used or candidate.cls != "Chart":
                continue
            overlap = merge.iou(original.bbox, candidate.bbox)
            body = candidate.body
            valid = len(re.findall(r"<chart\b[^>]*>", body)) == body.count(
                "</chart>"
            ) == 1 and (policies.MDTABLE.search(body) or "<table" in body)
            if (
                overlap >= 0.5
                and valid
                and merge.text_similarity(original.body, body) >= 0.65
            ):
                eligible.append((overlap, candidate_index, candidate))
        if eligible:
            _, candidate_index, candidate = max(eligible, key=lambda item: item[:2])
            originals[index] = merge.block_with_raw(
                original.bbox, original.cls, candidate.body
            )
            used.add(candidate_index)
    additions = []
    for candidate in candidates:
        if any(
            candidate.cls == existing.cls
            and merge.iou(candidate.bbox, existing.bbox) >= 0.5
            for existing in originals + additions
        ):
            continue
        additions.append(candidate)
    # Keep the original raw output intact if no serialized blocks were found.
    if not originals:
        return raw
    return merge.rebuild_blocks(
        merge.insert_additions_preserving_direct_order(originals, additions)
    )


class DocumentParser:
    """Parse images with an OpenAI-compatible vLLM server or an injected callable.

    backend(image, *, prompt, stage, settings) may return a raw string or a dict
    containing raw_out, finish_reason, usage, and error. A dict permits truncation
    checks; custom string backends are responsible for detecting truncation.
    The caller owns a supplied httpx client. Custom backends used by parse_many
    must support concurrent calls, or callers should set concurrency=1.
    """

    def __init__(
        self,
        base_url="http://localhost:8000/v1",
        model="nemotron-parse",
        *,
        api_key=None,
        timeout=1800.0,
        client=None,
        backend=None,
        options=None,
        fullpage=None,
        detection=None,
        crop=None,
        max_regions=80,
        crop_margin_ratio=0.04,
        crop_margin_px=12,
        merge_policy="alternatives",
    ):
        if not isinstance(max_regions, int) or max_regions <= 0:
            raise ValueError("max_regions must be a positive integer")
        if not all(
            math.isfinite(v) and v >= 0 for v in (crop_margin_ratio, crop_margin_px)
        ):
            raise ValueError("Crop margins must be nonnegative and finite")
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "EMPTY")
        self.backend = backend
        self.options = options or PostprocessOptions()
        self.settings = {
            "fullpage": fullpage or GenerationSettings(),
            "detection": detection or GenerationSettings(max_tokens=4096),
            "crop": crop or GenerationSettings(temperature=0.1, repetition_penalty=1.0),
        }
        self.max_regions = max_regions
        self.crop_margin_ratio = crop_margin_ratio
        if merge_policy not in {"alternatives", "guarded"}:
            raise ValueError("merge_policy must be alternatives or guarded")
        self.merge_policy = merge_policy
        self.crop_margin_px = crop_margin_px
        self._owned_client = client is None and backend is None
        if self._owned_client:
            import httpx

            client = httpx.Client(
                timeout=timeout,
                limits=httpx.Limits(max_connections=512, max_keepalive_connections=256),
            )
        self.client = client

    def close(self):
        if self._owned_client:
            self.client.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def _generate(self, image, stage, *, data_url=None):
        settings = self.settings[stage]
        prompt = DETECT_PROMPT if stage == "detection" else PROMPT
        if self.backend is not None:
            result = self.backend(image, prompt=prompt, stage=stage, settings=settings)
            response = {"raw_out": result} if isinstance(result, str) else dict(result)
        else:
            if data_url is None:
                stream = BytesIO()
                image.save(stream, format="PNG")
                encoded = base64.b64encode(stream.getvalue()).decode("ascii")
                data_url = "data:image/png;base64," + encoded
            payload = {
                "model": self.model,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {"url": data_url},
                            },
                        ],
                    }
                ],
                **asdict(settings),
                "skip_special_tokens": False,
            }
            reply = self.client.post(
                self.base_url + "/chat/completions",
                json=payload,
                headers={"Authorization": "Bearer " + self.api_key},
            )
            reply.raise_for_status()
            body = reply.json()
            choice = body["choices"][0]
            response = {
                "raw_out": choice["message"].get("content") or "",
                "finish_reason": choice.get("finish_reason"),
                "usage": body.get("usage", {}),
            }
        if (
            response.get("error")
            or response.get("finish_reason") not in (None, "stop", "eos")
            or not isinstance(response.get("raw_out"), str)
            or not response["raw_out"].strip()
        ):
            raise GenerationError(stage, response)
        return response

    def parse(self, image, *, mode="single", triple_regions="union"):
        """Return Markdown, structured blocks, raw stages, and inference settings.

        double: full-page parse followed by Table/Chart crop parsing.
        triple: full-page parse, separate detection, then Table/Chart crops.
        union adds nonoverlapping parser regions to the detector's proposals.
        A pass describes a stage; a crop stage may make multiple model calls.
        """
        if mode not in ("single", "double", "triple"):
            raise ValueError("mode must be single, double, or triple")
        if triple_regions not in ("detector", "union"):
            raise ValueError("triple_regions must be detector or union")
        if isinstance(image, Image.Image):
            image = image.convert("RGB")
        else:
            with Image.open(image) as source:
                image = source.convert("RGB")
        stages = {"fullpage": self._generate(image, "fullpage"), "crops": []}
        raw = clean_raw_output(
            stages["fullpage"]["raw_out"], deduplicate=self.options.deduplicate
        )
        regions = []
        if mode != "single":
            page = geometry.PageImage(
                "page", "image.png", Path("image.png"), *image.size
            )
            args = SimpleNamespace(
                include_detector_labels="table,chart",
                detector_order="model",
                detector_score_min=0.0,
                max_elements_per_page=self.max_regions,
                model_coordinate_frame="content_normalized",
                canvas_width=1664.0,
                canvas_height=2048.0,
                pad_position="top-left",
            )

            def boxes(value):
                return geometry.parse_model_detection_predictions(
                    args, [page], [{"key": "page", "raw_out": clean_raw_output(value)}]
                )

            direct_regions = boxes(raw)
            if mode == "triple":
                stages["detection"] = self._generate(image, "detection")
                detected = boxes(stages["detection"]["raw_out"])
                selected = (
                    select_union_regions(
                        detected, direct_regions, max_regions=self.max_regions
                    )
                    if triple_regions == "union"
                    else detected
                )
            else:
                selected = direct_regions
            regions = selected[0]["detections"]
            for index, region in enumerate(regions):
                pixels = geometry.crop_bbox_pixels(
                    tuple(region["bbox_norm"]),
                    *image.size,
                    margin_px=self.crop_margin_px,
                    margin_ratio=self.crop_margin_ratio,
                )
                if pixels[2] - pixels[0] < 8 or pixels[3] - pixels[1] < 8:
                    continue
                crop_image = image.crop(pixels)
                # Match reference crop JPEG compression without creating temporary files.
                stream = BytesIO()
                crop_image.save(stream, format="JPEG", quality=95)
                stream.seek(0)
                with Image.open(stream) as decoded:
                    response = self._generate(decoded.convert("RGB"), "crop")
                stages["crops"].append(
                    {
                        **response,
                        "crop_id": f"page__{index:03d}",
                        "page_key": "page",
                        "detector_label": region["label"],
                        "detector_score": region["score"],
                        "detector_bbox_norm": region["bbox_norm"],
                        "crop_bbox_pixels": pixels,
                        "page_size": image.size,
                    }
                )
            raw = merge_crops(raw, stages["crops"], policy=self.merge_policy)
        document = postprocess_document(
            raw, image_size=image.size, options=self.options
        )
        document["stages"] = stages
        document["inference"] = {
            "model": self.model,
            "backend": (
                type(self.backend).__name__ if self.backend is not None else "vllm_http"
            ),
            "base_url": self.base_url if self.backend is None else None,
            "mode": mode,
            "triple_regions": triple_regions if mode == "triple" else None,
            "image_size": list(image.size),
            "regions": regions,
            "requests": 1 + int(mode == "triple") + len(stages["crops"]),
            "sampling": {
                stage: asdict(setting) for stage, setting in self.settings.items()
            },
            "postprocessing": asdict(self.options),
            "max_regions": self.max_regions,
            "merge_policy": self.merge_policy,
            "crop_margin_ratio": self.crop_margin_ratio,
            "crop_margin_px": self.crop_margin_px,
            "crop_jpeg_quality": 95,
        }
        return document

    def parse_double(self, image):
        return self.parse(image, mode="double")

    def parse_triple(self, image, *, triple_regions="union"):
        return self.parse(image, mode="triple", triple_regions=triple_regions)

    def prepare(self, images, *, workers=8):
        """Decode and encode page images once, before timed parsing; caches pixels in RAM."""
        from .batch import prepare_images

        return prepare_images(images, workers=workers)

    def parse_prepared(
        self,
        prepared,
        *,
        mode="single",
        concurrency=128,
        postprocess_workers=8,
        triple_regions="union",
        allow_partial=False,
        schedule="pipelined",
    ):
        """Run a prepared batch with overlapping stages and bounded concurrency and detailed timings.

        Strict by default. allow_partial=True records capped responses, labels
        affected pages incomplete, and omits incomplete crops from refinement.
        """
        from .batch import parse_prepared

        return parse_prepared(
            self,
            prepared,
            mode=mode,
            concurrency=concurrency,
            postprocess_workers=postprocess_workers,
            triple_regions=triple_regions,
            allow_partial=allow_partial,
            schedule=schedule,
        )

    def parse_many(
        self, images, *, mode="single", concurrency=None, triple_regions="union"
    ):
        """Parse in input order; default to 128 HTTP requests or 4 backend workers."""
        if concurrency is None:
            concurrency = 128 if self.backend is None else 4
        if concurrency < 1:
            raise ValueError("concurrency must be positive")
        if self.backend is None:
            return self.parse_prepared(
                self.prepare(images),
                mode=mode,
                concurrency=concurrency,
                triple_regions=triple_regions,
            ).documents
        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            return list(
                pool.map(
                    lambda image: self.parse(
                        image, mode=mode, triple_regions=triple_regions
                    ),
                    images,
                )
            )


def main():
    import argparse
    import json

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", action="append", default=[])
    parser.add_argument("--image-dir", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--mode", choices=["single", "double", "triple"], default="single"
    )
    parser.add_argument(
        "--triple-regions", choices=["detector", "union"], default="union"
    )
    parser.add_argument("--base-url", default="http://localhost:8000/v1")
    parser.add_argument("--model", default="nemotron-parse")
    parser.add_argument("--concurrency", type=int, default=128)
    parser.add_argument("--postprocess-workers", type=int, default=8)
    parser.add_argument("--allow-partial", action="store_true")
    parser.add_argument(
        "--merge-policy", choices=["alternatives", "guarded"], default="alternatives"
    )
    parser.add_argument("--write-page-files", action="store_true")
    parser.add_argument(
        "--split-chart-value-percent",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--relocate-chart-captions", action=argparse.BooleanOptionalAction, default=True
    )
    args = parser.parse_args()
    images = list(map(Path, args.image))
    if args.image_dir:
        images.extend(
            sorted(
                p
                for p in args.image_dir.iterdir()
                if p.suffix.lower()
                in {".png", ".jpg", ".jpeg", ".webp", ".tif", ".tiff"}
            )
        )
    images = list(dict.fromkeys(images))
    if not images:
        parser.error("Provide --image or --image-dir")
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("Choose an empty output directory")
    options = PostprocessOptions(
        split_chart_value_percent=args.split_chart_value_percent,
        relocate_chart_captions=args.relocate_chart_captions,
    )
    with DocumentParser(
        args.base_url, args.model, options=options, merge_policy=args.merge_policy
    ) as runner:
        prepared = runner.prepare(images)
        result = runner.parse_prepared(
            prepared,
            mode=args.mode,
            concurrency=args.concurrency,
            postprocess_workers=args.postprocess_workers,
            triple_regions=args.triple_regions,
            allow_partial=args.allow_partial,
        )
    import gzip

    args.output.mkdir(parents=True, exist_ok=True)
    with gzip.open(
        args.output / "documents.jsonl.gz", "wt", encoding="utf-8"
    ) as output:
        for path, document in zip(images, result.documents):
            output.write(
                json.dumps({"image_path": str(path), **document}, ensure_ascii=False)
                + "\n"
            )
    (args.output / "metrics.json").write_text(
        json.dumps(result.metrics, indent=2) + "\n", encoding="utf-8"
    )
    if args.write_page_files:
        for index, (path, document) in enumerate(zip(images, result.documents)):
            stem = f"{index:04d}_{path.stem}"
            (args.output / (stem + ".md")).write_text(
                document["markdown"], encoding="utf-8"
            )
            (args.output / (stem + ".json")).write_text(
                json.dumps(
                    {"image_path": str(path), **document}, ensure_ascii=False, indent=2
                ),
                encoding="utf-8",
            )
    print(json.dumps(result.metrics, indent=2))


if __name__ == "__main__":
    main()
