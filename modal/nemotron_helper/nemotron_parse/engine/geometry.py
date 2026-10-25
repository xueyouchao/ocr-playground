# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

from __future__ import annotations

import argparse


import base64


import json


import math


import mimetypes


import re


from dataclasses import dataclass


from pathlib import Path


from typing import Any


import numpy as np


from PIL import Image


BLOCK_RE = re.compile(
    r"<x_(\d+(?:\.\d+)?)><y_(\d+(?:\.\d+)?)>(.*?)"
    r"<x_(\d+(?:\.\d+)?)><y_(\d+(?:\.\d+)?)><class_([^>]+)>",
    re.MULTILINE | re.DOTALL,
)


@dataclass(frozen=True)
class PageImage:
    key: str
    image_name: str
    image_path: Path
    width: int
    height: int
    category: str | None = None


@dataclass(frozen=True)
class CropSpec:
    crop_id: str
    page_key: str
    image_name: str
    page_image_path: str
    crop_image_name: str
    crop_image_path: str
    element_index: int
    detector_label: str
    detector_score: float
    detector_bbox_norm: tuple[float, float, float, float]
    crop_bbox_pixels: tuple[int, int, int, int]
    page_size: tuple[int, int]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def normalize_detector_label(label: Any, labels: Any = None) -> str:
    if isinstance(label, (int, float, np.integer, np.floating)) and labels is not None:
        try:
            label_idx = int(label)
            if isinstance(labels, dict):
                label = labels.get(label_idx, labels.get(str(label_idx), label))
            else:
                label = labels[label_idx]
        except Exception:
            pass
    text = str(label).strip().lower()
    text = (
        text.replace("&", "and").replace("/", "_").replace("-", "_").replace(" ", "_")
    )
    text = text.replace("__", "_")
    text = re.sub(r"_+", "_", text).strip("_")
    if text in {
        "headers_and_footers",
        "header_and_footer",
        "header_footer",
        "headers_footers",
    }:
        return "header_footer"
    if text in {
        "header",
        "footer",
        "page_header",
        "page_footer",
        "header_image",
        "footer_image",
    }:
        return "header_footer"
    if text in {"tables"}:
        return "table"
    if text in {"charts"}:
        return "chart"
    if text in {
        "texts",
        "paragraph",
        "paragraphs",
        "paragraph_title",
        "content",
        "abstract",
        "references",
        "reference",
        "reference_content",
        "footnote",
        "page_number",
        "number",
        "formula",
        "formula_number",
        "inline_formula",
        "display_formula",
        "algorithm",
        "aside_text",
        "sidebar_text",
        "list",
        "lists",
        "vertical_text",
    }:
        return "text"
    if text in {
        "title",
        "titles",
        "document_title",
        "section_header",
        "figure_title",
        "table_title",
        "chart_title",
        "figure_table_title",
        "figure_caption",
        "table_caption",
        "image_caption",
    }:
        return "title"
    if text in {
        "image",
        "images",
        "figure",
        "figures",
        "picture",
        "pictures",
        "seal",
        "stamp",
    }:
        return "infographic"
    if text in {"infographics"}:
        return "infographic"
    return text


def parse_label_filter(value: str) -> set[str]:
    if value.strip().lower() in {"", "*", "all"}:
        return set()
    return {normalize_detector_label(item) for item in value.split(",") if item.strip()}


def clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return min(high, max(low, value))


def crop_bbox_pixels(
    bbox_norm: tuple[float, float, float, float],
    width: int,
    height: int,
    *,
    margin_px: int,
    margin_ratio: float,
) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = bbox_norm
    px1, py1, px2, py2 = x1 * width, y1 * height, x2 * width, y2 * height
    mx = margin_px + (px2 - px1) * margin_ratio
    my = margin_px + (py2 - py1) * margin_ratio
    left = max(0, math.floor(px1 - mx))
    top = max(0, math.floor(py1 - my))
    right = min(width, math.ceil(px2 + mx))
    bottom = min(height, math.ceil(py2 + my))
    return left, top, right, bottom


def filter_detection_rows(
    args: argparse.Namespace, rows: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    include_labels = parse_label_filter(args.include_detector_labels)
    filtered_rows = []
    for row in rows:
        detections = []
        for det in row.get("detections") or []:
            label = normalize_detector_label(det.get("label") or "")
            if include_labels and label not in include_labels:
                continue
            if float(det.get("score") or 0.0) < args.detector_score_min:
                continue
            kept = dict(det)
            kept["label"] = label
            detections.append(kept)
        if args.detector_order == "spatial":
            detections.sort(
                key=lambda item: (
                    item["bbox_norm"][1],
                    item["bbox_norm"][0],
                    -float(item.get("score") or 0.0),
                )
            )
        if args.max_elements_per_page > 0:
            detections = detections[: args.max_elements_per_page]
        out = dict(row)
        out["detections"] = detections
        filtered_rows.append(out)
    return filtered_rows


def make_fallback_detection(page: PageImage) -> dict[str, Any]:
    return {
        "label": "text",
        "score": 0.0,
        "bbox_norm": [0.0, 0.0, 1.0, 1.0],
        "fallback": True,
    }


def build_crops(
    args: argparse.Namespace,
    pages: list[PageImage],
    detections: list[dict[str, Any]],
    crops_dir: Path,
    crops_manifest_path: Path,
) -> list[CropSpec]:
    pages_by_name = {page.image_name: page for page in pages}
    pages_by_key = {page.key: page for page in pages}
    crops: list[CropSpec] = []
    for row in detections:
        page = pages_by_key.get(str(row.get("key"))) or pages_by_name.get(
            str(row.get("image_name"))
        )
        if page is None:
            continue
        items = list(row.get("detections") or [])
        if not items and args.fallback_full_page:
            items = [make_fallback_detection(page)]
        if not items:
            continue
        with Image.open(page.image_path) as image:
            image = image.convert("RGB")
            for element_index, det in enumerate(items):
                bbox_norm = tuple(float(v) for v in det["bbox_norm"])
                crop_px = crop_bbox_pixels(
                    bbox_norm,
                    page.width,
                    page.height,
                    margin_px=args.crop_margin_px,
                    margin_ratio=args.crop_margin_ratio,
                )
                left, top, right, bottom = crop_px
                if (
                    right - left < args.min_crop_size
                    or bottom - top < args.min_crop_size
                ):
                    continue
                crop_image_name = f"{page.key}__pe_{element_index:03d}.jpg"
                crop_image_path = crops_dir / crop_image_name
                crop_image_path.parent.mkdir(parents=True, exist_ok=True)
                if not crop_image_path.is_file() or crop_image_path.stat().st_size == 0:
                    image.crop((left, top, right, bottom)).save(
                        crop_image_path, quality=95
                    )
                crops.append(
                    CropSpec(
                        crop_id=f"{page.key}__pe_{element_index:03d}",
                        page_key=page.key,
                        image_name=page.image_name,
                        page_image_path=str(page.image_path),
                        crop_image_name=crop_image_name,
                        crop_image_path=str(crop_image_path),
                        element_index=element_index,
                        detector_label=str(det.get("label") or "text"),
                        detector_score=float(det.get("score") or 0.0),
                        detector_bbox_norm=bbox_norm,
                        crop_bbox_pixels=crop_px,
                        page_size=(page.width, page.height),
                    )
                )
    write_jsonl(crops_manifest_path, [crop.__dict__ for crop in crops])
    print(f"Prepared {len(crops)} crops under {crops_dir}", flush=True)
    return crops


def image_data_url(path: Path) -> str:
    mime = mimetypes.guess_type(path.name)[0] or "image/jpeg"
    data = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{data}"


def parse_model_detection_predictions(
    args: argparse.Namespace,
    pages: list[PageImage],
    prediction_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    page_by_key = {page.key: page for page in pages}
    rows = []
    for prediction in prediction_rows:
        page = page_by_key.get(str(prediction.get("key")))
        if page is None:
            continue
        detections = []
        for match in BLOCK_RE.finditer(str(prediction.get("raw_out") or "")):
            x1, y1, _body, x2, y2, cls = match.groups()
            bbox_norm = model_bbox_to_image_norm(
                (float(x1), float(y1), float(x2), float(y2)),
                page.width,
                page.height,
                coordinate_frame=args.model_coordinate_frame,
                canvas_width=args.canvas_width,
                canvas_height=args.canvas_height,
                pad_position=args.pad_position,
            )
            if bbox_norm[2] <= bbox_norm[0] or bbox_norm[3] <= bbox_norm[1]:
                continue
            detections.append(
                {
                    "label": normalize_detector_label(cls),
                    "score": 1.0,
                    "bbox_norm": list(bbox_norm),
                    "model_class": cls,
                    "model_order": len(detections),
                }
            )
        rows.append(
            {
                "key": page.key,
                "image_name": page.image_name,
                "image_path": str(page.image_path),
                "page_size": [page.width, page.height],
                "detector_backend": "model-detection",
                "model_coordinate_frame": args.model_coordinate_frame,
                "detections": detections,
                "detection_error": prediction.get("error"),
            }
        )
    return filter_detection_rows(args, rows)


def model_canvas_norm_to_image_norm(
    bbox: tuple[float, float, float, float],
    width: int,
    height: int,
    *,
    canvas_width: float,
    canvas_height: float,
    pad_position: str,
) -> tuple[float, float, float, float]:
    x1, y1, x2, y2 = bbox
    max_coord = max(abs(x1), abs(y1), abs(x2), abs(y2))
    if max_coord > 1.5 and max_coord <= 1000.0:
        norm = (x1 / 1000.0, y1 / 1000.0, x2 / 1000.0, y2 / 1000.0)
    elif max_coord > 1000.0:
        norm = (
            x1 / max(1, width),
            y1 / max(1, height),
            x2 / max(1, width),
            y2 / max(1, height),
        )
    else:
        norm = (x1, y1, x2, y2)
    scale = min(1.0, canvas_width / max(1, width), canvas_height / max(1, height))
    resized_width = width * scale
    resized_height = height * scale
    if pad_position == "top-left":
        pad_x = 0.0
        pad_y = 0.0
    elif pad_position == "bottom-right":
        pad_x = canvas_width - resized_width
        pad_y = canvas_height - resized_height
    else:
        pad_x = (canvas_width - resized_width) / 2.0
        pad_y = (canvas_height - resized_height) / 2.0
    out = (
        (norm[0] * canvas_width - pad_x) / max(1.0, resized_width),
        (norm[1] * canvas_height - pad_y) / max(1.0, resized_height),
        (norm[2] * canvas_width - pad_x) / max(1.0, resized_width),
        (norm[3] * canvas_height - pad_y) / max(1.0, resized_height),
    )
    left, right = sorted((clamp(out[0]), clamp(out[2])))
    top, bottom = sorted((clamp(out[1]), clamp(out[3])))
    return left, top, right, bottom


def model_bbox_to_image_norm(
    bbox: tuple[float, float, float, float],
    width: int,
    height: int,
    *,
    coordinate_frame: str,
    canvas_width: float,
    canvas_height: float,
    pad_position: str,
) -> tuple[float, float, float, float]:
    if coordinate_frame in {
        "content_normalized",
        "original_normalized",
        "navit_content",
    }:
        x1, y1, x2, y2 = bbox
        max_coord = max(abs(x1), abs(y1), abs(x2), abs(y2))
        if max_coord > 1.5 and max_coord <= 1000.0:
            x1, y1, x2, y2 = (value / 1000.0 for value in (x1, y1, x2, y2))
        elif max_coord > 1000.0:
            x1, x2 = x1 / max(1, width), x2 / max(1, width)
            y1, y2 = y1 / max(1, height), y2 / max(1, height)
        left, right = sorted((clamp(x1), clamp(x2)))
        top, bottom = sorted((clamp(y1), clamp(y2)))
        return left, top, right, bottom
    return model_canvas_norm_to_image_norm(
        bbox,
        width,
        height,
        canvas_width=canvas_width,
        canvas_height=canvas_height,
        pad_position=pad_position,
    )
