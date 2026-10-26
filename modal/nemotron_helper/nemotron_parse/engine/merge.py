# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

from __future__ import annotations

import argparse


import json


import re


from dataclasses import dataclass


from pathlib import Path


from typing import Any


BLOCK_RE = re.compile(
    r"<x_(\d+(?:\.\d+)?)><y_(\d+(?:\.\d+)?)>(.*?)"
    r"<x_(\d+(?:\.\d+)?)><y_(\d+(?:\.\d+)?)><class_([^>]+)>",
    re.DOTALL,
)


TOKEN_RE = re.compile(r"</?s>|<pad>|<unk>|<image>|<\|[^>]+\|>")


@dataclass(frozen=True)
class Block:
    x1: float
    y1: float
    body: str
    x2: float
    y2: float
    cls: str
    raw: str
    detector_label: str = ""
    origin: str = ""
    detector_score: float = 0.0

    @property
    def bbox(self) -> tuple[float, float, float, float]:
        left, right = sorted((self.x1, self.x2))
        top, bottom = sorted((self.y1, self.y2))
        return left, top, right, bottom


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8", errors="replace") as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: {exc}") from exc
    return rows


def row_aliases(row: dict[str, Any]) -> set[str]:
    aliases = set()
    for key in ("key", "image_name"):
        value = row.get(key)
        if value:
            text = str(value)
            aliases.add(text)
            aliases.add(Path(text).stem)
            aliases.add(Path(text).name)
    path = row.get("image_path")
    if path:
        aliases.add(Path(str(path)).name)
        aliases.add(Path(str(path)).stem)
    aliases = {alias for alias in aliases if alias}
    return aliases


def index_rows(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    out = {}
    for row in rows:
        for alias in row_aliases(row):
            out.setdefault(alias, row)
    return out


def prediction_text(row: dict[str, Any] | None) -> str:
    if not row:
        return ""
    for key in ("raw_out", "output_text", "prediction", "pred", "text"):
        value = row.get(key)
        if isinstance(value, str):
            return value
    return ""


def parse_blocks(raw: str) -> list[Block]:
    blocks = []
    for match in BLOCK_RE.finditer(raw or ""):
        x1, y1, body, x2, y2, cls = match.groups()
        blocks.append(
            Block(float(x1), float(y1), body, float(x2), float(y2), cls, match.group(0))
        )
    return blocks


def block_with_raw(
    bbox: tuple[float, float, float, float],
    cls: str,
    body: str,
    *,
    detector_label: str = "",
    origin: str = "",
    detector_score: float = 0.0,
) -> Block:
    x1, y1, x2, y2 = bbox
    raw = make_block(bbox, cls, body)
    return Block(x1, y1, body, x2, y2, cls, raw, detector_label, origin, detector_score)


def rebuild_blocks(blocks: list[Block]) -> str:
    return "\n\n".join(block.raw for block in blocks).strip()


def iou(
    a: tuple[float, float, float, float], b: tuple[float, float, float, float]
) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    denom = area_a + area_b - inter
    return inter / denom if denom > 0 else 0.0


def block_area(block: Block) -> float:
    left, top, right, bottom = block.bbox
    return max(0.0, right - left) * max(0.0, bottom - top)


def normalized_text(text: str) -> str:
    return " ".join(re.sub(r"<[^>]+>", " ", text or "").lower().split())


def text_similarity(a: str, b: str) -> float:
    a_tokens = set(normalized_text(a).split())
    b_tokens = set(normalized_text(b).split())
    if not a_tokens and not b_tokens:
        return 1.0
    if not a_tokens or not b_tokens:
        return 0.0
    return len(a_tokens & b_tokens) / len(a_tokens | b_tokens)


def clamp(value: float) -> float:
    return min(1.0, max(0.0, float(value)))


def fmt_coord(value: float) -> str:
    return f"{clamp(value):.6f}".rstrip("0").rstrip(".") or "0"


def make_block(bbox: tuple[float, float, float, float], cls: str, text: str) -> str:
    x1, y1, x2, y2 = bbox
    return f"<x_{fmt_coord(x1)}><y_{fmt_coord(y1)}>{text}<x_{fmt_coord(x2)}><y_{fmt_coord(y2)}><class_{cls}>"


def strip_model_tokens(text: str) -> str:
    return TOKEN_RE.sub("", text or "").replace("<tbc>", "").strip()


def crop_markdown(raw_text: str) -> str:
    blocks = []
    for match in BLOCK_RE.finditer(raw_text or ""):
        body = strip_model_tokens(match.group(3))
        if body:
            blocks.append(body)
    if blocks:
        return "\n\n".join(blocks).strip()
    cleaned = re.sub(r"<x_[^>]+>|<y_[^>]+>|<class_[^>]+>", "", raw_text or "")
    return strip_model_tokens(cleaned)


def normalize_detector_label(label: Any) -> str:
    text = str(label).strip().lower()
    text = (
        text.replace("&", "and").replace("/", "_").replace("-", "_").replace(" ", "_")
    )
    text = re.sub(r"_+", "_", text).strip("_")
    return text


def parse_label_filter(value: str) -> set[str]:
    return {normalize_detector_label(item) for item in value.split(",") if item.strip()}


def class_for_detector(label: str, bbox_norm: tuple[float, float, float, float]) -> str:
    label = normalize_detector_label(label)
    if label == "table":
        return "Table"
    if label == "chart":
        return "Chart"
    if label == "infographic":
        return "Picture"
    if label == "title":
        return "Title"
    if label == "header_footer":
        return (
            "Page-header"
            if ((bbox_norm[1] + bbox_norm[3]) / 2.0) < 0.5
            else "Page-footer"
        )
    return "Text"


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


def image_norm_to_model_canvas_norm(
    bbox: tuple[float, float, float, float],
    width: int,
    height: int,
    *,
    canvas_width: float,
    canvas_height: float,
    pad_position: str,
) -> tuple[float, float, float, float]:
    x1, y1, x2, y2 = bbox
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
    return (
        clamp((x1 * resized_width + pad_x) / canvas_width),
        clamp((y1 * resized_height + pad_y) / canvas_height),
        clamp((x2 * resized_width + pad_x) / canvas_width),
        clamp((y2 * resized_height + pad_y) / canvas_height),
    )


def page_norm_to_output_frame(
    bbox: tuple[float, float, float, float],
    width: int,
    height: int,
    args: argparse.Namespace,
) -> tuple[float, float, float, float]:
    """Encode a page-normalized box in the direct prediction coordinate frame."""
    frame = getattr(args, "output_coordinate_frame", None)
    if frame in {"content_normalized", "original_normalized", "navit_content"}:
        x1, y1, x2, y2 = bbox
        left, right = sorted((clamp(x1), clamp(x2)))
        top, bottom = sorted((clamp(y1), clamp(y2)))
        return left, top, right, bottom
    if frame not in {None, "padded_grid", "fixed_canvas", "navit_grid"}:
        raise ValueError(f"Unsupported hybrid output coordinate frame: {frame!r}")
    return image_norm_to_model_canvas_norm(
        bbox,
        width,
        height,
        canvas_width=args.canvas_width,
        canvas_height=args.canvas_height,
        pad_position=args.prediction_pad_position,
    )


def crop_box_to_page_canvas(
    crop_row: dict[str, Any],
    crop_local_model_bbox: tuple[float, float, float, float],
    args: argparse.Namespace,
) -> tuple[float, float, float, float]:
    left, top, right, bottom = [int(v) for v in crop_row["crop_bbox_pixels"]]
    crop_width = max(1, right - left)
    crop_height = max(1, bottom - top)
    page_width, page_height = [int(v) for v in crop_row["page_size"]]
    crop_frame = getattr(args, "crop_coordinate_frame", "padded_grid")
    if crop_frame in {"content_normalized", "original_normalized", "navit_content"}:
        x1, y1, x2, y2 = crop_local_model_bbox
        max_coord = max(abs(x1), abs(y1), abs(x2), abs(y2))
        if max_coord > 1.5 and max_coord <= 1000.0:
            x1, y1, x2, y2 = (value / 1000.0 for value in (x1, y1, x2, y2))
        left_norm, right_norm = sorted((clamp(x1), clamp(x2)))
        top_norm, bottom_norm = sorted((clamp(y1), clamp(y2)))
        crop_norm = (left_norm, top_norm, right_norm, bottom_norm)
    else:
        crop_norm = model_canvas_norm_to_image_norm(
            crop_local_model_bbox,
            crop_width,
            crop_height,
            canvas_width=args.canvas_width,
            canvas_height=args.canvas_height,
            pad_position=args.crop_pad_position,
        )
    page_norm = (
        (left + crop_norm[0] * crop_width) / page_width,
        (top + crop_norm[1] * crop_height) / page_height,
        (left + crop_norm[2] * crop_width) / page_width,
        (top + crop_norm[3] * crop_height) / page_height,
    )
    return page_norm_to_output_frame(
        page_norm,
        page_width,
        page_height,
        args,
    )


def page_bbox_to_page_canvas(
    crop_row: dict[str, Any], args: argparse.Namespace
) -> tuple[float, float, float, float]:
    page_width, page_height = [int(v) for v in crop_row["page_size"]]
    bbox = tuple(float(v) for v in crop_row["detector_bbox_norm"])
    return page_norm_to_output_frame(
        bbox,
        page_width,
        page_height,
        args,
    )


def rebuild_crop_row_blocks(
    crop_row: dict[str, Any], args: argparse.Namespace
) -> list[Block]:
    detector_label = normalize_detector_label(crop_row.get("detector_label") or "")
    detector_score = float(crop_row.get("detector_score") or 0.0)
    raw = str(crop_row.get("raw_out") or "")
    blocks = []
    for match in BLOCK_RE.finditer(raw):
        x1, y1, body, x2, y2, cls = match.groups()
        text = strip_model_tokens(body)
        if not text:
            continue
        bbox = crop_box_to_page_canvas(
            crop_row, (float(x1), float(y1), float(x2), float(y2)), args
        )
        blocks.append(
            block_with_raw(
                bbox,
                cls,
                text,
                detector_label=detector_label,
                origin="crop_element",
                detector_score=detector_score,
            )
        )
    if blocks:
        return blocks
    text = crop_markdown(raw)
    if not text:
        return []
    bbox = page_bbox_to_page_canvas(crop_row, args)
    cls = class_for_detector(
        detector_label, tuple(float(v) for v in crop_row["detector_bbox_norm"])
    )
    return [
        block_with_raw(
            bbox,
            cls,
            text,
            detector_label=detector_label,
            origin="crop_element_fallback",
            detector_score=detector_score,
        )
    ]


def parse_class_set(value: str) -> set[str]:
    return {item.strip() for item in value.split(",") if item.strip()}


def crop_element_blocks_by_page(args: argparse.Namespace) -> dict[str, list[Block]]:
    if not args.crop_element_predictions:
        return {}
    structural_classes = parse_class_set(args.structural_classes)
    include_detector_labels = parse_label_filter(args.include_detector_labels)
    by_page: dict[str, list[Block]] = {}
    for path in args.crop_element_predictions:
        for row in load_jsonl(path):
            if row.get("error"):
                continue
            detector_label = normalize_detector_label(row.get("detector_label") or "")
            if (
                include_detector_labels
                and detector_label not in include_detector_labels
            ):
                continue
            raw_text = prediction_text(row)
            if (
                args.max_crop_output_chars > 0
                and len(raw_text) > args.max_crop_output_chars
            ):
                continue
            if (
                args.max_crop_output_blocks > 0
                and len(BLOCK_RE.findall(raw_text)) > args.max_crop_output_blocks
            ):
                continue
            page_key = str(row.get("page_key") or "")
            if not page_key:
                continue
            for block in rebuild_crop_row_blocks(row, args):
                if block.cls not in structural_classes:
                    continue
                if (
                    args.min_structural_area > 0
                    and block_area(block) < args.min_structural_area
                ):
                    continue
                by_page.setdefault(page_key, []).append(block)
    for blocks in by_page.values():
        blocks.sort(key=reading_order_key)
    return by_page


def find_row(
    index: dict[str, dict[str, Any]], label: dict[str, Any]
) -> dict[str, Any] | None:
    for alias in row_aliases(label):
        if alias in index:
            return index[alias]
    key = str(label.get("key") or "")
    image_name = str(label.get("image_name") or "")
    return index.get(key) or index.get(Path(image_name).stem) or index.get(image_name)


def clone_prediction(
    label: dict[str, Any], source_row: dict[str, Any] | None, *, source_name: str
) -> dict[str, Any]:
    raw_out = prediction_text(source_row)
    row = {
        "key": label["key"],
        "image_name": label.get("image_name") or f"{label['key']}.jpg",
        "image_path": (source_row or {}).get("image_path") or label.get("image_path"),
        "category": label.get("category"),
        "raw_out": raw_out,
        "hybrid_source": source_name,
        "error": (
            (source_row or {}).get("error") if source_row else "missing_prediction"
        ),
    }
    for key in (
        "coordinate_frame",
        "image_preprocessing",
        "model_padding",
        "model_grid_size",
        "resized_content_size",
    ):
        if (source_row or {}).get(key) is not None:
            row[key] = source_row[key]
    return row


def structural_blocks(
    args: argparse.Namespace, row: dict[str, Any] | None
) -> list[Block]:
    structural_classes = parse_class_set(args.structural_classes)
    blocks = [
        block
        for block in parse_blocks(prediction_text(row))
        if block.cls in structural_classes
    ]
    if args.min_structural_area > 0:
        blocks = [
            block for block in blocks if block_area(block) >= args.min_structural_area
        ]
    return blocks


def candidate_structural_blocks(
    args: argparse.Namespace,
    label: dict[str, Any],
    structural_row: dict[str, Any] | None,
    element_blocks: dict[str, list[Block]],
) -> list[Block]:
    if element_blocks:
        return list(element_blocks.get(str(label.get("key") or ""), []))
    return structural_blocks(args, structural_row)


def detector_label_counts(blocks: list[Block]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for block in blocks:
        label = block.detector_label or "unknown"
        counts[label] = counts.get(label, 0) + 1
    return counts


def is_duplicate(
    candidate: Block, existing: list[Block], args: argparse.Namespace
) -> bool:
    for block in existing:
        if block.cls != candidate.cls:
            continue
        if iou(block.bbox, candidate.bbox) < args.dedupe_overlap_threshold:
            continue
        if text_similarity(block.body, candidate.body) >= args.dedupe_text_threshold:
            return True
    return False


def reading_order_key(block: Block) -> tuple[float, float, float, float]:
    return block.y1, block.x1, block.y2, block.x2


def insert_additions_preserving_direct_order(
    direct_ordered: list[Block], additions: list[Block]
) -> list[Block]:
    """Keep the direct model's block sequence stable, inserting new blocks by bbox."""
    if not additions:
        return direct_ordered
    additions_sorted = sorted(additions, key=reading_order_key)
    merged = []
    add_idx = 0
    for block in direct_ordered:
        block_key = reading_order_key(block)
        while (
            add_idx < len(additions_sorted)
            and reading_order_key(additions_sorted[add_idx]) < block_key
        ):
            merged.append(additions_sorted[add_idx])
            add_idx += 1
        merged.append(block)
    merged.extend(additions_sorted[add_idx:])
    return merged


def build_direct_plus_ordered_structural_addonly(
    args: argparse.Namespace,
    labels: list[dict[str, Any]],
    indexes: dict[str, dict[str, dict[str, Any]]],
    element_blocks: dict[str, list[Block]],
) -> list[dict[str, Any]]:
    structural_index = indexes.get(args.structural_source)
    if structural_index is None and not element_blocks:
        raise SystemExit(
            f"Missing --{args.structural_source}-predictions for structural merge"
        )
    rows = []
    for label in labels:
        direct = find_row(indexes["direct"], label)
        structural = (
            find_row(structural_index, label) if structural_index is not None else None
        )
        direct_blocks = parse_blocks(prediction_text(direct))
        added = []
        skipped_duplicates = 0
        for block in candidate_structural_blocks(
            args, label, structural, element_blocks
        ):
            if is_duplicate(block, direct_blocks + added, args):
                skipped_duplicates += 1
                continue
            added.append(block)
        merged_blocks = insert_additions_preserving_direct_order(direct_blocks, added)
        row = clone_prediction(
            label,
            direct,
            source_name=f"direct-plus-{args.structural_source}-ordered-structural-addonly",
        )
        row["raw_out"] = (
            rebuild_blocks(merged_blocks)
            or prediction_text(direct)
            or prediction_text(structural)
        )
        row["hybrid_kept_direct_blocks"] = len(direct_blocks)
        row["hybrid_added_structural_blocks"] = len(added)
        row["hybrid_skipped_duplicate_blocks"] = skipped_duplicates
        row["hybrid_added_detector_labels"] = detector_label_counts(added)
        rows.append(row)
    return rows
