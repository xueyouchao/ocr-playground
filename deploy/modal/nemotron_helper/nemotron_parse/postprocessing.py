# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Prediction-only document rendering; no benchmark data or scorer dependencies."""
from dataclasses import dataclass
import math
import re

from . import tables
from .engine import markdown as adapter, policies
from .engine.header_inference_probe import infer_header_rows
from .engine.text_cleanup import (
    normalize_text,
    repair_empty_prefix,
    dedupe_exact_elements,
)
from .engine.chart_caption import transform as chart_captions
from .engine.chart_context import add_native_subtitles
from .engine.caption_relocation import relocate_above_captions


@dataclass(frozen=True)
class PostprocessOptions:
    """Structural cleanup is shared; presentation heuristics can be disabled.

    Numeric pair splitting adds Value/Percent column labels and is enabled by default.
    Caption association is enabled by default; either presentation rule can be disabled.
    """

    infer_headers: bool = True
    render_charts: bool = True
    split_chart_value_percent: bool = True
    relocate_chart_captions: bool = True
    normalize_text: bool = True
    deduplicate: bool = True


def clean_raw_output(raw, *, deduplicate=True):
    raw = repair_empty_prefix(raw or "")
    return dedupe_exact_elements(raw) if deduplicate else raw


def extract_classes_bboxes(text):
    """Extract normalized boxes and literal bodies, keeping predicted classes."""
    return adapter.extract_classes_bboxes(clean_raw_output(text))


def transform_bbox_to_original(
    bbox,
    original_width,
    original_height,
    target_w=1664,
    target_h=2048,
    padded_width=None,
    padded_height=None,
    pad_left=0,
    pad_top=0,
):
    """Scale content-normalized coordinates; padding arguments are legacy-only."""
    if original_width <= 0 or original_height <= 0:
        raise ValueError("Image dimensions must be positive")
    values = tuple(float(value) for value in bbox)
    if len(values) != 4 or not all(math.isfinite(value) for value in values):
        raise ValueError("Expected four finite bbox coordinates")
    x1, x2 = sorted((values[0], values[2]))
    y1, y2 = sorted((values[1], values[3]))
    return tuple(
        min(size, max(0.0, value * size))
        for value, size in zip((x1, y1, x2, y2), (original_width, original_height) * 2)
    )


def remove_nemotron_formatting(text):
    return text.replace("<tbc>", "").replace(r"\<|unk|\>", "").replace(r"\unknown", "")


def convert_mmd_to_plain_text_ours(text):
    """Remove simple markup while protecting code, math, and literal hashes."""
    protected = []

    def protect(match):
        protected.append(match[0])
        return f"\x00{len(protected)-1}\x00"

    text = re.sub(
        r"```.*?```|`[^`\n]*`|\$\$.*?\$\$|(?<!\\)\$[^$\n]+\$|\\\(.*?\\\)|\\\[.*?\\\]",
        protect,
        text,
        flags=re.S,
    )
    text = re.sub(r"<sup>(.*?)</sup>", lambda m: "^{" + m[1] + "}", text, flags=re.S)
    text = re.sub(r"<sub>(.*?)</sub>", lambda m: "_{" + m[1] + "}", text, flags=re.S)
    text = re.sub(r"<br\s*/?>", "\n", text, flags=re.I)
    text = re.sub(r"(?m)^ {0,3}#{1,6}[ \t]+", "", text)
    text = re.sub(r"\*\*(\S(?:[^\n]*?\S)?)\*\*", r"\1", text)
    text = re.sub(r"(?<!\w)\*(\S(?:[^*\n]*?\S)?)\*(?!\w)", r"\1", text)
    text = re.sub(r"(?<!\w)_(\S(?:[^_\n]*?\S)?)_(?!\w)", r"\1", text)
    return re.sub(r"\x00(\d+)\x00", lambda m: protected[int(m[1])], text).strip()


def _table_html(text, options):
    try:
        text = tables.latex_table_to_html(text)
    except (ValueError, IndexError):
        return text  # Preserve malformed source rather than inventing a table.
    if options.infer_headers:
        text = re.sub(
            r"<table\b[^>]*>.*?</table>",
            lambda m: infer_header_rows(m[0])[0],
            text,
            flags=re.S | re.I,
        )
        text = policies.promote_bold_header_row(text)
    return text


def postprocess_text(
    text,
    cls="Text",
    text_format="markdown",
    table_format="latex",
    blank_text_in_figures=False,
    *,
    options=None,
):
    """Process one block, retaining all previously supported export formats."""
    options = options or PostprocessOptions()
    if text_format not in ("markdown", "plain"):
        raise ValueError("text_format must be markdown or plain")
    formats = ("latex", "HTML", "markdown", "json", "json_hierarchical", "csv")
    if table_format not in formats:
        raise ValueError(f"table_format must be one of {formats}")
    text = remove_nemotron_formatting(text)
    if options.normalize_text:
        text = normalize_text(cls, text)
    if blank_text_in_figures and cls == "Picture":
        return ""
    if cls == "Table":
        if table_format == "latex":
            return text
        html = _table_html(text, options)
        converters = {
            "HTML": lambda s: s,
            "markdown": tables.convert_html_tables_to_markdown,
            "json": tables.convert_html_table_to_json,
            "json_hierarchical": tables.convert_html_table_to_json_hierarchical,
            "csv": tables.convert_html_table_to_csv,
        }
        return converters[table_format](html)
    if cls == "Chart" and options.render_charts:
        text = render_markdown(
            "<x_0><y_0>" + text + "<x_1><y_1><class_Chart>", options=options
        )
    return convert_mmd_to_plain_text_ours(text) if text_format == "plain" else text


def render_markdown(raw, *, options=None):
    """Render a page, preserving predicted block order and chart context."""
    options = options or PostprocessOptions()
    raw = clean_raw_output(raw, deduplicate=options.deduplicate)

    def base(value):
        if options.render_charts:
            value = policies.chart_render(
                value, panels=True, compound=options.split_chart_value_percent
            )

            def chart(match):
                if match[4] != "Chart":
                    return match[0]
                rendered = adapter.mmd_to_markdown(match[0])
                rendered, count = chart_captions(rendered)
                return match[1] + rendered + match[3] if count else match[0]

            value = policies.BLOCK.sub(chart, value)
        value = adapter.mmd_to_markdown(
            value, text_transform=normalize_text if options.normalize_text else None
        )
        if options.infer_headers:
            value = re.sub(
                r"<table\b[^>]*>.*?</table>",
                lambda m: infer_header_rows(m[0])[0],
                value,
                flags=re.S | re.I,
            )
            value = policies.promote_bold_header_row(value)
        return value

    text = base(raw)
    if options.render_charts:
        if options.relocate_chart_captions:
            text = relocate_above_captions(raw, text, base)
        text = add_native_subtitles(text)
    return text


def postprocess_document(raw, *, image_size=None, options=None):
    """Return JSON-serializable Markdown, normalized blocks, and cleaned raw text.

    Blocks preserve their model order and original per-block content; rendered
    content is provided separately. Pixel boxes are added when image_size=(w,h).
    """
    options = options or PostprocessOptions()
    cleaned = clean_raw_output(raw, deduplicate=options.deduplicate)
    classes, boxes, texts = adapter.extract_classes_bboxes(cleaned)
    blocks = []
    for cls, bbox, text in zip(classes, boxes, texts):
        block = {
            "class": cls,
            "bbox": list(bbox),
            "text": text,
            "content": postprocess_text(
                text, cls, table_format="HTML", options=options
            ),
        }
        if image_size is not None:
            block["bbox_pixels"] = list(transform_bbox_to_original(bbox, *image_size))
        blocks.append(block)
    return {
        "raw_out": cleaned,
        "markdown": render_markdown(cleaned, options=options),
        "blocks": blocks,
        "warnings": [
            {
                "block_index": index,
                "message": "Table markup could not be fully converted; original content preserved.",
            }
            for index, block in enumerate(blocks)
            if block["class"] == "Table" and r"\begin{tabular}" in block["content"]
        ],
        "coordinate_frame": "content_normalized",
    }
