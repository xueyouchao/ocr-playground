# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

from __future__ import annotations


import re


_RE_EXTRACT_CLASS_BBOX = re.compile(
    r"<x_(\d+(?:\.\d+)?)><y_(\d+(?:\.\d+)?)>(.*?)<x_(\d+(?:\.\d+)?)><y_(\d+(?:\.\d+)?)><class_([^>]+)>",
    re.DOTALL,
)


def extract_classes_bboxes(
    text: str,
) -> tuple[list[str], list[tuple[float, float, float, float]], list[str]]:
    classes: list[str] = []
    bboxes: list[tuple[float, float, float, float]] = []
    texts: list[str] = []
    for match in _RE_EXTRACT_CLASS_BBOX.finditer(text or ""):
        x1, y1, body, x2, y2, cls = match.groups()
        classes.append(cls)
        bboxes.append((float(x1), float(y1), float(x2), float(y2)))
        texts.append(body)
    return classes, bboxes, texts


def convert_latex_tables_in_markdown(markdown_text: str) -> str:
    if "\\begin{tabular}" not in markdown_text:
        return markdown_text
    try:
        from .convert.latex2html import latex_table_to_html

        return latex_table_to_html(markdown_text)
    except (ValueError, IndexError):
        return markdown_text


def mmd_to_markdown(prediction: str, text_transform=None) -> str:
    classes, _boxes, texts = extract_classes_bboxes(prediction or "")
    if not texts:
        cleaned = re.sub(r"<x_[^>]+>|<y_[^>]+>|<class_[^>]+>", "", prediction or "")
        return convert_latex_tables_in_markdown(cleaned.replace("<tbc>", "").strip())

    blocks = []
    for cls, text in zip(classes, texts):
        if text_transform is not None:
            text = text_transform(cls, text)
        text = text.replace("<tbc>", "").strip()
        if not text:
            continue
        if cls in {"Table", "Chart"}:
            text = convert_latex_tables_in_markdown(text)
        elif cls == "Title" and not text.startswith("#"):
            text = f"# {text}"
        elif cls == "Section-header" and not text.startswith("#"):
            text = f"## {text}"
        blocks.append(text)
    return "\n\n".join(blocks).strip()
