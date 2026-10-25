# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Literal parser cleanup and text-only typography; no benchmark routing."""
import re

ROLES = {
    "Text",
    "List-item",
    "Caption",
    "Section-header",
    "Page-header",
    "Page-footer",
    "Footnote",
    "Title",
}
SOFT = re.compile(r"(?<=[^\W\d_])\u00ad(?=[^\W\d_])")
BULLET = re.compile(r"^[ \t]*[•◦▪●‣⁃–—][ \t]+")
BLOCK = re.compile(
    r"<x_([\d.]+)><y_([\d.]+)>(.*?)<x_([\d.]+)><y_([\d.]+)><class_([^>]+)>", re.S
)
NUMBER = r"\d+(?:\.\d+)?"
START = rf"<x_{NUMBER}><y_{NUMBER}>"
PREFIX = re.compile(rf"(?m)^[ \t]*{START}<class_[A-Za-z][A-Za-z_-]*>\s*(?={START})")


def normalize_text(cls, text):
    if cls in ROLES and not any(s in text for s in ["`", "$", r"\(", r"\["]):
        text = SOFT.sub("", text)
    if cls == "List-item":
        text = BULLET.sub("- ", text, count=1)
    return text


def repair_empty_prefix(raw):
    if "```" in raw or re.search(r"<class_(?:Code|Formula)>", raw):
        return raw
    return PREFIX.sub("", raw)


def dedupe_exact_elements(raw):
    seen = set()

    def one(m):
        identity = (m[6], tuple(map(float, [m[1], m[2], m[4], m[5]])), m[3])
        if identity in seen:
            return ""
        seen.add(identity)
        return m[0]

    return BLOCK.sub(one, raw)
