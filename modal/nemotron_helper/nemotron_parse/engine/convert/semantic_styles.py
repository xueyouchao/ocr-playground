# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Lossless, bounded semantic inline styles shared by table and page conversion."""
import html
import re
from bs4 import BeautifulSoup

ALIASES = {
    "strong": "b",
    "em": "i",
    "ins": "u",
    "underline": "u",
    "del": "s",
    "strike": "s",
}


# Highlight color is deliberately outside the training target contract.
def mark_color(node):
    return None


def mark_open(node):
    return "<mark>"


def normalize_semantic_html(value):
    soup = BeautifulSoup(str(value), "html.parser")
    for node in list(soup.find_all(True)):
        node.name = ALIASES.get(node.name, node.name)
        if node.name == "mark":
            node.attrs = {}
        if node.name != "span":
            continue
        props = {
            k.strip().lower(): v.strip().lower()
            for part in str(node.get("style", "")).split(";")
            for k, sep, v in [part.partition(":")]
            if sep
        }
        styles = []
        if props.get("font-weight") in ["bold", "bolder", "600", "700", "800", "900"]:
            styles.append("b")
        if props.get("font-style") in ["italic", "oblique"]:
            styles.append("i")
        decoration = (
            props.get("text-decoration", "")
            + " "
            + props.get("text-decoration-line", "")
        )
        if "underline" in decoration:
            styles.append("u")
        if "line-through" in decoration:
            styles.append("s")
        if props.get("vertical-align") in ["super", "sub"]:
            styles.append("sup" if props["vertical-align"] == "super" else "sub")
        if props.get("background-color") or props.get("background"):
            styles.append("mark")
        if not styles:
            continue
        for name in reversed(styles):
            inner = soup.new_tag(name)
            for child in list(node.contents):
                inner.append(child.extract())
            node.append(inner)
        node.unwrap()
    return soup
