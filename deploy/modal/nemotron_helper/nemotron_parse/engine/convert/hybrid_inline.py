# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Bounded Markdown/markup inline dialect inside LaTeX table cells.

Escaped punctuation is literal. Formatting is parsed before literal decoding;
literal text is never reparsed as markup. Unknown syntax fails explicitly.
"""
import html
import re
from bs4 import NavigableString, BeautifulSoup

try:
    from .semantic_styles import mark_open, ALIASES
except ImportError:
    from semantic_styles import mark_open, ALIASES
try:
    from .rich_table_reader import TAGS, LITERALS, argument
except ImportError:
    from rich_table_reader import TAGS, LITERALS, argument

ESC = {
    "&": r"\&",
    "%": r"\%",
    "$": r"\$",
    "#": r"\#",
    "_": r"\_",
    "*": r"\*",
    "{": r"\{",
    "}": r"\}",
    "\\": r"\textbackslash{}",
    "~": r"\textasciitilde{}",
    "^": r"\textasciicircum{}",
    "<": r"\textless{}",
    ">": r"\textgreater{}",
}


def encode_node(node):
    if isinstance(node, NavigableString):
        return "".join(ESC.get(char, char) for char in str(node))
    if node.name == "mark":
        return (
            mark_open(node)
            + "".join(encode_node(child) for child in node.contents)
            + "</mark>"
        )
    if node.name in ALIASES:
        node = BeautifulSoup(str(node), "html.parser").find()
        node.name = ALIASES[node.name]
    if node.name == "br":
        return r"\linebreak{}"
    content = "".join(encode_node(child) for child in node.contents)
    if node.name in {"b", "strong"}:
        return "**" + content + "**"
    if node.name in {"i", "em"}:
        # The historical cleanup deliberately strips single punctuation emphasis.
        # Markup preserves that uncommon case without changing global cleanup.
        before = str(node.previous_sibling or "")[-1:]
        after = str(node.next_sibling or "")[:1]
        if (
            re.fullmatch(r"\\?[\W_]", node.get_text())
            or not node.get_text().strip()
            or re.match(r"\w", before)
            or re.match(r"\w", after)
        ):
            return "<i>" + content + "</i>"
        return "_" + content + "_"
    if node.name == "s":
        return "~~" + content + "~~"
    if node.name == "u":
        return "<u>" + content + "</u>"
    if node.name in {"sup", "sub"}:
        return "<" + node.name + ">" + content + "</" + node.name + ">"
    return content


def inline_html(text):
    def parse(position, closing=None):
        output = []
        while position < len(text):
            if closing and text.startswith(closing, position):
                return "".join(output), position + len(closing)
            if text.startswith("<br>", position):
                output.append("<br>")
                position += 4
                continue
            if text[position] == "\n":
                output.append("<br>")
                position += 1
                continue
            mark = re.match(r"<mark(?:\s[^<>]*)?>", text[position:], re.I)
            if mark:
                opener = mark_open(
                    BeautifulSoup(mark[0] + "</mark>", "html.parser").mark
                )
                inner, position = parse(position + len(mark[0]), "</mark>")
                output.append(opener + inner + "</mark>")
                continue
            marker = next(
                (
                    m
                    for m in ("**", "_", "~~", "<u>", "<s>", "<i>", "<sup>", "<sub>")
                    if text.startswith(m, position)
                ),
                None,
            )
            if marker:
                tag = {
                    "**": "b",
                    "_": "i",
                    "~~": "s",
                    "<u>": "u",
                    "<s>": "s",
                    "<i>": "i",
                    "<sup>": "sup",
                    "<sub>": "sub",
                }[marker]
                end = marker if marker in {"**", "_", "~~"} else "</" + tag + ">"
                inner, position = parse(position + len(marker), end)
                output.append("<" + tag + ">" + inner + "</" + tag + ">")
                continue
            if text[position] != "\\":
                if text[position] in "%$#_*{}^~<>":
                    raise ValueError("unsupported_unescaped_hybrid_special")
                output.append(html.escape(text[position]))
                position += 1
                continue
            if position + 1 < len(text) and text[position + 1] in "&%$#_*{}":
                output.append(html.escape(text[position + 1]))
                position += 2
                continue
            match = re.match(r"\\([A-Za-z]+)", text[position:])
            if not match:
                raise ValueError("unsupported_hybrid_escape")
            command = match[1]
            position += len(match[0])
            body, position = argument(text, position)
            if command == "colorbox":
                color = body
                body, position = argument(text, position)
                node = BeautifulSoup("<mark></mark>", "html.parser").mark
                node["background"] = color
                output.append(mark_open(node) + inline_html(body) + "</mark>")
                continue
            if command == "hl":
                output.append("<mark>" + inline_html(body) + "</mark>")
                continue
            if command == "linebreak" and not body:
                output.append("<br>")
            elif command == "mbox" and not body:
                pass
            elif command in TAGS:
                tag = TAGS[command]
                output.append("<" + tag + ">" + inline_html(body) + "</" + tag + ">")
            elif command in LITERALS and not body:
                output.append(html.escape(LITERALS[command]))
            else:
                raise ValueError("unsupported_hybrid_command_" + command)
        if closing:
            raise ValueError("unclosed_hybrid_style_" + closing)
        return "".join(output), position

    return parse(0)[0]
