# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Decode known rich inline commands for the legacy tabular fallback.

Used only after strict rich-grid parsing fails. This preserves unknown commands
and malformed arguments; it never invents cell contents or repairs table spans.
"""
import html
import re

try:
    from .hybrid_cleanup import decode_styles
except ImportError:
    from hybrid_cleanup import decode_styles

try:
    from .rich_table_reader import TAGS, LITERALS, argument
except ImportError:
    from rich_table_reader import TAGS, LITERALS, argument


def decode_known_inline(text, _depth=0):
    """Decode balanced, recognized inline arguments without exposing delimiters."""
    if _depth > 64:
        return text
    out = []
    i = 0
    known = set(TAGS) | set(LITERALS) | {"thead", "linebreak", "mbox", "hl", "colorbox"}
    while i < len(text):
        # Preserve escaped literal characters. In particular, an escaped
        # backslash followed by 'thead' must not become a formatting command.
        if text[i] == "\\" and i + 1 < len(text) and not text[i + 1].isalpha():
            literal = text[i + 1]
            out.append(
                "\\&#" + str(ord(literal)) + ";"
                if literal in "&%$#_*{}"
                else text[i : i + 2]
            )
            i += 2
            continue
        match = re.match(r"\\([A-Za-z]+)\{", text[i:])
        if not match or match[1] not in known:
            out.append(text[i])
            i += 1
            continue
        command = match[1]
        try:
            body, end = argument(text, i + len(command) + 1)
        except ValueError:
            out.append(text[i])
            i += 1
            continue
        if command in {"hl", "colorbox"}:
            try:
                from .semantic_styles import mark_open
            except ImportError:
                from semantic_styles import mark_open
            from bs4 import BeautifulSoup

            node = BeautifulSoup("<mark></mark>", "html.parser").mark
            if command == "colorbox":
                node["background"] = body
                try:
                    body, end = argument(text, end)
                except ValueError:
                    out.append(text[i:end])
                    i = end
                    continue
            out.append(
                mark_open(node) + decode_known_inline(body, _depth + 1) + "</mark>"
            )
        elif command in TAGS or command == "thead":
            inner = decode_known_inline(body, _depth + 1)
            tag = "nvp-header-cell" if command == "thead" else TAGS[command]
            out.append("<" + tag + ">" + inner + "</" + tag + ">")
        elif command in LITERALS and not body:
            # Entities cannot be mistaken for LaTeX row/column delimiters.
            out.append("\\&#" + str(ord(LITERALS[command])) + ";")
        elif command == "linebreak" and not body:
            out.append("<nvp-line-break></nvp-line-break>")
        elif command == "mbox" and not body:
            pass
        else:
            out.append(text[i:end])
        i = end
    return decode_styles("".join(out))


def restore_header_cells(html_text):
    from bs4 import BeautifulSoup

    # Restore explicit breaks after legacy whitespace normalization, which
    # otherwise collapses consecutive linebreak commands.
    html_text = html_text.replace("<nvp-line-break></nvp-line-break>", "<br>")
    if "<nvp-header-cell>" not in html_text:
        return html_text
    soup = BeautifulSoup(html_text, "html.parser")
    for cell in soup.find_all(["td", "th"]):
        markers = cell.find_all("nvp-header-cell")
        if markers:
            cell.name = "th"
            for marker in markers:
                marker.unwrap()
    return str(soup)
