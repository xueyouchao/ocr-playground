# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Escape LaTeX cell literals while preserving balanced nested inline markup."""
import re

ESCAPES = {
    "#": r"\#",
    "%": r"\%",
    "$": r"\$",
    "_": r"\_",
    "~": r"\textasciitilde{}",
    "^": r"\textasciicircum{}",
}
MATH_OR_TAG = re.compile(r"\$\$.*?\$\$|\$[^$]*\$|\\\(.*?\\\)|\\\[.*?\\\]|<[^>]+>", re.S)


def _process_cell(text, escape_literals):
    def scan(position, closing=None, depth=0):
        if depth > 64:
            raise ValueError("inline_markup_nesting_too_deep")
        output = []
        while position < len(text):
            if closing and text.startswith(closing, position):
                return "".join(output), position + len(closing), True
            protected = MATH_OR_TAG.match(text, position)
            if protected:
                output.append(protected[0])
                position = protected.end()
                continue
            if text[position] == "\\":
                # Preserve escaped literals and command prefixes. Formatting
                # inside command arguments is still scanned when reached.
                output.append(text[position : position + 2])
                position += 2
                continue
            marker = next(
                (m for m in ("**", "__", "~~", "_") if text.startswith(m, position)),
                None,
            )
            if marker:
                body, end, balanced = scan(position + len(marker), marker, depth + 1)
                if balanced:
                    if escape_literals:
                        output.extend((marker, body, marker))
                    else:
                        tag = {"**": "b", "__": "b", "_": "i", "~~": "s"}[marker]
                        output.extend(("<" + tag + ">", body, "</" + tag + ">"))
                    position = end
                    continue
            output.append(
                ESCAPES.get(text[position], text[position])
                if escape_literals
                else text[position]
            )
            position += 1
        return "".join(output), position, closing is None

    return scan(0)[0]


def escape_cell(text):
    return _process_cell(text, True)


def decode_styles(text):
    return _process_cell(text, False)
