# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Strict rich_tabular_v3 reverse reader. Other LaTeX uses the legacy parser."""
import html, re

TAGS = {
    "textstrong": "b",
    "textem": "i",
    "textbf": "b",
    "textit": "i",
    "underline": "u",
    "sout": "s",
    "textsuperscript": "sup",
    "textsubscript": "sub",
}

LITERALS = {
    "textbackslash": "\\",
    "textasciitilde": "~",
    "textasciicircum": "^",
    "textless": "<",
    "textgreater": ">",
}


def split_top(text, delimiter):
    parts = []
    start = 0
    depth = 0
    i = 0
    while i < len(text):
        if depth == 0 and text.startswith(delimiter, i):
            parts.append(text[start:i])
            i += len(delimiter)
            start = i
            continue
        if text[i] == "\\":
            i += 2
            continue
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
        if depth < 0:
            raise ValueError("unbalanced_generated_latex")
        i += 1
    if depth:
        raise ValueError("unbalanced_generated_latex")
    parts.append(text[start:])
    return parts


def argument(text, pos):
    if pos >= len(text) or text[pos] != "{":
        raise ValueError("expected_braced_argument")
    start = pos + 1
    depth = 1
    pos += 1
    while pos < len(text):
        if text[pos] == "\\":
            pos += 2
            continue
        if text[pos] == "{":
            depth += 1
        elif text[pos] == "}":
            depth -= 1
            if depth == 0:
                return text[start:pos], pos + 1
        pos += 1
    raise ValueError("unbalanced_generated_latex")


def inline_html(text):
    try:
        from .hybrid_inline import inline_html as decode
    except ImportError:
        from hybrid_inline import inline_html as decode
    return decode(text)


def cell_parts(text):
    text = text.strip()
    rs = cs = 1
    while text.startswith((r"\multicolumn", r"\multirow")):
        command = "multicolumn" if text.startswith(r"\multicolumn") else "multirow"
        pos = len(command) + 1
        count, pos = argument(text, pos)
        _, pos = argument(text, pos)
        body, pos = argument(text, pos)
        if text[pos:].strip():
            raise ValueError("trailing_span_content")
        if command == "multicolumn":
            cs = int(count)
        else:
            rs = int(count)
        text = body
    header = False
    if text.startswith(r"\thead{"):
        body, end = argument(text, len(r"\thead"))
        if text[end:].strip():
            raise ValueError("trailing_header_content")
        text = body
        header = True
    return rs, cs, inline_html(text), header


def to_html(latex):
    m = re.fullmatch(
        r"\\begin\{tabular\}\{[clrx]+\}\n(.*?)\n\\end\{tabular\}", latex, re.S
    )
    if not m:
        raise ValueError("unsupported_generated_tabular")
    rows = split_top(m.group(1), r"\\")
    if rows[-1].strip():
        raise ValueError("missing_final_row_separator")
    columns = len(re.match(r"\\begin\{tabular\}\{([clrx]+)\}", latex).group(1))
    rows.pop()
    occupied = set()
    out = ["<table>"]
    if not rows or len(rows) > 4096 or columns > 512:
        raise ValueError("generated_grid_size")
    for r, row in enumerate(rows):
        out.append("<tr>")
        c = 0
        for part in split_top(row, "&"):
            rs, cs, content, header = cell_parts(part)
            if rs < 1 or cs < 1 or r + rs > len(rows) or c + cs > columns:
                raise ValueError("invalid_generated_span")
            if (r, c) in occupied:
                if content or header or rs != 1:
                    raise ValueError("nonempty_span_placeholder")
                if any((r, cc) not in occupied for cc in range(c, c + cs)):
                    raise ValueError("invalid_span_placeholder")
                c += cs
                continue
            if any(
                (rr, cc) in occupied
                for rr in range(r, r + rs)
                for cc in range(c, c + cs)
            ):
                raise ValueError("overlapping_generated_span")
            tag = "th" if header else "td"
            out.append(
                "<"
                + tag
                + ' rowspan="'
                + str(rs)
                + '" colspan="'
                + str(cs)
                + '">'
                + content
                + "</"
                + tag
                + ">"
            )
            for rr in range(r, r + rs):
                for cc in range(c, c + cs):
                    occupied.add((rr, cc))
            c += cs
        if c != columns:
            raise ValueError("generated_column_count")
        out.append("</tr>")
    return "".join(out) + "</table>"
