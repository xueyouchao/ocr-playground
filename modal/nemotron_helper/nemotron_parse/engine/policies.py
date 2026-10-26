# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Fixed single-checkpoint output conversion; no ensemble routing or benchmark inputs."""

import re

from functools import lru_cache

BLOCK = re.compile(
    r"(<x_[^>]+><y_[^>]+>)(.*?)(<x_[^>]+><y_[^>]+><class_([^>]+)>)", re.S
)

MDTABLE = re.compile(
    r"(?m)^\|[^\n]*\|[ \t]*\n\|[ \t:|\-]+\|[ \t]*\n(?:\|[^\n]*\|[ \t]*(?:\n|$))+"
)


def promote_bold_header_row(markdown):
    """Promote a mostly bold first row to header cells, preserving its styling.

    Existing header cells take precedence. This rule uses predicted markup and
    table shape; it contains no language vocabulary and never adds bold text.
    """
    from bs4 import BeautifulSoup

    def transform(match):
        original = match[0]
        soup = BeautifulSoup(original, "html.parser")
        table = soup.find("table")
        if table.find("th"):
            return original
        rows = [
            row for row in table.find_all("tr") if row.find_parent("table") == table
        ]
        if len(rows) < 2:
            return original
        cells = rows[0].find_all(["td", "th"], recursive=False)
        next_cells = rows[1].find_all(["td", "th"], recursive=False)
        texts = [cell.get_text(" ", strip=True) for cell in cells]
        numeric = lambda text: bool(re.fullmatch(r"[\s\d.,%$€£()+−-]+", text))
        eligible = (
            2 <= len(cells) <= 20
            and len(next_cells) >= 2
            and all(len(text) <= 100 for text in texts)
            and sum(bool(text) and not numeric(text) for text in texts)
            >= len(cells) * 0.6
        )
        mostly_bold = (
            sum(bool(cell.find(["b", "strong"])) for cell in cells) >= len(cells) * 0.6
        )
        if not eligible or not mostly_bold:
            return original
        before = [
            (cell.get_text(), cell.get("rowspan"), cell.get("colspan"))
            for cell in table.find_all(["td", "th"])
        ]
        for cell in cells:
            cell.name = "th"
        after = [
            (cell.get_text(), cell.get("rowspan"), cell.get("colspan"))
            for cell in table.find_all(["td", "th"])
        ]
        assert before == after, "Header conversion changed cell content or spans"
        return str(table)

    return re.sub(r"<table\b[^>]*>.*?</table>", transform, markdown, flags=re.S | re.I)


def chart_render(raw, panels=False, compound=False):
    """Render explicit panels and existing numeric value/percent pairs."""
    from html import escape

    number = r"[+−-]?\d[\d,. ]*(?:[kKmMbB])?"
    pair = re.compile(r"^\s*(" + number + r")\s*\(\s*(" + number + r"\s*%)\s*\)\s*$")

    def chart(m):
        body = m[2]
        title = re.search(r"(?m)^title:[ \t]*([^\n]*)$", body)
        title = title[1].strip() if title else ""
        panel_names = {
            p[1].strip(): p[2].strip()
            for p in re.finditer(
                r"(?m)^panel:\s*([^;\n]+)\s*;[^;\n]+;\s*([^\n]+)", body
            )
        }
        matches = list(MDTABLE.finditer(body))
        changes = {}
        for table in matches:
            if r"\|" in table[0] or "`" in table[0]:
                continue
            rows = [
                [c.strip() for c in line.strip().strip("|").split("|")]
                for line in table[0].strip().splitlines()
            ]
            if (
                len(rows) < 3
                or any(len(row) != len(rows[0]) for row in rows)
                or any(not re.fullmatch(r":?-+:?", c.replace(" ", "")) for c in rows[1])
            ):
                continue
            marker = re.search(r"table\[([^\]\n]+)\]:\s*$", body[: table.start()])
            panel = panel_names.get(marker[1].strip(), "") if marker else ""
            split = {}
            if compound:
                for c in range(1, len(rows[0])):
                    nonempty = [row[c] for row in rows[2:] if row[c]]
                    if len(nonempty) >= 2 and all(pair.fullmatch(v) for v in nonempty):
                        split[c] = True
            if not split and not (panels and panel):
                continue
            caption = " — ".join(v for v in [title, panel if panels else ""] if v)
            out = ["<table>"]
            if caption:
                out.append("<caption>" + escape(caption) + "</caption>")
            out.append("<thead><tr>")
            for c, label in enumerate(rows[0]):
                span = (
                    ' colspan="2"' if c in split else (' rowspan="2"' if split else "")
                )
                out.append('<th scope="col"' + span + ">" + escape(label) + "</th>")
            out.append("</tr>")
            if split:
                out.append(
                    "<tr>"
                    + "".join(
                        '<th scope="col">Value</th><th scope="col">Percent</th>'
                        for c in split
                    )
                    + "</tr>"
                )
            out.append("</thead><tbody>")
            for row in rows[2:]:
                out.append("<tr>")
                for c, text in enumerate(row):
                    values = (
                        (pair.fullmatch(text).groups() if text else ("", ""))
                        if c in split
                        else (text,)
                    )
                    out.extend("<td>" + escape(v.strip()) + "</td>" for v in values)
                out.append("</tr>")
            out.append("</tbody></table>\n")
            changes[table.start()] = "".join(out)
        if not changes:
            return m[0]
        new = MDTABLE.sub(lambda t: changes.get(t.start(), t[0]), body)
        if len(changes) == len(matches) and title:
            new = re.sub(r"(?m)^title:[ \t]*[^\n]*\n?", "", new, count=1)
        return m[1] + new + m[3]

    return re.sub(r"(<chart\b[^>]*>)(.*?)(</chart>)", chart, raw, flags=re.S)


def native_grid(body, repair=False):
    from .convert.rich_table_reader import split_top, argument

    pattern = re.compile(
        r"\\begin\{tabular\}\{([clrx]+)\}\n(.*?)\n\\end\{tabular\}", re.S
    )
    matches = list(pattern.finditer(body))
    if len(matches) != 1:
        return body, None
    m = matches[0]
    new = m[0]
    changed = False

    def cell(part):
        text = part.strip()
        rs = cs = 1
        while text.startswith((r"\multicolumn", r"\multirow")):
            command = "multicolumn" if text.startswith(r"\multicolumn") else "multirow"
            pos = len(command) + 1
            count, pos = argument(text, pos)
            _, pos = argument(text, pos)
            inner, pos = argument(text, pos)
            if text[pos:].strip():
                raise ValueError("trailing_span_content")
            if command == "multicolumn":
                cs = int(count)
            else:
                rs = int(count)
            text = inner
        header = False
        if text.startswith(r"\thead{"):
            inner, end = argument(text, len(r"\thead"))
            header = True
            if text[end:].strip():
                raise ValueError("trailing_header_content")
            text = inner
        return rs, cs, text, header

    try:
        rows = split_top(m[2], r"\\")
        if rows[-1].strip():
            return body, {"valid": False, "reason": "unterminated_row"}
        rows.pop()
        parsed = [[cell(c) for c in split_top(row, "&")] for row in rows]
        widths = [sum(c[1] for c in row) for row in parsed]
        columns = len(m[1])
        if (
            repair
            and widths
            and len(set(widths)) == 1
            and 1 <= widths[0] <= 512
            and widths[0] != columns
        ):
            columns = widths[0]
            cols = (m[1] + "c" * columns)[:columns]
            new = r"\begin{tabular}{" + cols + "}\n" + m[2] + "\n" + r"\end{tabular}"
            changed = True
        occupied = set()
        headers = 0
        if not rows or len(rows) > 4096 or columns > 512:
            raise ValueError("grid_size")
        for ri, row in enumerate(parsed):
            ci = 0
            for rs, cs, content, header in row:
                if rs < 1 or cs < 1 or ri + rs > len(rows) or ci + cs > columns:
                    raise ValueError("invalid_span")
                if (ri, ci) in occupied:
                    empty = re.sub(r"\s+", "", content) in ("", r"\mbox{}")
                    if (
                        not empty
                        or header
                        or rs != 1
                        or any((ri, cc) not in occupied for cc in range(ci, ci + cs))
                    ):
                        raise ValueError("invalid_placeholder")
                else:
                    if any(
                        (rr, cc) in occupied
                        for rr in range(ri, ri + rs)
                        for cc in range(ci, ci + cs)
                    ):
                        raise ValueError("overlap")
                    headers += int(header)
                    for rr in range(ri, ri + rs):
                        for cc in range(ci, ci + cs):
                            occupied.add((rr, cc))
                ci += cs
            if ci != columns:
                raise ValueError("row_width")
        header_shape = []
        for row in parsed:
            if not any(c[3] for c in row):
                break
            header_shape.append(tuple((c[0], c[1], c[3]) for c in row))
        status = {
            "valid": True,
            "columns": columns,
            "rows": len(rows),
            "header_cells": headers,
            "header_shape": tuple(header_shape),
            "changed_column_declaration": changed,
        }
    except (ValueError, TypeError, IndexError, KeyError):
        return body, {"valid": False, "reason": "inconsistent_native_grid"}
    return body[: m.start()] + new + body[m.end() :], status


native_grid = lru_cache(maxsize=32768)(native_grid)


def replace_guarded(
    raw,
    candidates,
    merger,
    similarity=0.65,
    guard="valid",
    repair=False,
    min_rows_ratio=0.0,
    iou_threshold=0.5,
    protect_headers=False,
    emit_repair=True,
):
    blocks = merger.parse_blocks(raw)
    pairs = []
    for i, b in enumerate(blocks):
        if b.cls != "Table":
            continue
        _, oldstatus = native_grid(b.body, repair)
        for j, c in enumerate(candidates):
            if c.cls != "Table" or not c.body.strip():
                continue
            overlap = merger.iou(b.bbox, c.bbox)
            if overlap < iou_threshold:
                continue
            body, status = native_grid(c.body, repair)
            if status and not status["valid"]:
                continue
            if c.body.count(r"\begin{tabular}") != c.body.count(r"\end{tabular}"):
                continue
            if (
                guard == "columns"
                and status
                and oldstatus
                and oldstatus["valid"]
                and status["columns"] != oldstatus["columns"]
            ):
                continue
            if (
                min_rows_ratio
                and status
                and oldstatus
                and oldstatus["valid"]
                and status["rows"] < oldstatus["rows"] * min_rows_ratio
            ):
                continue
            if (
                protect_headers
                and status
                and oldstatus
                and oldstatus["valid"]
                and oldstatus["header_shape"]
                and status["header_shape"] != oldstatus["header_shape"]
            ):
                continue
            if not emit_repair:
                body = c.body
            sim = merger.text_similarity(b.body, body)
            if sim < similarity:
                continue
            pairs.append((overlap, sim, i, j, body))
    chosen = {}
    used = set()
    for overlap, sim, i, j, body in sorted(pairs, reverse=True):
        if i in chosen or j in used:
            continue
        chosen[i] = body
        used.add(j)
    index = iter(range(len(blocks)))

    def f(m):
        i = next(index)
        return m[1] + chosen[i] + m[3] if i in chosen else m[0]

    return BLOCK.sub(f, raw)
