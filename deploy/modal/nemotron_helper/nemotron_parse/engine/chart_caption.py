# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Render titles explicitly declared by native chart metadata as table captions."""
import re
from html import escape


def transform(text):
    changed = 0

    def chart(m):
        nonlocal changed
        body = m[2]
        if "<chart" in body:
            return m[0]
        title = re.search(r"(?m)^title:[ \t]*([^\n]*)$", body)
        if not title:
            return m[0]
        title_text = title[1].strip()
        if not title_text:
            return m[0]
        count = 0
        pattern = re.compile(
            r"(?m)^\|[^\n]*\|[ \t]*\n\|[ \t:|\-]+\|[ \t]*\n(?:\|[^\n]*\|[ \t]*(?:\n|$))+"
        )

        def table(t):
            nonlocal count
            raw = t[0]
            if r"\|" in raw or "`" in raw:
                return raw
            lines = [a.strip() for a in raw.strip().splitlines()]
            cells = [[c.strip() for c in line.strip("|").split("|")] for line in lines]
            if (
                len(cells) < 3
                or any(len(row) != len(cells[0]) for row in cells)
                or any(
                    not re.fullmatch(r":?-+:?", c.replace(" ", "")) for c in cells[1]
                )
            ):
                return raw
            count += 1
            result = (
                "<table><caption>"
                + escape(title_text)
                + "</caption><thead><tr>"
                + "".join('<th scope="col">' + escape(c) + "</th>" for c in cells[0])
                + "</tr></thead><tbody>"
            )
            result += (
                "".join(
                    "<tr>"
                    + "".join("<td>" + escape(c) + "</td>" for c in row)
                    + "</tr>"
                    for row in cells[2:]
                )
                + "</tbody></table>\n"
            )
            return result

        new = pattern.sub(table, body)
        if not count:
            return m[0]
        # The title is rendered in its associated captions, retaining text and roles.
        new = re.sub(r"(?m)^title:[ \t]*[^\n]*\n?", "", new, count=1)
        changed += count
        return m[1] + new + m[3]

    out = re.sub(r"(<chart\b[^>]*>)(.*?)(</chart>)", chart, text, flags=re.S)
    return out, changed
