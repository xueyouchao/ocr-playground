# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Preserve explicit chart subtitles in their existing table-caption context."""
import re
from bs4 import BeautifulSoup, NavigableString

TABLE = re.compile(r"<table\b[^>]*>.*?</table>", re.S | re.I)
CHART = re.compile(r"(<chart\b[^>]*>)(.*?)(</chart>)", re.S | re.I)


def add_native_subtitles(text):
    def chart(match):
        body = match[2]
        labels = [
            m[1].strip()
            for m in re.finditer(r"(?m)^subtitle:[ \t]*([^\n]*)", body)
            if m[1].strip()
        ]
        if not labels:
            return match[0]

        def table(m):
            soup = BeautifulSoup(m[0], "html.parser")
            node = soup.find("table")
            cap = node.find("caption", recursive=False)
            if not cap:
                return m[0]
            existing = cap.get_text(" ", strip=True)
            added = []
            for value in labels:
                if value not in existing and value not in added:
                    added.append(value)
            if not added:
                return m[0]
            cap.append(NavigableString(" — " + " — ".join(added)))
            return str(node)

        return match[1] + TABLE.sub(table, body) + match[3]

    return CHART.sub(chart, text)
