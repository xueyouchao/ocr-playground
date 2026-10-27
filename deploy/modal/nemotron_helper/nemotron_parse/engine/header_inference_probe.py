# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Infer leading header rows from predicted table spans.
No benchmark labels, source-image metadata, or dataset IDs are accepted as input.
"""
import re
from bs4 import BeautifulSoup


def infer_header_rows(raw):
    soup = BeautifulSoup(raw, "html.parser")
    diagnostics = []
    for table in soup.find_all("table"):
        if table.find_parent("table"):
            continue
        rows = [r for r in table.find_all("tr") if r.find_parent("table") == table]
        if not rows:
            continue
        # Respect existing explicit cell roles; this probe is only for td-only output.
        if table.find("th"):
            diagnostics.append({"changed": False, "reason": "existing_th"})
            continue
        start = 0
        while start < min(3, len(rows) - 1):
            cells = rows[start].find_all(["td", "th"], recursive=False)
            if (
                len(cells) == 1
                and int(cells[0].get("colspan", 1)) > 1
                and int(cells[0].get("rowspan", 1)) == 1
            ):
                start += 1
            else:
                break
        cells = rows[start].find_all(["td", "th"], recursive=False)
        if not cells:
            continue
        numeric = lambda s: bool(re.fullmatch(r"[\s\d.,%$€£()+−-]+", s))
        texts = [
            c.get_text(" ", strip=True) for c in cells if c.get_text(" ", strip=True)
        ]
        if not texts or sum(numeric(t) for t in texts) / len(texts) > 0.5:
            diagnostics.append({"changed": False, "reason": "numeric_first_row"})
            continue
        end = start + 1
        rs = max(int(c.get("rowspan", 1)) for c in cells)
        cs = max(int(c.get("colspan", 1)) for c in cells)
        if 1 < rs <= 4:
            end = min(len(rows), start + rs)
        elif rs > 4:
            diagnostics.append({"changed": False, "reason": "long_rowspan"})
            continue
        elif cs > 1 and start + 1 < len(rows):
            nxt = [
                c
                for c in rows[start + 1].find_all(["td", "th"], recursive=False)
                if c.get_text(" ", strip=True)
            ]
            # A group heading's colspan does not establish the next row's role.
            # Require explicit header styling on every nonempty next-row cell;
            # text-heavy data rows and year-valued data must not become headers.
            if (
                len(nxt) >= 2
                and all(c.find(["b", "strong"]) for c in nxt)
                and sum(numeric(c.get_text(" ", strip=True)) for c in nxt) / len(nxt)
                < 0.5
            ):
                end = start + 2
        # Only probe multi-level headers. Single-row tables remain byte-for-byte as supplied.
        if end - start <= 1:
            diagnostics.append({"changed": False, "reason": "single_level"})
            continue
        for row in rows[start:end]:
            for c in row.find_all(["td", "th"], recursive=False):
                c.name = "th"
        diagnostics.append(
            {
                "changed": True,
                "start": start,
                "end": end,
                "rule": "bounded_span_header_topology",
            }
        )
    return str(soup), diagnostics
