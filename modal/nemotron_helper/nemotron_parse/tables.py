# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Table conversion and exports shared by public postprocessing helpers."""
import csv
import io
import json
from bs4 import BeautifulSoup
from .engine.convert.latex2html import latex_table_to_html


def convert_single_table(table):
    """
    Convert a single HTML table to Markdown format.

    Multi-row headers are flattened into a single header row using
    dot-notation column paths built from ``_expand_spans`` /
    ``_build_column_paths``.

    Args:
        table: BeautifulSoup table element

    Returns:
        str: Markdown table string
    """
    grid, header_rows = _expand_spans(table)
    if not grid:
        return ""

    paths = _build_column_paths(grid, header_rows)
    data_rows = grid[header_rows:]

    markdown_lines = []

    header_data = [p.replace("|", "\\|") for p in paths]
    markdown_lines.append("| " + " | ".join(header_data) + " |")
    markdown_lines.append("| " + " | ".join(["---"] * len(paths)) + " |")

    for row in data_rows:
        row_data = [cell.replace("|", "\\|") for cell in row]
        markdown_lines.append("| " + " | ".join(row_data) + " |")

    return "\n".join(markdown_lines)


def convert_html_tables_to_markdown(html_content):
    """
    Find all HTML tables and convert them to Markdown while preserving all other content.

    Args:
        html_content (str): HTML content that may contain tables

    Returns:
        str: HTML content with tables converted to Markdown
    """
    soup = BeautifulSoup(html_content, "html.parser")

    # Find all tables
    tables = soup.find_all("table")

    if not tables:
        return html_content  # Return original content unchanged

    # Convert each table to markdown and replace it
    for table in tables:
        markdown_table = convert_single_table(table)

        # Create a new element to replace the table
        replacement = soup.new_string("\n" + markdown_table + "\n")
        table.replace_with(replacement)

    return str(soup)


def _expand_spans(table):
    """
    Expand colspan/rowspan in a BeautifulSoup <table> into a regular grid.
    Returns a tuple ``(grid, header_rows)`` where *grid* is a list-of-lists
    of cell text strings and *header_rows* is the number of rows inside
    ``<thead>`` (zero when no leading header cells are present).
    """
    thead = table.find("thead")
    header_rows = 0
    for row in table.find_all("tr"):
        cells = row.find_all(["td", "th"], recursive=False)
        if not cells or not all(cell.name == "th" for cell in cells):
            break
        header_rows += 1

    rows = table.find_all("tr")
    if not rows:
        return [], 0

    occupied = {}
    width = 0
    for r, row in enumerate(rows):
        c = 0
        for cell in row.find_all(["td", "th"], recursive=False):
            while (r, c) in occupied:
                c += 1
            colspan = int(cell.get("colspan", 1))
            rowspan = int(cell.get("rowspan", 1))
            if not 1 <= colspan <= 512 or not 1 <= rowspan <= len(rows) - r:
                raise ValueError("Invalid table span")
            value = cell.get_text(separator=" ", strip=True)
            for rr in range(r, r + rowspan):
                for cc in range(c, c + colspan):
                    if (rr, cc) in occupied:
                        raise ValueError("Overlapping table spans")
                    occupied[rr, cc] = value
            c += colspan
            width = max(width, c)
    grid = [[occupied.get((r, c), "") for c in range(width)] for r in range(len(rows))]

    return grid, header_rows


def _build_column_paths(grid, header_rows):
    """Build hierarchical column names from multi-row headers.

    For each column, walk down rows ``0..header_rows-1``, collect distinct
    non-empty labels (skipping duplicates produced by rowspan fill), and
    join with ``'.'``.
    """
    num_cols = len(grid[0]) if grid else 0
    paths = []
    for c in range(num_cols):
        parts = []
        prev = None
        for r in range(header_rows):
            val = grid[r][c].strip() if r < len(grid) else ""
            if val and val != prev:
                parts.append(val)
                prev = val
        paths.append(".".join(parts) if parts else f"column_{c}")
    return paths


def _deduplicate_headers(raw_headers):
    """Keep existing names and assign collision-free names to duplicates."""
    reserved = {h or "column" for h in raw_headers}
    used = set()
    result = []
    for value in raw_headers:
        base = value or "column"
        name = base
        suffix = 1
        if name in used:
            while True:
                name = f"{base}_{suffix}"
                suffix += 1
                if name not in reserved and name not in used:
                    break
        used.add(name)
        result.append(name)
    return result


def _grid_to_records(grid, header_rows=1):
    """
    Convert a grid (list-of-lists) to a list of dicts.  When *header_rows* > 1
    the column names are built by joining the distinct labels down each column
    with ``'.'``.  Duplicate or empty headers get disambiguated with a suffix.
    """
    if len(grid) < header_rows + 1:
        return grid

    raw_headers = _build_column_paths(grid, header_rows)
    headers = _deduplicate_headers(raw_headers)
    return [dict(zip(headers, row)) for row in grid[header_rows:]]


def convert_html_table_to_json(html_content, records=True):
    """
    Convert HTML tables produced by ``latex_table_to_html`` into JSON.

    Parameters
    ----------
    html_content : str
        HTML string (may contain one or more ``<table>`` elements and
        surrounding non-table text).
    records : bool
        If *True* (default) the header rows are used to build hierarchical
        column names (dot-separated) and each subsequent row becomes a
        ``{header: value, ...}`` dict.
        If *False* the table is returned as a plain list-of-lists.

    Returns
    -------
    str
        A JSON string.  If the input contains a single table the top-level
        value is a list (of dicts or lists).  If there are multiple tables
        the top-level value is a list of such lists.
    """
    soup = BeautifulSoup(html_content, "html.parser")
    tables = soup.find_all("table")

    if not tables:
        return html_content

    results = []
    for table in tables:
        grid, header_rows = _expand_spans(table)
        if records:
            results.append(_grid_to_records(grid, header_rows))
        else:
            results.append(grid)

    payload = results[0] if len(results) == 1 else results
    return json.dumps(payload, ensure_ascii=False, indent=2)


def _extract_cell_objects(table):
    """Extract rows of cell-objects that preserve span metadata.

    Each cell is represented as a dict with ``text``, ``rowspan``,
    ``colspan``, and ``is_header`` fields.  Cells covered by a prior
    span are **not** duplicated — only the originating cell appears.
    This makes the representation lossless: you can reconstruct the
    HTML (or LaTeX) ``rowspan``/``colspan`` structure from it.

    Returns ``(header_cell_rows, body_cell_rows, num_cols)`` where each
    ``*_cell_rows`` is a list of lists of cell-dicts.
    """
    thead = table.find("thead")
    tbody = table.find("tbody")

    def rows_from(section):
        return section.find_all("tr") if section else []

    header_count = _expand_spans(table)[1]
    header_trs = table.find_all("tr")[:header_count]
    if tbody:
        body_trs = rows_from(tbody)
    else:
        all_trs = table.find_all("tr")
        body_trs = all_trs[len(header_trs) :]

    num_cols = 0
    for row in table.find_all("tr"):
        col_count = sum(int(c.get("colspan", 1)) for c in row.find_all(["td", "th"]))
        num_cols = max(num_cols, col_count)

    def process_rows(trs):
        out = []
        for tr in trs:
            row_cells = []
            for cell in tr.find_all(["td", "th"]):
                colspan = int(cell.get("colspan", 1))
                rowspan = int(cell.get("rowspan", 1))
                text = cell.get_text(separator=" ", strip=True)
                is_header = cell.name == "th"
                obj = {"text": text}
                if rowspan != 1:
                    obj["rowspan"] = rowspan
                if colspan != 1:
                    obj["colspan"] = colspan
                if is_header:
                    obj["is_header"] = True
                row_cells.append(obj)
            out.append(row_cells)
        return out

    return process_rows(header_trs), process_rows(body_trs), num_cols


def convert_html_table_to_json_hierarchical(html_content):
    """
    Convert HTML tables into a structured JSON format that explicitly
    separates header rows from data rows and preserves cell-level span
    metadata for lossless round-tripping.

    Returns
    -------
    str
        A JSON string.  Each table becomes an object with:

        - ``header_rows`` (int): number of header rows.
        - ``columns`` (list[str]): pre-flattened dot-notation column
          names (e.g. ``"International.EMEA.Q1"``).
        - ``headers`` (list[list[cell]]): header rows where each cell
          is either a plain string or an object with ``text``,
          ``rowspan``, and/or ``colspan`` keys.  Cells spanned by a
          prior cell are omitted (not duplicated), so the original
          merge structure is fully recoverable.
        - ``data`` (list[list[str]]): body rows (flat string arrays).

        If the input contains a single table the top-level value is a
        single object; multiple tables produce a list of objects.
    """
    soup = BeautifulSoup(html_content, "html.parser")
    tables = soup.find_all("table")

    if not tables:
        return html_content

    results = []
    for table in tables:
        header_cells, body_cells, num_cols = _extract_cell_objects(table)
        grid, header_row_count = _expand_spans(table)
        columns = _build_column_paths(grid, header_row_count)
        columns = _deduplicate_headers(columns)

        def simplify_cell(obj):
            if obj.get("rowspan", 1) == 1 and obj.get("colspan", 1) == 1:
                return obj["text"]
            return {k: v for k, v in obj.items() if k != "is_header"}

        headers = [[simplify_cell(c) for c in row] for row in header_cells]
        data = [[c["text"] for c in row] for row in body_cells]

        results.append(
            {
                "header_rows": header_row_count,
                "columns": columns,
                "headers": headers,
                "data": data,
                "body_cells": body_cells,
            }
        )

    payload = results[0] if len(results) == 1 else results
    return json.dumps(payload, ensure_ascii=False, indent=2)


def convert_html_table_to_csv(html_content):
    """
    Convert HTML tables produced by ``latex_table_to_html`` into CSV.

    Multi-row headers are collapsed into a single CSV header row using
    dot-notation column paths.  Each table becomes a CSV block separated
    by a blank line.

    Returns
    -------
    str
        CSV-formatted string.
    """
    soup = BeautifulSoup(html_content, "html.parser")
    tables = soup.find_all("table")

    if not tables:
        return html_content

    csv_blocks = []
    for table in tables:
        grid, header_rows = _expand_spans(table)
        paths = _build_column_paths(grid, header_rows)
        data_rows = grid[header_rows:]

        buf = io.StringIO()
        writer = csv.writer(buf)
        writer.writerow(paths)
        writer.writerows(data_rows)
        csv_blocks.append(buf.getvalue().rstrip("\r\n"))

    return "\n\n".join(csv_blocks)
