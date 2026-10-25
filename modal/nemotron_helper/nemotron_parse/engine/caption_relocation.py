# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Relocate only plain above-chart captions with geometric/preservation guards."""
import re
from bs4 import BeautifulSoup

BLOCK = re.compile(
    r"(<x_([\d.]+)><y_([\d.]+)>)(.*?)(<x_([\d.]+)><y_([\d.]+)><class_([^>]+)>)", re.S
)


def plain(text):
    text = text.replace("<tbc>", "").strip()
    if any(marker in text for marker in ["**", "__", "~~", "`"]):
        return None
    if re.search(r"(?<!\w)_[^_\n]+_(?!\w)|<[/a-zA-Z!]|\\[A-Za-z&%_]|\]\(", text):
        return None
    return text if text and "\n" not in text else None


def relocate_above_captions(raw, markdown, render_base):
    matches = list(BLOCK.finditer(raw))
    eligible = {}
    targets = {}
    moved = set()
    allowed = {"Chart"}
    boxes = [tuple(map(float, [m[2], m[3], m[6], m[7]])) for m in matches]
    for i, m in enumerate(matches):
        if m[8] != "Caption" or plain(m[4]) is None:
            continue
        box = boxes[i]
        candidates = []
        for j in [i - 1, i + 1]:
            if not 0 <= j < len(matches) or matches[j][8] not in allowed:
                continue
            other = boxes[j]
            if box[3] > other[1]:
                continue  # Attach above-object captions only; retain notes below as standalone.
            width = min(box[2] - box[0], other[2] - other[0])
            overlap = max(0, min(box[2], other[2]) - max(box[0], other[0]))
            gap = max(other[1] - box[3], box[1] - other[3])
            if width > 0 and overlap / width >= 0.8 and 0 <= gap <= 0.03:
                candidates.append((gap, j))
        candidates.sort()
        if not candidates or (
            len(candidates) > 1 and abs(candidates[0][0] - candidates[1][0]) < 1e-9
        ):
            continue
        gap, j = candidates[0]
        eligible.setdefault(j, []).append((gap, i))
    for j, caps in eligible.items():
        caps.sort()
        if len(caps) > 1 and abs(caps[0][0] - caps[1][0]) < 1e-9:
            continue
        caption_index = caps[0][1]
        label = plain(matches[caption_index][4])
        m = matches[j]
        body = m[4]
        if len(re.findall(r"<chart\b", body)) != 1:
            continue
        title = re.search(r"(?m)^title:[ \t]*([^\n]*)", body)
        if title:
            if label not in title[1]:
                body = (
                    body[: title.start()]
                    + "title: "
                    + title[1].strip()
                    + " — "
                    + label
                    + body[title.end() :]
                )
        else:
            body = re.sub(
                r"(<chart\b[^>]*>)", lambda q: q[0] + "\ntitle: " + label, body, count=1
            )
        derived = m[1] + body + m[5]
        rendered = render_base(derived)
        # Do not move text unless it survives inside an actual table caption.
        captions = BeautifulSoup(rendered, "html.parser").find_all("caption")
        if not any(label in cap.get_text(" ", strip=True) for cap in captions):
            continue
        targets[m.start()] = derived
        moved.add(matches[caption_index].start())
    if not targets:
        return markdown
    derived = BLOCK.sub(
        lambda m: "" if m.start() in moved else targets.get(m.start(), m[0]), raw
    )
    return render_base(derived)
