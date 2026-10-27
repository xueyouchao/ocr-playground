# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Optional detector/parser region union using guarded overlap filtering."""
from .merge import iou


def select_union_regions(primary, secondary, max_regions=80):
    other = {r["key"]: r["detections"] for r in secondary}
    result = []
    for row in primary:
        original = row["detections"]
        boxes = list(original)
        for box in other.get(row["key"], []):
            if any(
                box["label"] == b["label"]
                and iou(tuple(box["bbox_norm"]), tuple(b["bbox_norm"])) >= 0.85
                for b in boxes
            ):
                continue
            boxes.append(dict(box, model_order=len(boxes)))
        boxes = boxes[:max_regions]
        boxes = [
            b
            for i, b in enumerate(boxes)
            if i < len(original)
            or not any(
                b["label"] == o["label"]
                and iou(tuple(b["bbox_norm"]), tuple(o["bbox_norm"])) >= 0.5
                for o in original
            )
        ]
        result.append(dict(row, detections=boxes))
    return result
