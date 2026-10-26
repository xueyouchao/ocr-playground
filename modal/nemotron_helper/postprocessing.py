# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Backwards-compatible helpers backed by the shared document renderer."""
from nemotron_parse.postprocessing import (
    PostprocessOptions,
    clean_raw_output,
    extract_classes_bboxes,
    transform_bbox_to_original,
    postprocess_text,
    remove_nemotron_formatting,
    convert_mmd_to_plain_text_ours,
    render_markdown,
    postprocess_document,
)
