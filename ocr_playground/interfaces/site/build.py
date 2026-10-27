"""Inline the evidence into one shareable page.

Parser output is untrusted, so `<` becomes `\\u003c` before it is embedded: a record containing
`</script>` must not close the tag the evidence lives in.
"""
from __future__ import annotations

import json
from pathlib import Path

from .layout import site


def main(root: Path) -> None:
    directory = site(root)
    evidence = json.loads((directory / "evidence.json").read_text(encoding="utf-8"))
    payload = json.dumps(evidence, ensure_ascii=False).replace("<", "\\u003c")
    template = (directory / "template.html").read_text(encoding="utf-8")
    target = directory / "index.html"
    target.write_text(template.replace("__EVIDENCE__", payload), encoding="utf-8")
    print(f"wrote {target} ({target.stat().st_size:,} bytes)")
