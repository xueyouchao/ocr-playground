"""Inline the evidence into one shareable page."""
import json
from pathlib import Path

here = Path(__file__).resolve().parent
evidence = json.loads((here / "evidence.json").read_text(encoding="utf-8"))
payload = json.dumps(evidence, ensure_ascii=False).replace("<", "\\u003c")
template = (here / "template.html").read_text(encoding="utf-8")
(here / "index.html").write_text(template.replace("__EVIDENCE__", payload), encoding="utf-8")
print(f"wrote {here / 'index.html'} ({(here / 'index.html').stat().st_size:,} bytes)")
