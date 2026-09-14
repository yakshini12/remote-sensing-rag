"""Reproducible smoke-sample previews."""

from __future__ import annotations

import hashlib
import html
from pathlib import Path
from typing import Any


def _stable_rank(image_id: str, seed: int) -> str:
    return hashlib.sha256(f"{seed}::{image_id}".encode("utf-8")).hexdigest()


def select_smoke_samples(
    records: list[dict[str, Any]],
    *,
    seed: int,
    n: int,
) -> list[dict[str, Any]]:
    """Select n records deterministically by hashing image_id."""
    ranked = sorted(records, key=lambda r: _stable_rank(r["image_id"], seed))
    return ranked[: max(0, n)]


def write_smoke_html(
    path: Path,
    *,
    title: str,
    samples: list[dict[str, Any]],
) -> None:
    """Write a simple HTML preview (images linked by absolute path)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for s in samples:
        caps = "".join(f"<li>{html.escape(c)}</li>" for c in s.get("captions", [])[:5])
        img = html.escape(s.get("image_path", ""))
        sha = html.escape((s.get("sha256") or "")[:16])
        rows.append(
            f"""
            <tr>
              <td><code>{html.escape(s.get('image_id', ''))}</code><br/>
                  split={html.escape(str(s.get('split', '')))}<br/>
                  ok={s.get('ok')}<br/>
                  sha256={sha}...</td>
              <td><img src="file://{img}" alt="img" style="max-width:256px;max-height:256px;"/></td>
              <td><ul>{caps}</ul></td>
            </tr>
            """
        )
    doc = f"""<!doctype html>
<html><head><meta charset="utf-8"/><title>{html.escape(title)}</title>
<style>
body {{ font-family: sans-serif; margin: 24px; }}
table {{ border-collapse: collapse; width: 100%; }}
th, td {{ border: 1px solid #ccc; padding: 8px; vertical-align: top; }}
code {{ font-size: 12px; }}
</style></head>
<body>
<h1>{html.escape(title)}</h1>
<p>Local file:// image previews work when opened on the same machine.</p>
<table>
<tr><th>Record</th><th>Image</th><th>Captions</th></tr>
{''.join(rows)}
</table>
</body></html>
"""
    path.write_text(doc, encoding="utf-8")
