"""Manifest IO helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from rsrag.schema import validate_record


def write_manifest_jsonl(records: list[dict[str, Any]], path: Path) -> None:
    """Write records as JSON Lines (one image record per line)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False, sort_keys=True) + "\n")


def write_manifest_csv(records: list[dict[str, Any]], path: Path) -> None:
    """Write a flat CSV view (captions joined with || for readability)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for rec in records:
        row = dict(rec)
        row["captions"] = " || ".join(rec.get("captions", []))
        rows.append(row)
    pd.DataFrame(rows).to_csv(path, index=False)


def read_manifest_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def validate_manifest(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Validate schema and return a summary dict."""
    problems = []
    for i, rec in enumerate(records):
        for msg in validate_record(rec):
            problems.append({"index": i, "image_id": rec.get("image_id"), "problem": msg})
    return {
        "num_records": len(records),
        "num_ok": sum(1 for r in records if r.get("ok")),
        "num_bad": sum(1 for r in records if not r.get("ok")),
        "num_schema_problems": len(problems),
        "schema_problems_preview": problems[:50],
    }
