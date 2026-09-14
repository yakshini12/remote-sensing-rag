"""Report writers for Stage 0–1."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_markdown_report(path: Path, report: dict[str, Any]) -> None:
    """Human-readable Stage 0–1 audit report."""
    path.parent.mkdir(parents=True, exist_ok=True)
    ucm_i = report["integrity"]["ucm"]
    kb_i = report["integrity"]["rsitmd"]
    leak = report["leakage"]
    c = leak["counts"]
    lines = [
        "# Stage 0–1 Dataset Manifest & Leakage Audit",
        "",
        f"- Generated (UTC): `{report.get('generated_at_utc', '')}`",
        f"- Config: `{report.get('config_path', '')}`",
        f"- Seed: `{report.get('seed', '')}`",
        f"- Schema version: `{report.get('schema_version', '')}`",
        "",
        "## Summary",
        "",
        f"- UCM images: **{ucm_i['num_images']}** (ok={ucm_i['num_ok']}, bad={ucm_i['num_bad']})",
        f"- RSITMD images: **{kb_i['num_images']}** (ok={kb_i['num_ok']}, bad={kb_i['num_bad']})",
        f"- Hard SHA image matches: **{c['hard_image_sha_matches']}**",
        f"- Soft dhash matches: **{c['soft_image_dhash_matches']}**",
        f"- Exact caption overlaps: **{c['exact_caption_overlaps']}**",
        f"- Normalized caption overlaps: **{c['normalized_caption_overlaps']}**",
        f"- Hard matches on UCM test: **{c['hard_image_sha_matches_on_ucm_test']}**",
        "",
        "## UCM integrity",
        "",
        "```json",
        json.dumps(ucm_i, indent=2),
        "```",
        "",
        "## RSITMD integrity",
        "",
        "```json",
        json.dumps(kb_i, indent=2),
        "```",
        "",
        "## Leakage warnings",
        "",
    ]
    warnings = leak.get("warnings") or []
    if warnings:
        lines.extend(f"- {w}" for w in warnings)
    else:
        lines.append("- None")

    lines.extend(
        [
            "",
            "## Design notes",
            "",
            "- Hard leakage = identical file SHA-256 between UCM and RSITMD.",
            "- Soft leakage = difference-hash Hamming distance <= threshold.",
            "- Caption overlap uses raw exact match and normalized match.",
            "- This stage does **not** train models or run retrieval.",
            "",
            "## Pass / fail gates",
            "",
            f"- fail_on_hard_image_leakage: `{report.get('gates', {}).get('fail_on_hard_image_leakage')}`",
            f"- fail_on_caption_overlap: `{report.get('gates', {}).get('fail_on_caption_overlap')}`",
            f"- gate_status: **{report.get('gate_status', 'unknown')}**",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def build_master_report(
    *,
    cfg: dict[str, Any],
    ucm_integrity: dict[str, Any],
    kb_integrity: dict[str, Any],
    leakage: dict[str, Any],
    outputs: dict[str, str],
) -> dict[str, Any]:
    hard = leakage["counts"]["hard_image_sha_matches"]
    exact_caps = leakage["counts"]["exact_caption_overlaps"]
    fail_hard = bool(cfg["leakage"].get("fail_on_hard_image_leakage", False))
    fail_caps = bool(cfg["leakage"].get("fail_on_caption_overlap", False))

    gate_status = "PASS"
    if fail_hard and hard > 0:
        gate_status = "FAIL"
    if fail_caps and exact_caps > 0:
        gate_status = "FAIL"

    return {
        "generated_at_utc": _utc_now(),
        "config_path": cfg.get("_config_path", ""),
        "seed": cfg.get("seed"),
        "schema_version": cfg.get("schema_version"),
        "integrity": {"ucm": ucm_integrity, "rsitmd": kb_integrity},
        "leakage": leakage,
        "outputs": outputs,
        "gates": {
            "fail_on_hard_image_leakage": fail_hard,
            "fail_on_caption_overlap": fail_caps,
        },
        "gate_status": gate_status,
    }
