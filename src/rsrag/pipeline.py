"""Stage 0–1 orchestration: manifests + integrity + leakage audit."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from rsrag.audit.integrity import run_integrity_checks
from rsrag.audit.leakage import audit_leakage
from rsrag.audit.report import build_master_report, write_json, write_markdown_report
from rsrag.audit.smoke import select_smoke_samples, write_smoke_html
from rsrag.config import ensure_dirs, load_config
from rsrag.data.manifest import validate_manifest, write_manifest_csv, write_manifest_jsonl
from rsrag.data.rsitmd import load_rsitmd
from rsrag.data.splits import apply_splits, load_official_ucm_splits, split_counts
from rsrag.data.ucm import load_ucm_captions


def run_stage0_stage1(
    config_path: str | Path,
    *,
    path_overrides: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Build manifests and run the leakage audit."""
    cfg = load_config(config_path, path_overrides=path_overrides)
    ensure_dirs(cfg)

    processed = Path(cfg["paths"]["processed_dir"])
    reports = Path(cfg["paths"]["report_dir"])
    seed = int(cfg["seed"])

    ucm = load_ucm_captions(cfg)
    official = None
    if cfg["splits"].get("prefer_official_split_files", True):
        official = load_official_ucm_splits(Path(cfg["paths"]["ucm_root"]))
    ucm = apply_splits(
        ucm,
        seed=seed,
        train_ratio=float(cfg["splits"]["train_ratio"]),
        val_ratio=float(cfg["splits"]["val_ratio"]),
        test_ratio=float(cfg["splits"]["test_ratio"]),
        official=official,
    )

    kb = load_rsitmd(cfg)

    ucm_jsonl = processed / "ucm_captions_manifest.jsonl"
    kb_jsonl = processed / "rsitmd_manifest.jsonl"
    ucm_csv = processed / "ucm_captions_manifest.csv"
    kb_csv = processed / "rsitmd_manifest.csv"

    write_manifest_jsonl(ucm, ucm_jsonl)
    write_manifest_jsonl(kb, kb_jsonl)
    write_manifest_csv(ucm, ucm_csv)
    write_manifest_csv(kb, kb_csv)

    write_json(reports / "ucm_schema_validation.json", validate_manifest(ucm))
    write_json(reports / "rsitmd_schema_validation.json", validate_manifest(kb))

    ucm_integrity = run_integrity_checks(ucm, "ucm_captions")
    kb_integrity = run_integrity_checks(kb, "rsitmd")
    ucm_integrity["split_counts_recomputed"] = split_counts(ucm)
    ucm_integrity["used_official_splits"] = official is not None
    write_json(reports / "ucm_integrity.json", ucm_integrity)
    write_json(reports / "rsitmd_integrity.json", kb_integrity)

    leakage = audit_leakage(ucm, kb, cfg)
    write_json(reports / "leakage_audit.json", leakage)

    n = int(cfg["smoke"]["num_samples_per_source"])
    ucm_samples = select_smoke_samples(ucm, seed=seed, n=n)
    kb_samples = select_smoke_samples(kb, seed=seed, n=n)
    write_json(reports / "ucm_smoke_samples.json", {"samples": ucm_samples})
    write_json(reports / "rsitmd_smoke_samples.json", {"samples": kb_samples})
    if cfg["smoke"].get("write_html_preview", True):
        write_smoke_html(
            reports / "ucm_smoke_preview.html",
            title="UCM smoke samples",
            samples=ucm_samples,
        )
        write_smoke_html(
            reports / "rsitmd_smoke_preview.html",
            title="RSITMD smoke samples",
            samples=kb_samples,
        )

    outputs = {
        "ucm_manifest_jsonl": str(ucm_jsonl),
        "rsitmd_manifest_jsonl": str(kb_jsonl),
        "ucm_manifest_csv": str(ucm_csv),
        "rsitmd_manifest_csv": str(kb_csv),
        "ucm_integrity": str(reports / "ucm_integrity.json"),
        "rsitmd_integrity": str(reports / "rsitmd_integrity.json"),
        "leakage_audit": str(reports / "leakage_audit.json"),
        "stage0_stage1_report_md": str(reports / "stage0_stage1_report.md"),
        "stage0_stage1_summary": str(reports / "stage0_stage1_summary.json"),
    }

    master = build_master_report(
        cfg=cfg,
        ucm_integrity=ucm_integrity,
        kb_integrity=kb_integrity,
        leakage=leakage,
        outputs=outputs,
    )
    write_json(reports / "stage0_stage1_report.json", master)
    write_markdown_report(reports / "stage0_stage1_report.md", master)

    summary = {
        "gate_status": master["gate_status"],
        "ucm_images": ucm_integrity["num_images"],
        "rsitmd_images": kb_integrity["num_images"],
        "hard_sha_matches": leakage["counts"]["hard_image_sha_matches"],
        "exact_caption_overlaps": leakage["counts"]["exact_caption_overlaps"],
        "outputs": outputs,
    }
    write_json(reports / "stage0_stage1_summary.json", summary)
    return master
