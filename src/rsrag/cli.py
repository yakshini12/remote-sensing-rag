"""CLI entrypoint for Stage 0–1."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="rsrag-audit",
        description="Stage 0–1: build UCM+RSITMD manifests and run leakage audit.",
    )
    p.add_argument(
        "--config",
        type=str,
        default="configs/stage0_stage1_ucm_rsitmd.yaml",
        help="Path to YAML config.",
    )
    p.add_argument(
        "--print-summary",
        action="store_true",
        help="Print JSON summary to stdout.",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    from rsrag.pipeline import run_stage0_stage1

    report = run_stage0_stage1(Path(args.config))
    summary_path = Path(report["outputs"]["stage0_stage1_summary"])
    summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.is_file() else {
        "gate_status": report.get("gate_status")
    }

    print("Stage 0–1 complete.")
    print(f"Gate status: {report.get('gate_status')}")
    print(f"Report: {report['outputs']['stage0_stage1_report_md']}")
    if args.print_summary:
        print(json.dumps(summary, indent=2))
    return 0 if report.get("gate_status") == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
