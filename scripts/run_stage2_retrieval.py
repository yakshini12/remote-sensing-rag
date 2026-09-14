#!/usr/bin/env python3
"""Run Stage 2 frozen-CLIP retrieval only."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rsrag.stage2 import load_stage2_config, run_stage2


def parser() -> argparse.ArgumentParser:
    command = argparse.ArgumentParser(description="Stage 2: UCM -> RSITMD retrieval only")
    command.add_argument("--config", default="configs/stage2_retrieval.yaml")
    command.add_argument("--ucm-manifest", default="")
    command.add_argument("--rsitmd-manifest", default="")
    command.add_argument("--leakage-audit", default="")
    command.add_argument("--artifact-root", default="")
    command.add_argument("--print-paths", action="store_true")
    return command


def main() -> int:
    args = parser().parse_args()
    overrides = {
        key: value
        for key, value in {
            "ucm_manifest": args.ucm_manifest,
            "rsitmd_manifest": args.rsitmd_manifest,
            "leakage_audit": args.leakage_audit,
            "artifact_root": args.artifact_root,
        }.items()
        if value
    }
    if args.print_paths:
        cfg = load_stage2_config(args.config, path_overrides=overrides)
        print(json.dumps(cfg["paths"], indent=2))
        return 0
    result = run_stage2(args.config, path_overrides=overrides)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
