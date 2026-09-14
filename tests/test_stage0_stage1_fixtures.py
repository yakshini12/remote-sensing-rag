"""Stage 0–1 fixture tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from rsrag.pipeline import run_stage0_stage1


ROOT = Path(__file__).resolve().parents[1]
FIXTURE_CONFIG = ROOT / "configs" / "stage0_stage1_fixtures.yaml"


@pytest.fixture(scope="module")
def fixture_report():
    # Ensure fixtures exist.
    from scripts.make_fixtures import main as make_fixtures

    make_fixtures()
    assert FIXTURE_CONFIG.is_file()
    return run_stage0_stage1(FIXTURE_CONFIG)


def test_gate_pass_by_default(fixture_report):
    assert fixture_report["gate_status"] == "PASS"


def test_manifests_written(fixture_report):
    outs = fixture_report["outputs"]
    for key in ("ucm_manifest_jsonl", "rsitmd_manifest_jsonl"):
        path = Path(outs[key])
        assert path.is_file()
        lines = path.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) >= 3


def test_hard_image_leakage_detected(fixture_report):
    counts = fixture_report["leakage"]["counts"]
    assert counts["hard_image_sha_matches"] >= 1
    assert counts["hard_image_sha_matches_on_ucm_test"] >= 1


def test_exact_caption_overlap_detected(fixture_report):
    counts = fixture_report["leakage"]["counts"]
    assert counts["exact_caption_overlaps"] >= 1


def test_summary_json(fixture_report):
    summary_path = Path(fixture_report["outputs"]["stage0_stage1_summary"])
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["ucm_images"] >= 3
    assert summary["rsitmd_images"] >= 3
    assert "hard_sha_matches" in summary
