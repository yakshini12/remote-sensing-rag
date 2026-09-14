"""Path portability tests (no CLIP / training)."""

from __future__ import annotations

import os
from pathlib import Path

from rsrag.config import describe_paths, load_config


ROOT = Path(__file__).resolve().parents[1]
REAL_CONFIG = ROOT / "configs" / "stage0_stage1_ucm_rsitmd.yaml"


def test_committed_yaml_has_no_macos_home_path():
    text = REAL_CONFIG.read_text(encoding="utf-8")
    assert "/Users/" not in text
    assert "yakshinipatila" not in text


def test_default_paths_are_relative_to_repo(tmp_path, monkeypatch):
    monkeypatch.delenv("RSRAG_UCM_ROOT", raising=False)
    monkeypatch.delenv("RSRAG_RSITMD_ROOT", raising=False)
    monkeypatch.delenv("RSRAG_PROCESSED_DIR", raising=False)
    monkeypatch.delenv("RSRAG_REPORT_DIR", raising=False)
    monkeypatch.delenv("RSRAG_PROJECT_ROOT", raising=False)
    cfg = load_config(REAL_CONFIG)
    assert cfg["paths"]["ucm_root"] == str((ROOT / "data/raw/ucm_captions").resolve())
    assert cfg["paths"]["rsitmd_root"] == str((ROOT / "data/raw/rsitmd").resolve())


def test_env_vars_override_yaml(monkeypatch, tmp_path):
    ucm = tmp_path / "drive" / "ucm"
    kb = tmp_path / "drive" / "rsitmd"
    out = tmp_path / "artifacts"
    ucm.mkdir(parents=True)
    kb.mkdir(parents=True)
    monkeypatch.setenv("RSRAG_UCM_ROOT", str(ucm))
    monkeypatch.setenv("RSRAG_RSITMD_ROOT", str(kb))
    monkeypatch.setenv("RSRAG_PROCESSED_DIR", str(out / "processed"))
    monkeypatch.setenv("RSRAG_REPORT_DIR", str(out / "reports"))
    cfg = load_config(REAL_CONFIG)
    assert cfg["paths"]["ucm_root"] == str(ucm.resolve())
    assert cfg["paths"]["rsitmd_root"] == str(kb.resolve())
    assert cfg["paths"]["processed_dir"] == str((out / "processed").resolve())


def test_cli_overrides_win_over_env(monkeypatch, tmp_path):
    env_ucm = tmp_path / "env_ucm"
    cli_ucm = tmp_path / "cli_ucm"
    env_ucm.mkdir()
    cli_ucm.mkdir()
    monkeypatch.setenv("RSRAG_UCM_ROOT", str(env_ucm))
    cfg = load_config(
        REAL_CONFIG,
        path_overrides={"ucm_root": str(cli_ucm)},
    )
    assert cfg["paths"]["ucm_root"] == str(cli_ucm.resolve())


def test_describe_paths_reports_missing_real_data(monkeypatch):
    monkeypatch.delenv("RSRAG_UCM_ROOT", raising=False)
    monkeypatch.delenv("RSRAG_RSITMD_ROOT", raising=False)
    cfg = load_config(REAL_CONFIG)
    info = describe_paths(cfg)
    # Real datasets are not in Git; missing dirs are expected on a fresh clone.
    assert "ucm_root" in info
    assert "rsitmd_root" in info
    assert info["ucm_root"]["path"].endswith("ucm_captions")
