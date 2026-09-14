"""Configuration loading utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def project_root_from(config_path: Path) -> Path:
    """Infer project root as the parent of the configs/ directory."""
    config_path = config_path.resolve()
    if config_path.parent.name == "configs":
        return config_path.parent.parent
    return Path.cwd().resolve()


def load_config(config_path: str | Path) -> dict[str, Any]:
    """Load YAML config and resolve path fields against the project root."""
    config_path = Path(config_path).expanduser().resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"Config not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    if not isinstance(cfg, dict):
        raise ValueError(f"Config must be a mapping: {config_path}")

    root = project_root_from(config_path)
    paths = cfg.setdefault("paths", {})
    paths["project_root"] = str(root)

    for key in ("ucm_root", "rsitmd_root", "processed_dir", "report_dir"):
        raw = Path(paths.get(key, ""))
        paths[key] = str(raw if raw.is_absolute() else (root / raw).resolve())

    cfg["_config_path"] = str(config_path)
    return cfg


def ensure_dirs(cfg: dict[str, Any]) -> None:
    """Create output directories used by Stage 0–1."""
    Path(cfg["paths"]["processed_dir"]).mkdir(parents=True, exist_ok=True)
    Path(cfg["paths"]["report_dir"]).mkdir(parents=True, exist_ok=True)
