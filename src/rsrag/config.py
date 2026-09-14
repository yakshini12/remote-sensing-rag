"""Configuration loading utilities.

Path resolution order (later wins)
----------------------------------
1. YAML `paths:` in the chosen config file (relative paths are portable)
2. Optional `configs/local.yaml` overlay (gitignored; machine-specific)
3. Environment variables
4. Explicit CLI / function overrides

No Mac-specific absolute paths belong in committed files.
Real datasets must live outside Git (Google Drive, another laptop, Colab `/content`).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Mapping

import yaml

PATH_KEYS = ("ucm_root", "rsitmd_root", "processed_dir", "report_dir")

ENV_PATH_MAP = {
    "RSRAG_UCM_ROOT": "ucm_root",
    "RSRAG_RSITMD_ROOT": "rsitmd_root",
    "RSRAG_PROCESSED_DIR": "processed_dir",
    "RSRAG_REPORT_DIR": "report_dir",
}


def project_root_from(config_path: Path) -> Path:
    """Infer project root as the parent of the configs/ directory, or CWD."""
    config_path = config_path.resolve()
    env_root = os.environ.get("RSRAG_PROJECT_ROOT", "").strip()
    if env_root:
        return Path(env_root).expanduser().resolve()
    if config_path.parent.name == "configs":
        return config_path.parent.parent
    return Path.cwd().resolve()


def _deep_update(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for key, value in overlay.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_update(out[key], value)
        else:
            out[key] = value
    return out


def _read_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(f"Config must be a mapping: {path}")
    return data


def _apply_env_path_overrides(paths: dict[str, Any]) -> None:
    for env_name, key in ENV_PATH_MAP.items():
        value = os.environ.get(env_name, "").strip()
        if value:
            paths[key] = value


def _resolve_path(raw: str | Path, root: Path) -> Path:
    path = Path(str(raw)).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (root / path).resolve()


def load_config(
    config_path: str | Path,
    *,
    path_overrides: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Load YAML config and resolve portable path fields.

    Parameters
    ----------
    config_path:
        Committed YAML, typically `configs/stage0_stage1_ucm_rsitmd.yaml`.
    path_overrides:
        Optional mapping of path keys to strings (from CLI). Highest priority.
    """
    config_path = Path(config_path).expanduser().resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"Config not found: {config_path}")

    cfg = _read_yaml(config_path)

    local_yaml = config_path.parent / "local.yaml"
    if local_yaml.is_file():
        cfg = _deep_update(cfg, _read_yaml(local_yaml))

    root = project_root_from(config_path)
    paths = cfg.setdefault("paths", {})
    paths["project_root"] = str(root)

    _apply_env_path_overrides(paths)

    if path_overrides:
        for key, value in path_overrides.items():
            if key in PATH_KEYS and value:
                paths[key] = value

    for key in PATH_KEYS:
        raw = paths.get(key, "")
        if raw in (None, ""):
            raise ValueError(f"Missing required path: paths.{key}")
        paths[key] = str(_resolve_path(raw, root))

    cfg["_config_path"] = str(config_path)
    return cfg


def describe_paths(cfg: dict[str, Any]) -> dict[str, Any]:
    """Return resolved paths plus existence flags (no I/O beyond stat)."""
    paths = cfg["paths"]
    described = {}
    for key in ("project_root",) + PATH_KEYS:
        path = Path(paths[key])
        described[key] = {
            "path": str(path),
            "exists": path.exists(),
            "is_dir": path.is_dir(),
        }
    return described


def ensure_dirs(cfg: dict[str, Any]) -> None:
    """Create output directories used by Stage 0–1."""
    Path(cfg["paths"]["processed_dir"]).mkdir(parents=True, exist_ok=True)
    Path(cfg["paths"]["report_dir"]).mkdir(parents=True, exist_ok=True)
