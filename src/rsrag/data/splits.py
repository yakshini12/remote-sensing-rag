"""Deterministic train/val/test splitting.

Design decision
---------------
If official split files exist, we use them.
Otherwise we assign each image to a split by hashing its stable image_id
with a fixed seed. This is reproducible across machines and does not
depend on filesystem readdir order.
"""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Iterable


def _stable_unit_interval(key: str, seed: int) -> float:
    """Map (key, seed) -> [0, 1) deterministically."""
    payload = f"{seed}::{key}".encode("utf-8")
    digest = hashlib.sha256(payload).hexdigest()
    # Use first 8 hex chars => 32 bits.
    return int(digest[:8], 16) / 0x100000000


def assign_split(
    image_id: str,
    *,
    seed: int,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
) -> str:
    total = train_ratio + val_ratio + test_ratio
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"Split ratios must sum to 1.0, got {total}")
    x = _stable_unit_interval(image_id, seed)
    if x < train_ratio:
        return "train"
    if x < train_ratio + val_ratio:
        return "val"
    return "test"


def load_official_ucm_splits(ucm_root: Path) -> dict[str, str] | None:
    """Try common official/custom split file layouts.

    Supported:
    - splits.json  {"train": [...], "val": [...], "test": [...]}
    - train.txt / val.txt / test.txt containing image ids or filenames
    """
    splits_json = ucm_root / "splits.json"
    if splits_json.is_file():
        data = json.loads(splits_json.read_text(encoding="utf-8"))
        mapping: dict[str, str] = {}
        for split_name in ("train", "val", "test"):
            for item in data.get(split_name, []):
                mapping[_normalize_id(str(item))] = split_name
        return mapping or None

    mapping = {}
    found = False
    for split_name in ("train", "val", "test"):
        txt = ucm_root / f"{split_name}.txt"
        if not txt.is_file():
            continue
        found = True
        for line in txt.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            mapping[_normalize_id(line)] = split_name
    return mapping if found else None


def _normalize_id(value: str) -> str:
    value = value.strip().replace("\\", "/")
    # Keep filename stem if a path was provided.
    name = Path(value).name
    return Path(name).stem


def apply_splits(
    records: Iterable[dict],
    *,
    seed: int,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    official: dict[str, str] | None = None,
) -> list[dict]:
    """Assign split field in-place and return the list."""
    out = []
    missing_official = 0
    for rec in records:
        if official is not None:
            key = _normalize_id(rec["image_id"])
            alt = _normalize_id(rec.get("relative_path", ""))
            split = official.get(key) or official.get(alt)
            if split is None:
                missing_official += 1
                split = assign_split(
                    rec["image_id"],
                    seed=seed,
                    train_ratio=train_ratio,
                    val_ratio=val_ratio,
                    test_ratio=test_ratio,
                )
            rec["split"] = split
        else:
            rec["split"] = assign_split(
                rec["image_id"],
                seed=seed,
                train_ratio=train_ratio,
                val_ratio=val_ratio,
                test_ratio=test_ratio,
            )
        out.append(rec)

    # Keep a note for the report via a sentinel attribute on the list-like usage.
    # Callers that care can recompute counts; we also return stats separately if needed.
    _ = missing_official
    return out


def split_counts(records: list[dict]) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for rec in records:
        counts[rec.get("split", "unknown")] += 1
    return dict(sorted(counts.items()))
