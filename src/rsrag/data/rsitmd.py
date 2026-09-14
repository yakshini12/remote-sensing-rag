"""RSITMD knowledge-base loader.

Design decision
---------------
RSITMD is treated as an *external knowledge base*, not as the caption
train/val/test set. Every RSITMD image gets split='kb'.

Accepted layouts
----------------
rsitmd_root/
  dataset_RSITMD.json   (or other candidate names from config)
  images/               (jpg/png/...)

Common JSON shapes (AMFMN-style and close variants):
- {"images": [{"filename": ..., "captions"|"sentences": [...], "split": ...}]}
- list of such image objects
- {"images":[...], "annotations":[...]} COCO-like
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from tqdm import tqdm

from rsrag.hashing import inspect_image
from rsrag.schema import SCHEMA_VERSION, ImageRecord


def _find_first(root: Path, candidates: list[str]) -> Path | None:
    for name in candidates:
        path = root / name
        if path.is_file():
            return path
    # Also allow nested common folders.
    for sub in ("", "annotations", "anno"):
        base = root / sub if sub else root
        for name in candidates:
            path = base / name
            if path.is_file():
                return path
    return None


def _iter_images(root: Path, extensions: set[str]) -> list[Path]:
    # Prefer an images/ subfolder if present.
    search_roots = [root / "images", root / "Images", root]
    files: list[Path] = []
    seen: set[Path] = set()
    for sroot in search_roots:
        if not sroot.is_dir():
            continue
        for p in sroot.rglob("*"):
            if not p.is_file() or p.suffix.lower() not in extensions:
                continue
            rp = p.resolve()
            if rp in seen:
                continue
            seen.add(rp)
            files.append(p)
        if files:
            break
    return sorted(files, key=lambda p: p.relative_to(root).as_posix().lower())


def _load_annotation_map(path: Path) -> dict[str, dict[str, Any]]:
    """Map filename/stem -> {captions: [...], class_name: str}."""
    data = json.loads(path.read_text(encoding="utf-8"))
    mapping: dict[str, dict[str, Any]] = {}

    def add(key: str, captions: list[str], class_name: str = "") -> None:
        key = key.replace("\\", "/").strip()
        if not key:
            return
        name = Path(key).name
        stem = Path(name).stem
        clean = [c.strip() for c in captions if str(c).strip()]
        payload = {"captions": clean, "class_name": class_name or ""}
        # filename and key are commonly the same value. Register each alias
        # once, but preserve repeated caption annotations from the dataset.
        for k in dict.fromkeys((name, stem, key)):
            if k not in mapping:
                mapping[k] = {"captions": [], "class_name": ""}
            mapping[k]["captions"].extend(clean)
            if class_name:
                mapping[k]["class_name"] = class_name

    def captions_from(item: dict[str, Any]) -> list[str]:
        caps = item.get("captions") or item.get("sentences") or item.get("raw") or []
        if isinstance(caps, str):
            caps = [caps]
        if caps and isinstance(caps[0], dict):
            caps = [c.get("raw") or c.get("caption") or c.get("sent") or "" for c in caps]
        return [str(c) for c in caps]

    if isinstance(data, list):
        for item in data:
            if not isinstance(item, dict):
                continue
            fname = (
                item.get("filename")
                or item.get("filepath")
                or item.get("image")
                or item.get("file_name")
            )
            if fname:
                add(str(fname), captions_from(item), str(item.get("class") or item.get("category") or ""))
        return mapping

    if isinstance(data, dict):
        if "images" in data and "annotations" in data:
            id_to_meta = {}
            for img in data["images"]:
                iid = img.get("id")
                fname = img.get("filename") or img.get("file_name") or img.get("filepath")
                if iid is not None and fname:
                    id_to_meta[iid] = {
                        "filename": str(fname),
                        "class_name": str(img.get("class") or img.get("category") or ""),
                    }
            caps_by_id: dict[Any, list[str]] = defaultdict(list)
            for ann in data["annotations"]:
                caps_by_id[ann.get("image_id")].append(str(ann.get("caption") or ann.get("raw") or ""))
            for iid, meta in id_to_meta.items():
                add(meta["filename"], caps_by_id.get(iid, []), meta["class_name"])
            return mapping

        images = data.get("images") or data.get("data") or []
        if isinstance(images, list):
            for item in images:
                if not isinstance(item, dict):
                    continue
                fname = (
                    item.get("filename")
                    or item.get("filepath")
                    or item.get("image")
                    or item.get("file_name")
                )
                if fname:
                    add(
                        str(fname),
                        captions_from(item),
                        str(item.get("class") or item.get("category") or ""),
                    )
            return mapping

    raise ValueError(f"Unsupported RSITMD annotation structure: {path}")


def _dedupe_preserve(seq: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in seq:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out


def load_rsitmd(cfg: dict[str, Any]) -> list[dict[str, Any]]:
    """Load RSITMD into canonical knowledge-base ImageRecord dicts."""
    root = Path(cfg["paths"]["rsitmd_root"])
    if not root.is_dir():
        raise FileNotFoundError(
            f"RSITMD root not found: {root}\n"
            "Place the dataset under data/raw/rsitmd/ "
            "(or update paths.rsitmd_root in the config)."
        )

    loaders = cfg["loaders"]
    hashing = cfg["hashing"]
    extensions = {e.lower() for e in loaders["image_extensions"]}

    ann_file = _find_first(root, loaders["rsitmd_annotation_candidates"])
    if ann_file is None:
        raise FileNotFoundError(
            f"No RSITMD annotation JSON found under {root}. "
            f"Tried: {loaders['rsitmd_annotation_candidates']}"
        )
    ann_map = _load_annotation_map(ann_file)

    images = _iter_images(root, extensions)
    if not images:
        raise FileNotFoundError(f"No RSITMD images found under {root}")

    records: list[dict[str, Any]] = []
    for path in tqdm(images, desc="RSITMD images", unit="img"):
        rel = path.relative_to(root).as_posix()
        name = path.name
        stem = path.stem
        meta_ann = ann_map.get(name) or ann_map.get(stem) or ann_map.get(rel) or {
            "captions": [],
            "class_name": "",
        }
        captions = [
            c.strip() for c in meta_ann.get("captions", []) if str(c).strip()
        ]

        meta = inspect_image(
            path,
            chunk_size=hashing["chunk_size_bytes"],
            hash_size=hashing["perceptual_hash_size"],
        )
        rec = ImageRecord(
            schema_version=cfg.get("schema_version", SCHEMA_VERSION),
            source="rsitmd",
            role="knowledge_base",
            image_id=f"rsitmd::{rel}",
            image_path=str(path.resolve()),
            relative_path=rel,
            split="kb",
            captions=captions,
            num_captions=len(captions),
            class_name=str(meta_ann.get("class_name") or ""),
            width=meta["width"],
            height=meta["height"],
            file_size_bytes=meta["file_size_bytes"],
            sha256=meta["sha256"],
            dhash64=meta["dhash64"],
            ok=bool(meta["ok"] and captions),
            error=meta["error"] if meta["error"] else ("" if captions else "missing_captions"),
        )
        records.append(rec.to_dict())

    return records
