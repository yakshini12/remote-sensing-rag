"""UCM-Captions dataset loader.

Accepted layouts (any one is enough)
------------------------------------
A) JSON captions + images
   ucm_root/
     dataset.json | captions.json | ...
     images/   OR class folders (agricultural/, airplane/, ...)

B) Sidecar .txt captions
   next to each image, or under captions/<stem>.txt

JSON shapes accepted:
- {"images": [{"filename"|"filepath"|"image": ..., "captions"|"sentences": [...]}]}
- [{"filename": ..., "captions": [...]}]
- RSICD-like {"images":[...], "annotations":[{"image_id":..., "caption":...}]}
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
    return None


def _iter_images(root: Path, extensions: set[str]) -> list[Path]:
    files = [
        p
        for p in root.rglob("*")
        if p.is_file()
        and p.suffix.lower() in extensions
        and "caption" not in {part.lower() for part in p.parts}
    ]
    # Deterministic order: sort by relative posix path.
    return sorted(files, key=lambda p: p.relative_to(root).as_posix().lower())


def _class_from_path(path: Path, root: Path) -> str:
    rel = path.relative_to(root)
    if len(rel.parts) >= 2:
        for part in rel.parts[:-1]:
            if part.lower() not in {"images", "image", "imgs", "img"}:
                return part
    return ""


def _load_json_captions(path: Path) -> dict[str, list[str]]:
    """Return mapping: image filename / stem / relative key -> captions."""
    data = json.loads(path.read_text(encoding="utf-8"))
    mapping: dict[str, list[str]] = defaultdict(list)

    def add(key: str, captions: list[str]) -> None:
        key = key.replace("\\", "/").strip()
        if not key:
            return
        name = Path(key).name
        stem = Path(name).stem
        clean = [c.strip() for c in captions if str(c).strip()]
        if not clean:
            return
        mapping[name].extend(clean)
        mapping[stem].extend(clean)
        mapping[key].extend(clean)

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
            caps = item.get("captions") or item.get("sentences") or item.get("raw") or []
            if isinstance(caps, str):
                caps = [caps]
            if caps and isinstance(caps[0], dict):
                caps = [c.get("raw") or c.get("caption") or "" for c in caps]
            if fname:
                add(str(fname), list(caps))
        return dict(mapping)

    if isinstance(data, dict):
        if "images" in data and "annotations" in data:
            id_to_file = {}
            for img in data["images"]:
                iid = img.get("id")
                fname = img.get("filename") or img.get("file_name") or img.get("filepath")
                if iid is not None and fname:
                    id_to_file[iid] = str(fname)
            for ann in data["annotations"]:
                fname = id_to_file.get(ann.get("image_id"))
                cap = ann.get("caption") or ann.get("raw") or ""
                if fname and cap:
                    add(fname, [str(cap)])
            return dict(mapping)

        if "images" in data:
            for item in data["images"]:
                fname = (
                    item.get("filename")
                    or item.get("filepath")
                    or item.get("image")
                    or item.get("file_name")
                )
                caps = item.get("captions") or item.get("sentences") or []
                if isinstance(caps, str):
                    caps = [caps]
                if caps and isinstance(caps[0], dict):
                    caps = [c.get("raw") or c.get("caption") or "" for c in caps]
                if fname:
                    add(str(fname), list(caps))
            return dict(mapping)

        for key, value in data.items():
            if isinstance(value, list):
                add(str(key), [str(v) for v in value])
            elif isinstance(value, str):
                add(str(key), [value])
        return dict(mapping)

    raise ValueError(f"Unsupported caption JSON structure: {path}")


def _dedupe_preserve(seq: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in seq:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out


def load_ucm_captions(cfg: dict[str, Any]) -> list[dict[str, Any]]:
    """Load UCM-Captions into canonical ImageRecord dicts."""
    root = Path(cfg["paths"]["ucm_root"])
    if not root.is_dir():
        raise FileNotFoundError(
            f"UCM root not found: {root}\n"
            "Place the dataset under data/raw/ucm_captions/ "
            "(or update paths.ucm_root in the config)."
        )

    loaders = cfg["loaders"]
    hashing = cfg["hashing"]
    extensions = {e.lower() for e in loaders["image_extensions"]}

    caption_file = _find_first(root, loaders["ucm_caption_candidates"])
    caption_map: dict[str, list[str]] = {}
    if caption_file is not None:
        caption_map = _load_json_captions(caption_file)

    images = _iter_images(root, extensions)
    if not images:
        raise FileNotFoundError(f"No images found under {root}")

    records: list[dict[str, Any]] = []
    for path in tqdm(images, desc="UCM images", unit="img"):
        rel = path.relative_to(root).as_posix()
        name = path.name
        stem = path.stem
        captions = (
            caption_map.get(name)
            or caption_map.get(stem)
            or caption_map.get(rel)
            or []
        )
        captions = _dedupe_preserve([c.strip() for c in captions if str(c).strip()])

        if not captions:
            for side in (path.with_suffix(".txt"), root / "captions" / f"{stem}.txt"):
                if side.is_file():
                    lines = [
                        ln.strip()
                        for ln in side.read_text(encoding="utf-8").splitlines()
                        if ln.strip()
                    ]
                    captions = _dedupe_preserve(lines)
                    break

        meta = inspect_image(
            path,
            chunk_size=hashing["chunk_size_bytes"],
            hash_size=hashing["perceptual_hash_size"],
        )
        rec = ImageRecord(
            schema_version=cfg.get("schema_version", SCHEMA_VERSION),
            source="ucm_captions",
            role="caption_dataset",
            image_id=f"ucm::{rel}",
            image_path=str(path.resolve()),
            relative_path=rel,
            split="unknown",
            captions=captions,
            num_captions=len(captions),
            class_name=_class_from_path(path, root),
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
