"""Deterministic hashing helpers.

Design decisions
----------------
1. Exact file identity uses SHA-256 over raw bytes.
   Same bytes => same hash on every machine.

2. Near-duplicate detection uses difference hash (dhash).
   We deliberately do NOT use deep embeddings here (no CLIP yet).
   dhash is cheap, deterministic given Pillow + imagehash versions,
   and good enough for a Stage-1 smoke audit.

3. We store dhash as a zero-padded 16-char hex string (64 bits).
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import imagehash
from PIL import Image, UnidentifiedImageError


def sha256_file(path: Path, chunk_size: int = 1_048_576) -> str:
    """Compute SHA-256 of a file in streaming chunks."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def dhash_hex(path: Path, hash_size: int = 8) -> tuple[str, int, int]:
    """Return (dhash_hex, width, height) for an image file.

    Raises OSError / UnidentifiedImageError on unreadable images.
    """
    with Image.open(path) as im:
        im = im.convert("RGB")
        width, height = im.size
        digest = imagehash.dhash(im, hash_size=hash_size)
    # imagehash returns an ImageHash object; str(digest) is hex.
    hex_str = str(digest).lower().zfill(16)
    return hex_str, width, height


def hamming_hex64(a: str, b: str) -> int:
    """Hamming distance between two 64-bit hex digests."""
    if not a or not b:
        return 64
    return (int(a, 16) ^ int(b, 16)).bit_count()


def inspect_image(
    path: Path,
    *,
    chunk_size: int = 1_048_576,
    hash_size: int = 8,
) -> dict:
    """Collect file size, sha256, dhash and geometry for one image."""
    out = {
        "file_size_bytes": path.stat().st_size if path.exists() else 0,
        "sha256": "",
        "dhash64": "",
        "width": 0,
        "height": 0,
        "ok": False,
        "error": "",
    }
    if not path.is_file():
        out["error"] = "file_missing"
        return out
    try:
        out["sha256"] = sha256_file(path, chunk_size=chunk_size)
        dhash, w, h = dhash_hex(path, hash_size=hash_size)
        out["dhash64"] = dhash
        out["width"] = w
        out["height"] = h
        out["ok"] = True
    except (OSError, UnidentifiedImageError, ValueError) as exc:
        out["error"] = f"{type(exc).__name__}: {exc}"
    return out
