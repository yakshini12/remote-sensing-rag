"""Integrity checks over a built manifest."""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any


def run_integrity_checks(records: list[dict[str, Any]], source_name: str) -> dict[str, Any]:
    """Check missing files, empty captions, duplicate IDs/hashes within one source."""
    duplicate_ids = [
        image_id
        for image_id, n in Counter(r["image_id"] for r in records).items()
        if n > 1
    ]

    sha_groups: dict[str, list[str]] = defaultdict(list)
    for r in records:
        if r.get("sha256"):
            sha_groups[r["sha256"]].append(r["image_id"])
    duplicate_sha = {k: v for k, v in sha_groups.items() if len(v) > 1}

    missing_files = [r["image_id"] for r in records if r.get("error") == "file_missing"]
    unreadable = [
        {"image_id": r["image_id"], "error": r.get("error")}
        for r in records
        if (not r.get("ok")) and r.get("error") and r.get("error") != "file_missing"
    ]
    empty_captions = [
        r["image_id"]
        for r in records
        if not r.get("captions")
    ]

    caption_lengths = [len(c.split()) for r in records for c in r.get("captions", [])]
    avg_caption_len = (
        round(sum(caption_lengths) / len(caption_lengths), 3) if caption_lengths else 0.0
    )

    return {
        "source": source_name,
        "num_images": len(records),
        "num_ok": sum(1 for r in records if r.get("ok")),
        "num_bad": sum(1 for r in records if not r.get("ok")),
        "num_captions_total": sum(len(r.get("captions", [])) for r in records),
        "avg_caption_words": avg_caption_len,
        "split_counts": dict(Counter(r.get("split", "unknown") for r in records)),
        "duplicate_image_ids": duplicate_ids[:50],
        "num_duplicate_image_ids": len(duplicate_ids),
        "num_duplicate_sha_groups": len(duplicate_sha),
        "duplicate_sha_preview": {k: v for i, (k, v) in enumerate(duplicate_sha.items()) if i < 20},
        "missing_files": missing_files[:50],
        "num_missing_files": len(missing_files),
        "unreadable_or_empty_preview": unreadable[:50],
        "num_unreadable_or_flagged": len(unreadable),
        "empty_caption_images": empty_captions[:50],
        "num_empty_caption_images": len(empty_captions),
    }
