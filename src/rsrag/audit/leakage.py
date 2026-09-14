"""Cross-dataset leakage audit: UCM-Captions vs RSITMD knowledge base.

Design decisions
----------------
1. Hard image leakage = identical SHA-256.
2. Soft image leakage = dhash Hamming distance <= threshold.
3. Caption leakage = exact raw match and/or normalized match.
4. UCM-test vs KB overlap is reported separately because it is the
   most dangerous evaluation-contamination case.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from rsrag.hashing import hamming_hex64
from rsrag.textnorm import normalize_caption


def _caption_index(records: list[dict[str, Any]]) -> dict[str, list[str]]:
    idx: dict[str, list[str]] = defaultdict(list)
    for r in records:
        for cap in r.get("captions", []):
            idx[cap].append(r["image_id"])
    return idx


def _normalized_caption_index(records: list[dict[str, Any]]) -> dict[str, list[str]]:
    idx: dict[str, list[str]] = defaultdict(list)
    for r in records:
        for cap in r.get("captions", []):
            norm = normalize_caption(cap)
            if norm:
                idx[norm].append(r["image_id"])
    return idx


def audit_leakage(
    ucm_records: list[dict[str, Any]],
    kb_records: list[dict[str, Any]],
    cfg: dict[str, Any],
) -> dict[str, Any]:
    """Compare UCM caption dataset against RSITMD knowledge base."""
    thr = int(cfg["leakage"]["perceptual_hamming_threshold"])

    kb_by_sha: dict[str, list[str]] = defaultdict(list)
    kb_by_dhash: list[tuple[str, str]] = []
    for r in kb_records:
        if r.get("sha256"):
            kb_by_sha[r["sha256"]].append(r["image_id"])
        if r.get("dhash64"):
            kb_by_dhash.append((r["dhash64"], r["image_id"]))

    hard_matches: list[dict[str, Any]] = []
    soft_matches: list[dict[str, Any]] = []

    for r in ucm_records:
        sha = r.get("sha256")
        if sha and sha in kb_by_sha:
            hard_matches.append(
                {
                    "ucm_image_id": r["image_id"],
                    "ucm_split": r.get("split"),
                    "kb_image_ids": kb_by_sha[sha],
                    "sha256": sha,
                }
            )

    # O(n*m) dhash compare. Fine for Stage-1 / fixtures; may take minutes
    # on full UCM×RSITMD and that is acceptable as a one-time audit.
    hard_ucm_ids = {m["ucm_image_id"] for m in hard_matches}
    for r in ucm_records:
        dh = r.get("dhash64")
        if not dh:
            continue
        for kb_dh, kb_id in kb_by_dhash:
            dist = hamming_hex64(dh, kb_dh)
            if dist <= thr:
                soft_matches.append(
                    {
                        "ucm_image_id": r["image_id"],
                        "ucm_split": r.get("split"),
                        "kb_image_id": kb_id,
                        "hamming": dist,
                        "already_hard_match": r["image_id"] in hard_ucm_ids,
                    }
                )

    exact_overlaps: list[dict[str, Any]] = []
    norm_overlaps: list[dict[str, Any]] = []

    if cfg["leakage"].get("caption_exact_match", True):
        ucm_caps = _caption_index(ucm_records)
        kb_caps = _caption_index(kb_records)
        for cap, ucm_ids in ucm_caps.items():
            if cap in kb_caps:
                exact_overlaps.append(
                    {
                        "caption": cap,
                        "ucm_image_ids": ucm_ids[:20],
                        "kb_image_ids": kb_caps[cap][:20],
                    }
                )

    if cfg["leakage"].get("caption_normalized_match", True):
        ucm_norm = _normalized_caption_index(ucm_records)
        kb_norm = _normalized_caption_index(kb_records)
        for cap, ucm_ids in ucm_norm.items():
            if cap in kb_norm:
                norm_overlaps.append(
                    {
                        "normalized_caption": cap,
                        "ucm_image_ids": ucm_ids[:20],
                        "kb_image_ids": kb_norm[cap][:20],
                    }
                )

    test_hard = [m for m in hard_matches if m.get("ucm_split") == "test"]
    test_soft = [m for m in soft_matches if m.get("ucm_split") == "test"]

    warnings: list[str] = []
    if hard_matches:
        warnings.append("HARD leakage: identical image bytes found between UCM and RSITMD.")
    if test_hard:
        warnings.append("CRITICAL: UCM test images share SHA-256 with RSITMD KB entries.")
    if exact_overlaps:
        warnings.append("Caption text overlap exists between UCM and RSITMD (exact string).")

    return {
        "thresholds": {"perceptual_hamming_threshold": thr},
        "counts": {
            "hard_image_sha_matches": len(hard_matches),
            "soft_image_dhash_matches": len(soft_matches),
            "exact_caption_overlaps": len(exact_overlaps),
            "normalized_caption_overlaps": len(norm_overlaps),
            "hard_image_sha_matches_on_ucm_test": len(test_hard),
            "soft_image_dhash_matches_on_ucm_test": len(test_soft),
            "ucm_images": len(ucm_records),
            "ucm_test_images": sum(1 for r in ucm_records if r.get("split") == "test"),
            "kb_images": len(kb_records),
        },
        "hard_image_matches_preview": hard_matches[:50],
        "soft_image_matches_preview": soft_matches[:50],
        "exact_caption_overlaps_preview": exact_overlaps[:50],
        "normalized_caption_overlaps_preview": norm_overlaps[:50],
        "warnings": warnings,
    }
