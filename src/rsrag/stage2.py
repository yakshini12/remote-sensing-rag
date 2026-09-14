"""Stage 2 retrieval-only pipeline.

R0 uses frozen CLIP image/text encoders and normalized inner-product search.
It does not train a model, generate captions, or use caption-string matching
as the retrieval mechanism.
"""

from __future__ import annotations

import base64
import hashlib
import html
import json
import os
import platform
import random
import statistics
import time
from collections import defaultdict
from io import BytesIO
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import yaml
from PIL import Image

from rsrag.data.manifest import read_manifest_jsonl
from rsrag.hashing import hamming_hex64
from rsrag.textnorm import normalize_caption

STAGE2_ENV_PATHS = {
    "RSRAG_UCM_MANIFEST": "ucm_manifest",
    "RSRAG_RSITMD_MANIFEST": "rsitmd_manifest",
    "RSRAG_LEAKAGE_AUDIT": "leakage_audit",
    "RSRAG_ARTIFACT_ROOT": "artifact_root",
}


def _resolve(raw: str, root: Path) -> Path:
    path = Path(raw).expanduser()
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def load_stage2_config(
    config_path: str | Path,
    *,
    path_overrides: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Load Stage 2 YAML with portable environment and CLI path overrides."""
    config_path = Path(config_path).expanduser().resolve()
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(cfg, dict):
        raise ValueError(f"Config must be a mapping: {config_path}")
    root = config_path.parent.parent if config_path.parent.name == "configs" else Path.cwd()
    root = Path(os.environ.get("RSRAG_PROJECT_ROOT", root)).expanduser().resolve()
    paths = cfg.setdefault("paths", {})
    for env_name, key in STAGE2_ENV_PATHS.items():
        if os.environ.get(env_name, "").strip():
            paths[key] = os.environ[env_name]
    for key, value in (path_overrides or {}).items():
        if key in STAGE2_ENV_PATHS.values() and value:
            paths[key] = value
    for key in STAGE2_ENV_PATHS.values():
        if not paths.get(key):
            raise ValueError(f"Missing Stage 2 path: paths.{key}")
        paths[key] = str(_resolve(str(paths[key]), root))
    cfg["_config_path"] = str(config_path)
    cfg["_project_root"] = str(root)
    return cfg


def _json_dump(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _jsonl_dump(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1_048_576), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_caption_metadata(rsitmd_records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Create one deterministic metadata row per RSITMD caption."""
    rows: list[dict[str, Any]] = []
    for image in rsitmd_records:
        for index, caption in enumerate(image.get("captions", [])):
            rows.append(
                {
                    "caption_id": f"{image['image_id']}::caption::{index:02d}",
                    "caption_index": index,
                    "source_image_id": image["image_id"],
                    "source_relative_path": image["relative_path"],
                    "caption": caption,
                    "source_image_dhash64": image.get("dhash64", ""),
                    "source_image_sha256": image.get("sha256", ""),
                }
            )
    return rows


def build_query_metadata(ucm_records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "query_id": image["image_id"],
            "query_relative_path": image["relative_path"],
            "query_image_path": image["image_path"],
            "split": image["split"],
            "reference_captions": image.get("captions", []),
            "query_dhash64": image.get("dhash64", ""),
            "query_sha256": image.get("sha256", ""),
        }
        for image in ucm_records
    ]


def _device(requested: str) -> str:
    import torch

    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _l2_normalize(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    if np.any(norms == 0):
        raise ValueError("Embedding matrix contains a zero vector")
    result = matrix / norms
    if not np.isfinite(result).all():
        raise ValueError("Embedding matrix contains non-finite values")
    return result.astype(np.float32, copy=False)


def encode_clip(
    cfg: dict[str, Any],
    queries: list[dict[str, Any]],
    captions: list[dict[str, Any]],
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Encode every UCM image and RSITMD caption using frozen OpenCLIP."""
    import open_clip
    import torch
    from torch.utils.data import DataLoader, Dataset

    clip_cfg = cfg["clip"]
    device = _device(str(clip_cfg.get("device", "auto")))
    seed = int(cfg["experiment"]["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    model, _, preprocess = open_clip.create_model_and_transforms(
        clip_cfg["model_name"],
        pretrained=clip_cfg["pretrained"],
        device=device,
    )
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    tokenizer = open_clip.get_tokenizer(clip_cfg["model_name"])

    class ImageDataset(Dataset):
        def __len__(self) -> int:
            return len(queries)

        def __getitem__(self, index: int) -> Any:
            with Image.open(queries[index]["query_image_path"]) as image:
                return preprocess(image.convert("RGB"))

    started = time.perf_counter()
    image_chunks: list[np.ndarray] = []
    loader = DataLoader(
        ImageDataset(),
        batch_size=int(clip_cfg["image_batch_size"]),
        shuffle=False,
        num_workers=int(clip_cfg.get("num_workers", 0)),
    )
    with torch.inference_mode():
        for batch in loader:
            vectors = model.encode_image(batch.to(device))
            image_chunks.append(vectors.float().cpu().numpy())
    image_seconds = time.perf_counter() - started

    started = time.perf_counter()
    text_chunks: list[np.ndarray] = []
    batch_size = int(clip_cfg["text_batch_size"])
    texts = [row["caption"] for row in captions]
    with torch.inference_mode():
        for start in range(0, len(texts), batch_size):
            tokens = tokenizer(texts[start : start + batch_size]).to(device)
            vectors = model.encode_text(tokens)
            text_chunks.append(vectors.float().cpu().numpy())
    text_seconds = time.perf_counter() - started

    images = _l2_normalize(np.concatenate(image_chunks))
    texts_matrix = _l2_normalize(np.concatenate(text_chunks))
    if images.shape[1] != texts_matrix.shape[1]:
        raise ValueError(f"CLIP image/text dimensions differ: {images.shape} vs {texts_matrix.shape}")
    expected = int(clip_cfg.get("expected_embedding_dimension", images.shape[1]))
    if images.shape[1] != expected:
        raise ValueError(f"Expected embedding dimension {expected}, got {images.shape[1]}")
    return images, texts_matrix, {
        "device": device,
        "embedding_dimension": int(images.shape[1]),
        "image_embedding_seconds": image_seconds,
        "text_embedding_seconds": text_seconds,
    }


def exact_topk(
    query_embeddings: np.ndarray,
    kb_embeddings: np.ndarray,
    *,
    k: int,
    query_batch_size: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    """Deterministic exact cosine search over normalized vectors.

    Stable argsort makes equal-score ties follow deterministic KB metadata order.
    """
    if k < 1 or k > len(kb_embeddings):
        raise ValueError(f"k must be in [1, {len(kb_embeddings)}], got {k}")
    all_indices: list[np.ndarray] = []
    all_scores: list[np.ndarray] = []
    latencies_ms: list[float] = []
    for start in range(0, len(query_embeddings), query_batch_size):
        query_batch = query_embeddings[start : start + query_batch_size]
        before = time.perf_counter()
        similarities = query_batch @ kb_embeddings.T
        order = np.argsort(-similarities, axis=1, kind="stable")[:, :k]
        scores = np.take_along_axis(similarities, order, axis=1)
        elapsed = time.perf_counter() - before
        latencies_ms.extend([elapsed * 1000 / len(query_batch)] * len(query_batch))
        all_indices.append(order)
        all_scores.append(scores)
    return (
        np.concatenate(all_indices),
        np.concatenate(all_scores),
        {
            "total_search_seconds": sum(latencies_ms) / 1000,
            "mean_query_latency_ms": statistics.fmean(latencies_ms),
            "p95_query_latency_ms": float(np.percentile(latencies_ms, 95)),
        },
    )


def build_near_duplicate_pairs(
    queries: list[dict[str, Any]],
    captions: list[dict[str, Any]],
    threshold: int,
) -> dict[str, dict[str, int]]:
    """Map query IDs to flagged RSITMD image IDs and dHash distances."""
    kb_hashes: dict[str, str] = {}
    for row in captions:
        kb_hashes.setdefault(row["source_image_id"], row.get("source_image_dhash64", ""))
    pairs: dict[str, dict[str, int]] = defaultdict(dict)
    for query in queries:
        qhash = query.get("query_dhash64", "")
        if not qhash:
            continue
        for kb_id, khash in kb_hashes.items():
            if khash:
                distance = hamming_hex64(qhash, khash)
                if distance <= threshold:
                    pairs[query["query_id"]][kb_id] = distance
    return dict(pairs)


def materialize_results(
    queries: list[dict[str, Any]],
    captions: list[dict[str, Any]],
    indices: np.ndarray,
    scores: np.ndarray,
    near_pairs: dict[str, dict[str, int]],
) -> list[dict[str, Any]]:
    rows = []
    for query, query_indices, query_scores in zip(queries, indices, scores, strict=True):
        flagged = near_pairs.get(query["query_id"], {})
        retrieved = []
        for rank, (caption_index, score) in enumerate(
            zip(query_indices.tolist(), query_scores.tolist(), strict=True), start=1
        ):
            metadata = captions[caption_index]
            source_id = metadata["source_image_id"]
            retrieved.append(
                {
                    "rank": rank,
                    "caption_id": metadata["caption_id"],
                    "source_image_id": source_id,
                    "source_relative_path": metadata["source_relative_path"],
                    "caption": metadata["caption"],
                    "similarity": float(score),
                    "flagged_near_duplicate_pair": source_id in flagged,
                    "near_duplicate_hamming": flagged.get(source_id),
                }
            )
        rows.append(
            {
                **query,
                "has_any_flagged_kb_near_duplicate": bool(flagged),
                "flagged_kb_near_duplicate_count": len(flagged),
                "retrieved": retrieved,
            }
        )
    return rows


def _tokens(text: str) -> set[str]:
    return set(normalize_caption(text).split())


def _pairwise_jaccard_distance(captions: list[str]) -> float | None:
    values = []
    token_sets = [_tokens(caption) for caption in captions]
    for left in range(len(token_sets)):
        for right in range(left + 1, len(token_sets)):
            union = token_sets[left] | token_sets[right]
            values.append(1 - len(token_sets[left] & token_sets[right]) / len(union) if union else 0)
    return statistics.fmean(values) if values else None


def summarize_exact(
    results: list[dict[str, Any]], k_values: list[int]
) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "ground_truth": {
            "cross_dataset_recall_at_k_defined": False,
            "reason": (
                "UCM queries and RSITMD captions have no labeled cross-dataset relevance "
                "or shared identity relation. Caption overlap is not used as ground truth."
            ),
        },
        "query_count": len(results),
        "queries_with_any_flagged_kb_near_duplicate": sum(
            bool(row["has_any_flagged_kb_near_duplicate"]) for row in results
        ),
        "by_k": {},
        "caption_overlap_contamination_diagnostic": {
            "note": (
                "Counts retrieved strings that exactly match a UCM reference. "
                "This is reported as contamination risk, never retrieval quality."
            ),
            "by_k": {},
        },
    }
    for k in k_values:
        subsets = [row["retrieved"][:k] for row in results]
        similarities = [hit["similarity"] for hits in subsets for hit in hits]
        top_scores = [hits[0]["similarity"] for hits in subsets]
        diversity = [
            value
            for hits in subsets
            if (value := _pairwise_jaccard_distance([hit["caption"] for hit in hits])) is not None
        ]
        summary["by_k"][str(k)] = {
            "mean_similarity_all_ranks": statistics.fmean(similarities),
            "mean_top1_similarity": statistics.fmean(top_scores),
            "p05_top1_similarity": float(np.percentile(top_scores, 5)),
            "p95_top1_similarity": float(np.percentile(top_scores, 95)),
            "mean_unique_source_image_ratio": statistics.fmean(
                len({hit["source_image_id"] for hit in hits}) / k for hits in subsets
            ),
            "mean_caption_jaccard_distance": statistics.fmean(diversity) if diversity else None,
            "queries_retrieving_flagged_near_duplicate": sum(
                any(hit["flagged_near_duplicate_pair"] for hit in hits) for hits in subsets
            ),
        }
        summary["caption_overlap_contamination_diagnostic"]["by_k"][str(k)] = {
            "queries_with_exact_reference_string": sum(
                any(
                    hit["caption"] in set(row.get("reference_captions", []))
                    for hit in row["retrieved"][:k]
                )
                for row in results
            )
        }
    clean_results = [
        row for row in results if not row["has_any_flagged_kb_near_duplicate"]
    ]
    sensitivity: dict[str, Any] = {
        "excluded_query_count": len(results) - len(clean_results),
        "remaining_query_count": len(clean_results),
        "by_k": {},
    }
    for k in k_values:
        subsets = [row["retrieved"][:k] for row in clean_results]
        similarities = [hit["similarity"] for hits in subsets for hit in hits]
        sensitivity["by_k"][str(k)] = {
            "mean_similarity_all_ranks": (
                statistics.fmean(similarities) if similarities else None
            ),
            "mean_unique_source_image_ratio": (
                statistics.fmean(
                    len({hit["source_image_id"] for hit in hits}) / k
                    for hits in subsets
                )
                if subsets
                else None
            ),
        }
    summary["sensitivity_excluding_queries_with_near_duplicates"] = sensitivity
    return summary


def run_hnsw(
    cfg: dict[str, Any],
    queries: np.ndarray,
    kb: np.ndarray,
    exact_indices: np.ndarray,
    exact_scores: np.ndarray,
) -> tuple[dict[str, Any], Any]:
    """Build HNSW and evaluate ANN overlap against exact top-k neighbors."""
    import hnswlib

    hcfg = cfg["hnsw"]
    index = hnswlib.Index(space=hcfg["space"], dim=kb.shape[1])
    before = time.perf_counter()
    index.init_index(
        max_elements=len(kb),
        ef_construction=int(hcfg["ef_construction"]),
        M=int(hcfg["m"]),
        random_seed=int(cfg["experiment"]["seed"]),
    )
    index.add_items(kb, np.arange(len(kb)), num_threads=1)
    build_seconds = time.perf_counter() - before
    index.set_ef(int(hcfg["ef_search"]))
    index.set_num_threads(1)
    before = time.perf_counter()
    labels, distances = index.knn_query(queries, k=exact_indices.shape[1], num_threads=1)
    search_seconds = time.perf_counter() - before
    by_k = {}
    for k in cfg["retrieval"]["k_values"]:
        per_query = [
            len(set(exact_indices[i, :k]) & set(labels[i, :k])) / k
            for i in range(len(queries))
        ]
        tie_aware = []
        for i in range(len(queries)):
            returned_scores = queries[i] @ kb[labels[i, :k]].T
            exact_boundary = exact_scores[i, k - 1]
            tie_aware.append(
                float(np.count_nonzero(returned_scores >= exact_boundary - 1e-6)) / k
            )
        by_k[str(k)] = {
            "mean_exact_neighbor_recall": statistics.fmean(per_query),
            "mean_tie_aware_exact_recall": statistics.fmean(tie_aware),
            "full_top_k_set_agreement_rate": statistics.fmean(value == 1 for value in per_query),
        }
    return {
        "build_seconds": build_seconds,
        "total_search_seconds": search_seconds,
        "mean_query_latency_ms": search_seconds * 1000 / len(queries),
        "parameters": hcfg,
        "by_k": by_k,
        "note": "This is ANN recall against exact vector search, not semantic Recall@K.",
    }, index


def _thumbnail_data_uri(image_path: str, size: int) -> str:
    with Image.open(image_path) as image:
        image = image.convert("RGB")
        image.thumbnail((size, size))
        output = BytesIO()
        image.save(output, "JPEG", quality=82)
    return "data:image/jpeg;base64," + base64.b64encode(output.getvalue()).decode("ascii")


def write_qualitative_html(
    path: Path,
    results: list[dict[str, Any]],
    *,
    seed: int,
    count: int,
    thumbnail_size: int,
) -> list[str]:
    """Write a deterministic, self-contained sample report and return sampled IDs."""
    rng = random.Random(seed)
    by_split: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for result in results:
        by_split[result["split"]].append(result)
    sampled: list[dict[str, Any]] = []
    split_names = [name for name in ("train", "val", "test") if by_split[name]]
    quotas = {name: count // len(split_names) for name in split_names}
    for name in split_names[: count % len(split_names)]:
        quotas[name] += 1
    for name in split_names:
        sampled.extend(rng.sample(by_split[name], min(quotas[name], len(by_split[name]))))
    sampled.sort(key=lambda row: (row["split"], row["query_id"]))

    cards = []
    for row in sampled:
        references = "".join(f"<li>{html.escape(text)}</li>" for text in row["reference_captions"])
        hits = "".join(
            "<tr>"
            f"<td>{hit['rank']}</td><td>{hit['similarity']:.4f}</td>"
            f"<td><code>{html.escape(hit['caption_id'])}</code></td>"
            f"<td>{html.escape(hit['caption'])}</td>"
            f"<td>{'YES' if hit['flagged_near_duplicate_pair'] else 'no'}</td>"
            "</tr>"
            for hit in row["retrieved"]
        )
        cards.append(
            "<section><div class='query'>"
            f"<img src='{_thumbnail_data_uri(row['query_image_path'], thumbnail_size)}'>"
            f"<div><h2>{html.escape(row['query_id'])} ({row['split']})</h2>"
            f"<p>Any flagged KB near-duplicate: <b>{row['has_any_flagged_kb_near_duplicate']}</b></p>"
            f"<h3>UCM references</h3><ul>{references}</ul></div></div>"
            "<table><thead><tr><th>Rank</th><th>Cosine</th><th>RSITMD caption ID</th>"
            f"<th>Retrieved caption</th><th>Flagged pair</th></tr></thead><tbody>{hits}</tbody></table>"
            "<p class='rating'>Human relevance (0–3): ___ &nbsp; Notes: ____________________</p></section>"
        )
    document = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>Stage 2 R0 qualitative retrieval</title>
<style>
body{{font:14px system-ui;margin:24px;max-width:1400px}} section{{border:1px solid #ccc;padding:16px;margin:20px 0}}
.query{{display:flex;gap:20px}} img{{width:{thumbnail_size}px;height:{thumbnail_size}px;object-fit:contain}}
table{{width:100%;border-collapse:collapse}} th,td{{border:1px solid #ddd;padding:6px;text-align:left}}
.rating{{background:#fff8d5;padding:8px}} code{{font-size:11px}}
</style></head><body><h1>Stage 2 R0 — UCM → RSITMD qualitative retrieval</h1>
<p>Frozen CLIP exact cosine retrieval. Caption strings were not used for search or relevance labels.</p>
{''.join(cards)}</body></html>"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(document, encoding="utf-8")
    return [row["query_id"] for row in sampled]


def _hardware(device: str) -> dict[str, Any]:
    details = {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "device": device,
    }
    try:
        import torch

        details["torch_version"] = torch.__version__
        if device == "cuda":
            details["gpu"] = torch.cuda.get_device_name(0)
        elif device == "mps":
            details["gpu"] = "Apple Metal Performance Shaders"
    except ImportError:
        pass
    return details


def run_stage2(
    config_path: str | Path,
    *,
    path_overrides: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Run R0 exact retrieval, then optional HNSW comparison, and stop."""
    started_at = time.time()
    cfg = load_stage2_config(config_path, path_overrides=path_overrides)
    paths = {key: Path(value) for key, value in cfg["paths"].items()}
    for key in ("ucm_manifest", "rsitmd_manifest", "leakage_audit"):
        if not paths[key].is_file():
            raise FileNotFoundError(f"{key} not found: {paths[key]}")
    artifact_root = paths["artifact_root"]
    embeddings_dir = artifact_root / "embeddings"
    indexes_dir = artifact_root / "indexes"
    retrieval_dir = artifact_root / "retrieval"
    reports_dir = artifact_root / "reports"
    for directory in (embeddings_dir, indexes_dir, retrieval_dir, reports_dir):
        directory.mkdir(parents=True, exist_ok=True)

    ucm_records = read_manifest_jsonl(paths["ucm_manifest"])
    rsitmd_records = read_manifest_jsonl(paths["rsitmd_manifest"])
    queries = build_query_metadata(ucm_records)
    captions = build_caption_metadata(rsitmd_records)
    if not queries or not captions:
        raise ValueError("Stage 2 requires non-empty UCM queries and RSITMD captions")
    _jsonl_dump(embeddings_dir / "ucm_image_metadata.jsonl", queries)
    _jsonl_dump(embeddings_dir / "rsitmd_caption_metadata.jsonl", captions)

    image_embeddings, text_embeddings, embedding_runtime = encode_clip(cfg, queries, captions)
    np.save(embeddings_dir / "ucm_image_embeddings.npy", image_embeddings)
    np.save(embeddings_dir / "rsitmd_text_embeddings.npy", text_embeddings)
    embedding_manifest = {
        "clip": cfg["clip"],
        "ucm_manifest_sha256": _sha256(paths["ucm_manifest"]),
        "rsitmd_manifest_sha256": _sha256(paths["rsitmd_manifest"]),
        "ucm_query_count": len(queries),
        "rsitmd_caption_count": len(captions),
        "embedding_dimension": int(image_embeddings.shape[1]),
        "dtype": str(image_embeddings.dtype),
        "normalized": True,
        "runtime": embedding_runtime,
    }
    _json_dump(embeddings_dir / "embedding_manifest.json", embedding_manifest)

    max_k = max(int(k) for k in cfg["retrieval"]["k_values"])
    indices, scores, exact_runtime = exact_topk(
        image_embeddings,
        text_embeddings,
        k=max_k,
        query_batch_size=int(cfg["retrieval"]["exact_query_batch_size"]),
    )
    np.save(indexes_dir / "exact_top_indices.npy", indices)
    np.save(indexes_dir / "exact_top_scores.npy", scores)
    _json_dump(
        indexes_dir / "exact_index_manifest.json",
        {
            "type": "numpy_exact_normalized_inner_product",
            "kb_embedding_file": "../embeddings/rsitmd_text_embeddings.npy",
            "kb_count": len(captions),
            "dimension": int(text_embeddings.shape[1]),
            "deterministic_tie_break": "stable KB metadata order",
        },
    )

    near_pairs = build_near_duplicate_pairs(
        queries, captions, int(cfg["leakage"]["perceptual_hamming_threshold"])
    )
    results = materialize_results(queries, captions, indices, scores, near_pairs)
    _jsonl_dump(retrieval_dir / "exact_top10_results.jsonl", results)
    _json_dump(
        retrieval_dir / "near_duplicate_pairs.json",
        {
            "threshold": cfg["leakage"]["perceptual_hamming_threshold"],
            "pairs_by_query": near_pairs,
        },
    )
    exact_summary = summarize_exact(results, [int(k) for k in cfg["retrieval"]["k_values"]])
    exact_summary["latency"] = exact_runtime
    _json_dump(reports_dir / "exact_retrieval_metrics.json", exact_summary)

    hnsw_summary: dict[str, Any] | None = None
    if cfg.get("hnsw", {}).get("enabled", False):
        hnsw_summary, hnsw_index = run_hnsw(
            cfg, image_embeddings, text_embeddings, indices, scores
        )
        hnsw_path = indexes_dir / "rsitmd_hnsw.bin"
        hnsw_index.save_index(str(hnsw_path))
        hnsw_summary["index_size_bytes"] = hnsw_path.stat().st_size
        _json_dump(reports_dir / "hnsw_vs_exact.json", hnsw_summary)

    qualitative_ids = write_qualitative_html(
        reports_dir / "qualitative_retrieval.html",
        results,
        seed=int(cfg["experiment"]["seed"]),
        count=int(cfg["qualitative"]["num_queries"]),
        thumbnail_size=int(cfg["qualitative"]["thumbnail_size"]),
    )
    near_pair_count = sum(len(matches) for matches in near_pairs.values())
    near_query_split_counts: dict[str, int] = defaultdict(int)
    for query in queries:
        if query["query_id"] in near_pairs:
            near_query_split_counts[query["split"]] += 1
    experiment = {
        "experiment_id": cfg["experiment"]["id"],
        "stage": "2_retrieval_only",
        "status": "PASS",
        "dataset_manifests": {
            "ucm": str(paths["ucm_manifest"]),
            "ucm_sha256": embedding_manifest["ucm_manifest_sha256"],
            "rsitmd": str(paths["rsitmd_manifest"]),
            "rsitmd_sha256": embedding_manifest["rsitmd_manifest_sha256"],
        },
        "clip_checkpoint": {
            "architecture": cfg["clip"]["model_name"],
            "pretrained": cfg["clip"]["pretrained"],
            "library": cfg["clip"]["library"],
            "paper_interpretation": (
                "KARG-RSIC states CLIP-ViT-B/32 pretrained on LAION-400M but gives no "
                "checkpoint ID. R0 uses OpenCLIP "
                f"{cfg['clip']['model_name']}/{cfg['clip']['pretrained']} explicitly."
            ),
        },
        "embedding_dimension": embedding_manifest["embedding_dimension"],
        "normalization": "L2; exact cosine equals inner product",
        "index_type": "exact normalized inner product; HNSW comparison enabled",
        "k": cfg["retrieval"]["k_values"],
        "random_seed": cfg["experiment"]["seed"],
        "runtime_seconds": time.time() - started_at,
        "hardware": _hardware(embedding_runtime["device"]),
        "retrieval_results": exact_summary,
        "hnsw_comparison": hnsw_summary,
        "qualitative_sample_query_ids": qualitative_ids,
        "qualitative_observations": (
            "Not auto-claimed. Complete human relevance ratings in qualitative_retrieval.html."
        ),
        "failure_cases": {
            "queries_with_any_flagged_kb_near_duplicate": len(near_pairs),
            "flagged_near_duplicate_pair_count": near_pair_count,
            "flagged_query_counts_by_split": dict(near_query_split_counts),
            "caption_overlap_not_used_for_retrieval": True,
        },
        "conclusion": (
            "Pipeline validity passed. Genuine usefulness requires human review because "
            "cross-dataset relevance labels do not exist."
        ),
        "next_decision": "Review and rate the 75-query qualitative sample; do not start a decoder.",
    }
    _json_dump(reports_dir / "experiment_log.json", experiment)

    report_lines = [
        "# Stage 2 R0 retrieval-only report",
        "",
        f"- Status: **{experiment['status']}**",
        f"- UCM queries embedded: **{len(queries)}**",
        f"- RSITMD captions embedded: **{len(captions)}**",
        f"- CLIP: `{cfg['clip']['model_name']}` / `{cfg['clip']['pretrained']}`",
        f"- Dimension: **{embedding_manifest['embedding_dimension']}**, L2 normalized",
        f"- Exact mean query latency: **{exact_runtime['mean_query_latency_ms']:.3f} ms**",
        f"- dHash-flagged query/image pairs: **{near_pair_count}** across "
        f"**{len(near_pairs)}** UCM queries",
        "",
        "## Metric validity",
        "",
        "Cross-dataset semantic Recall@K is **undefined**: UCM→RSITMD has no labeled",
        "relevance relation. Shared caption strings are not used as search or ground truth.",
        "HNSW recall below means agreement with exact vector neighbors only.",
        "",
        "## Exact-search diagnostics",
        "",
    ]
    for k, values in exact_summary["by_k"].items():
        report_lines.append(
            f"- k={k}: mean similarity={values['mean_similarity_all_ranks']:.4f}, "
            f"unique-source ratio={values['mean_unique_source_image_ratio']:.4f}, "
            f"caption diversity={values['mean_caption_jaccard_distance']}"
        )
    if hnsw_summary:
        report_lines.extend(["", "## HNSW versus exact", ""])
        for k, values in hnsw_summary["by_k"].items():
            report_lines.append(
                f"- k={k}: exact-neighbor recall={values['mean_exact_neighbor_recall']:.4f}, "
                f"tie-aware recall={values['mean_tie_aware_exact_recall']:.4f}, "
                f"full-set agreement={values['full_top_k_set_agreement_rate']:.4f}"
            )
    report_lines.extend(
        [
            "",
            "## Required human inspection",
            "",
            "Open `qualitative_retrieval.html` and rate the sampled queries. The pipeline",
            "does not manufacture semantic relevance labels or claim usefulness from cosine",
            "scores alone.",
            "",
            "## Stop boundary",
            "",
            "No GPT-2, caption generation, decoder integration, training, PCA, reranking,",
            "or proposed extension is implemented in this experiment.",
        ]
    )
    (reports_dir / "stage2_report.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    return experiment
