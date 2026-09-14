from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from rsrag.data.rsitmd import _load_annotation_map
from rsrag.data.splits import load_official_ucm_splits
from rsrag.data.ucm import _load_json_captions
from rsrag.stage2 import (
    build_caption_metadata,
    build_near_duplicate_pairs,
    exact_topk,
    materialize_results,
    summarize_exact,
)


def test_loads_official_splits_from_dataset_json(tmp_path: Path) -> None:
    (tmp_path / "dataset.json").write_text(
        json.dumps(
            {
                "images": [
                    {"filename": "1.tif", "split": "train"},
                    {"filename": "2.tif", "split": "val"},
                    {"filename": "3.tif", "split": "test"},
                ]
            }
        ),
        encoding="utf-8",
    )
    assert load_official_ucm_splits(tmp_path) == {
        "1": "train",
        "2": "val",
        "3": "test",
    }


def test_annotation_aliases_do_not_double_count_and_repeats_are_preserved(
    tmp_path: Path,
) -> None:
    payload = {
        "images": [
            {
                "filename": "1.tif",
                "sentences": [{"raw": "same"}, {"raw": "same"}, {"raw": "different"}],
            }
        ]
    }
    path = tmp_path / "dataset.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    assert _load_json_captions(path)["1.tif"] == ["same", "same", "different"]
    assert _load_annotation_map(path)["1.tif"]["captions"] == [
        "same",
        "same",
        "different",
    ]


def test_exact_search_is_ranked_and_deterministic() -> None:
    queries = np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    kb = np.asarray([[1.0, 0.0], [0.8, 0.6], [0.0, 1.0]], dtype=np.float32)
    first_indices, first_scores, _ = exact_topk(queries, kb, k=2, query_batch_size=1)
    second_indices, second_scores, _ = exact_topk(queries, kb, k=2, query_batch_size=2)
    assert first_indices.tolist() == [[0, 1], [2, 1]]
    np.testing.assert_array_equal(first_indices, second_indices)
    np.testing.assert_allclose(first_scores, second_scores)
    assert np.all(first_scores[:, :-1] >= first_scores[:, 1:])


def test_results_preserve_metadata_and_near_duplicate_flag() -> None:
    rsitmd = [
        {
            "image_id": "rsitmd::a",
            "relative_path": "images/a.tif",
            "captions": ["airport one", "airport two"],
            "dhash64": "0000000000000000",
            "sha256": "a",
        }
    ]
    captions = build_caption_metadata(rsitmd)
    assert [row["caption_id"] for row in captions] == [
        "rsitmd::a::caption::00",
        "rsitmd::a::caption::01",
    ]
    queries = [
        {
            "query_id": "ucm::q",
            "query_relative_path": "imgs/q.tif",
            "query_image_path": "/tmp/q.tif",
            "split": "test",
            "reference_captions": ["a query"],
            "query_dhash64": "0000000000000001",
            "query_sha256": "q",
        }
    ]
    pairs = build_near_duplicate_pairs(queries, captions, threshold=1)
    results = materialize_results(
        queries,
        captions,
        np.asarray([[0, 1]]),
        np.asarray([[0.9, 0.8]], dtype=np.float32),
        pairs,
    )
    assert results[0]["has_any_flagged_kb_near_duplicate"] is True
    assert results[0]["retrieved"][0]["source_image_id"] == "rsitmd::a"
    assert results[0]["retrieved"][0]["flagged_near_duplicate_pair"] is True
    assert results[0]["retrieved"][0]["near_duplicate_hamming"] == 1


def test_summary_does_not_claim_cross_dataset_recall() -> None:
    results = [
        {
            "has_any_flagged_kb_near_duplicate": False,
            "retrieved": [
                {
                    "similarity": 0.9,
                    "source_image_id": "a",
                    "caption": "an airport",
                    "flagged_near_duplicate_pair": False,
                },
                {
                    "similarity": 0.8,
                    "source_image_id": "b",
                    "caption": "a runway",
                    "flagged_near_duplicate_pair": False,
                },
            ],
        }
    ]
    summary = summarize_exact(results, [1, 2])
    assert summary["ground_truth"]["cross_dataset_recall_at_k_defined"] is False
    assert summary["by_k"]["2"]["mean_unique_source_image_ratio"] == 1.0
