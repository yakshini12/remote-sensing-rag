# Stage 2 R0 results — frozen CLIP exact retrieval

Run date: 2026-09-14

## Reproducibility record

- UCM manifest SHA-256: `3623642f78b8b7eea47ade0b6a848165e5abd4df27c71cb9a7120109de1198cb`
- RSITMD manifest SHA-256: `86125fa53e81b523339df7df92f8147e86cf009abce10a15ca6ea2e9db1ef77e`
- UCM queries: 2,100 (all embedded)
- RSITMD captions: 23,715 (all five annotation slots for all 4,743 images)
- Model: OpenCLIP `ViT-B-32-quickgelu`, pretrained tag `laion400m_e32`
- Projected embedding dimension: 512
- Normalization: L2
- Reference index: exact NumPy inner product over normalized embeddings
- Hardware: Apple Metal Performance Shaders, PyTorch 2.14.0
- Total experiment runtime: 214.8 seconds
- Repeatability check: exact rankings were identical across query batchings;
  two single-threaded HNSW rebuilds with seed 42 returned identical top-10 IDs.

The KARG-RSIC paper names CLIP ViT-B/32 with LAION-400M pretraining but does
not identify a checkpoint. This recorded OpenCLIP checkpoint is our
reproducible interpretation, not a claim about unpublished author code.

## Exact-search diagnostics

- k=1: mean similarity 0.3350; unique source-image ratio 1.0000.
- k=3: mean similarity 0.3292; unique source-image ratio 0.8832;
  mean caption Jaccard distance 0.7745.
- k=5: mean similarity 0.3255; unique source-image ratio 0.8368;
  mean caption Jaccard distance 0.7930.
- k=10: mean similarity 0.3197; unique source-image ratio 0.7820;
  mean caption Jaccard distance 0.8207.

Exact search took 3.94 seconds for all 2,100 queries (mean 1.877 ms/query on
this run). These similarity and diversity values are diagnostics, not semantic
relevance scores.

## Metric validity

Semantic Recall@K is **not defined** for UCM→RSITMD: no ground-truth mapping
labels which RSITMD caption IDs are relevant to each UCM image. The same-ID
Recall@K definition discussed in KARG-RSIC does not apply across these two
datasets. Shared caption strings are not used for retrieval or as positive
labels.

As a contamination diagnostic only, an exact UCM reference string appeared
within the retrieved set for 18 queries at k=1, 25 at k=3, 34 at k=5, and 51
at k=10. These counts are explicitly **not evidence of retrieval quality**.

## Stage 1 near-duplicate sensitivity

The 87 flagged dHash pairs involve 70 UCM queries:

- train: 53 pairs across 51 queries;
- validation: 24 pairs across 10 queries;
- test: 10 pairs across 9 queries.

At k=10, 14 queries actually retrieved a flagged paired RSITMD image. Excluding
all 70 affected queries changed mean k=10 similarity from 0.31975 to 0.31922,
so this aggregate diagnostic is not driven by those cases. The complete pair
map and per-result flags remain in the ignored experiment artifacts.

## HNSW versus exact

Initial HNSW settings were M=32, efConstruction=200, efSearch=100.

- k=1: caption-ID recall 0.8795; tie-aware recall 0.8800; full-set
  agreement 0.8795.
- k=3: caption-ID recall 0.8770; tie-aware recall 0.8789; full-set
  agreement 0.7552.
- k=5: caption-ID recall 0.8814; tie-aware recall 0.8841; full-set
  agreement 0.6690.
- k=10: caption-ID recall 0.8850; tie-aware recall 0.8936; full-set
  agreement 0.4995.

HNSW built in 14.50 seconds, searched in 0.775 seconds (0.369 ms/query), and
occupied 55,115,388 bytes. It is faster than exact search but its initial
agreement is insufficient, so it must **not** replace the exact reference.
Further HNSW tuning would be a separate controlled retrieval experiment.

## Qualitative observations

The generated self-contained HTML report contains a deterministic, split-aware
sample of 75 UCM queries with references, top-10 RSITMD captions, cosine scores,
caption/image IDs, and near-duplicate flags.

A preliminary spot check found clearly useful scene-level retrievals for
intersection, parking-lot, tennis-court, harbor, and storage-tank examples.
It also found important failures: one harbor query retrieved airport/factory
text; some farmland queries confused farmland, desert, meadow, and generic
texture descriptions; and repeated/mislabeled RSITMD source items can create
redundant results. Exact shared template captions occasionally rank first and
must not be counted as semantic evidence.

This is evidence that direct CLIP retrieval is sometimes useful, but not enough
to claim reliable external knowledge retrieval. Human ratings of the 75-query
report are still required before selecting k or proceeding to captioning.

## Decision and stop

R0's retrieval pipeline passes its engineering/audit criteria. Exact search
remains the reference. No k is selected yet; increasing k improves textual
diversity but lowers source-image diversity and average similarity.

Stage 2 stops here. No PCA, reranking, GPT-2, decoder integration, caption
generation, training, or proposed extension was started.
