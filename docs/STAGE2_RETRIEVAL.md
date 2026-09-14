# Stage 2 — retrieval only

Stage 2 embeds every UCM image and every RSITMD caption with a frozen CLIP
model, performs exact normalized inner-product search, and compares optional
HNSW results to exact neighbors. It does not generate captions or train models.

## Baseline interpretation

KARG-RSIC specifies **CLIP ViT-B/32 pretrained on LAION-400M**, loaded through
Hugging Face, but does not give a repository, model ID, checkpoint hash, or
LAION training epoch. Those details are insufficient to identify one exact
weight file.

Experiment R0 uses the reproducible OpenCLIP checkpoint:

```text
architecture: ViT-B-32-quickgelu
pretrained tag: laion400m_e32
projected image/text embedding dimension: 512
```

This is our documented interpretation of the paper, not a claim that it is the
authors' unpublished implementation. The QuickGELU variant is required by this
specific pretrained tag; OpenCLIP warns that plain `ViT-B-32` would mismatch
the checkpoint's activation function. PCA and reranking are deliberately
absent from R0.

## Install

Use Python 3.10–3.12 (PyTorch is not available for every newer Python release):

```bash
python3.12 -m venv .venv-stage2
source .venv-stage2/bin/activate
pip install -r requirements.txt
pip install -r requirements-stage2.txt
pip install -e .
```

## Portable paths

The committed config contains only repository-relative defaults. On Colab,
place generated artifacts on Drive:

```bash
export RSRAG_UCM_MANIFEST=/content/drive/MyDrive/rsrag-artifacts/processed/ucm_captions_manifest.jsonl
export RSRAG_RSITMD_MANIFEST=/content/drive/MyDrive/rsrag-artifacts/processed/rsitmd_manifest.jsonl
export RSRAG_LEAKAGE_AUDIT=/content/drive/MyDrive/rsrag-artifacts/reports/leakage_audit.json
export RSRAG_ARTIFACT_ROOT=/content/drive/MyDrive/rsrag-artifacts/stage2/R0_clip_exact
python scripts/run_stage2_retrieval.py --config configs/stage2_retrieval.yaml
```

Equivalent CLI flags are `--ucm-manifest`, `--rsitmd-manifest`,
`--leakage-audit`, and `--artifact-root`.

## Retrieval definition

- Query: one normalized CLIP image projection per UCM image.
- Knowledge item: one normalized CLIP text projection per RSITMD caption.
- Score: exact cosine similarity, equal to inner product after L2 normalization.
- Determinism: manifest order is fixed; equal-score ties retain KB metadata order.
- k diagnostics: 1, 3, 5, and 10.
- No caption string is used in search.

The exact reference implementation is NumPy matrix multiplication plus stable
sorting. HNSW runs only afterward and is measured by neighbor overlap against
exact search. This ANN recall is not semantic relevance Recall@K.

## Metric limitation

Cross-dataset UCM→RSITMD Recall@K is undefined because there are no labeled
relevant RSITMD caption IDs for each UCM query. The paper's same-ID definition
does not apply to this external cross-dataset setup. Shared generic captions are
also not valid relevance labels.

The report therefore gives:

- cosine-score distributions;
- retrieved-caption and source-image diversity;
- exact-search latency;
- HNSW agreement, latency, and index size;
- dHash near-duplicate sensitivity counts;
- a deterministic 75-query HTML report for human relevance ratings.

Claims about genuine usefulness require completing that human review.

## Leakage auditability

Stage 2 recomputes all UCM↔RSITMD dHash pairs at the Stage 1 threshold and
stores the pair map. Every ranked result records whether its source image is a
flagged pair for that UCM query and its Hamming distance. Results are retained;
they are not silently filtered. Metrics can consequently be recomputed with
those queries excluded as a sensitivity analysis.

## Output and stop boundary

Generated data is stored under the configured artifact root and ignored by
Git. `experiment_log.json` records manifests and hashes, checkpoint,
normalization, index settings, k, seed, runtime, hardware, diagnostics,
limitations, and the next decision.

Stage 2 stops after retrieval evaluation. It contains no GPT-2, decoder,
caption generation, training, PCA, KARG reranking, LoRA/PEFT, verification, or
proposed extension.
