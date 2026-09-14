# Remote Sensing RAG — Stages 0–2

Deterministic **dataset manifests + leakage audit** for:

- **UCM-Captions** → caption train/val/test images
- **RSITMD** → external textual knowledge base

Stage 2 adds **retrieval only**: frozen CLIP embeddings, an exact UCM→RSITMD
search reference, and an HNSW comparison. It does not implement GPT-2, RAG
decoding, caption training/generation, PCA, or the five research extensions.

**Portable by design:** code lives on GitHub. Real datasets live on Google Drive (or another disk you control). Committed configs use relative paths only. Override with environment variables or CLI flags. See [docs/PORTABILITY.md](docs/PORTABILITY.md).

## Why this stage exists

Before training any model we need:

1. A canonical, machine-readable inventory of every image + caption
2. Proof that files are readable and captions are present
3. Deterministic train/val/test assignment
4. An audit of UCM ↔ RSITMD leakage (shared images / captions)

If the knowledge base overlaps the test set, later “RAG improvements” are scientifically invalid.

## Design decisions (short)

| Decision | Why |
|---|---|
| One manifest row per image | Matches multi-reference caption metrics; avoids hashing the same image 5× |
| SHA-256 for hard duplicates | Exact byte identity; fully deterministic |
| Difference-hash for soft duplicates | Cheap near-duplicate signal without CLIP |
| Hash-based splits when no official file exists | Reproducible across machines; independent of filesystem order |
| RSITMD forced `split=kb` | Keeps knowledge-base role explicit |
| Report leakage instead of crashing by default | Discovery first; fail gates are opt-in |
| Path env vars / CLI overrides | Survives laptop return; Colab + Drive |

## Repository layout

```text
configs/          # portable YAML (no machine home paths)
docs/PORTABILITY.md
notebooks/stage0_stage1_colab.ipynb
src/rsrag/        # Stage 0–1 package
scripts/
tests/
data/raw/         # gitignored real data (put zips on Drive)
```

## Setup on any machine (including after this Mac is gone)

```bash
git clone https://github.com/yakshini12/remote-sensing-rag.git
cd remote-sensing-rag
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
pip install -e ".[dev]"
```

### Smoke test with synthetic fixtures (no dataset download)

```bash
python scripts/make_fixtures.py
python scripts/run_stage0_stage1.py --config configs/stage0_stage1_fixtures.yaml --print-summary
pytest -q
```

Expected fixture signals:

- `hard_sha_matches >= 1`
- `exact_caption_overlaps >= 1`
- `gate_status = PASS` (fail gates are off by default)

### Real datasets (not in Git)

Store extracts on Drive, then:

```bash
export RSRAG_UCM_ROOT=/content/drive/MyDrive/rsrag-data/ucm_captions
export RSRAG_RSITMD_ROOT=/content/drive/MyDrive/rsrag-data/rsitmd
export RSRAG_PROCESSED_DIR=/content/drive/MyDrive/rsrag-artifacts/processed
export RSRAG_REPORT_DIR=/content/drive/MyDrive/rsrag-artifacts/reports

python scripts/run_stage0_stage1.py --config configs/stage0_stage1_ucm_rsitmd.yaml --print-paths
python scripts/run_stage0_stage1.py --config configs/stage0_stage1_ucm_rsitmd.yaml --print-summary
```

See `data/raw/README.md` for Drive folder layout and download links.

## Google Colab

Open `notebooks/stage0_stage1_colab.ipynb`. It now covers the real Stage 1
audit followed by the approved Stage 2 retrieval-only run, with large artifacts
written to Drive. Do **not** commit datasets or generated embeddings.

## Stage 2 retrieval-only

After the real Stage 1 audit passes, install `requirements-stage2.txt` and run:

```bash
python scripts/run_stage2_retrieval.py --config configs/stage2_retrieval.yaml
```

The R0 checkpoint is explicitly recorded as OpenCLIP `ViT-B-32-quickgelu` /
`laion400m_e32`, our reproducible interpretation of the paper's under-specified
"CLIP-ViT-B/32 pretrained on LAION-400M." Generated embeddings and indexes are
ignored by Git. See [docs/STAGE2_RETRIEVAL.md](docs/STAGE2_RETRIEVAL.md).

## What is intentionally missing

- GPT-2 captioning
- RAG decoder integration
- caption training or generation
- PCA and diversity reranking
- Bitemp / SAR / Graph / HalluGuard / PEFT extensions

Those remain outside Stage 2.
