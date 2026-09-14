# Remote Sensing RAG — Stage 0 & Stage 1

Deterministic **dataset manifests + leakage audit** for:

- **UCM-Captions** → caption train/val/test images
- **RSITMD** → external textual knowledge base

This repository stage does **not** implement CLIP, GPT-2, RAG decoding, or the five research extensions.

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

## Repository layout

```text
configs/
  stage0_stage1_ucm_rsitmd.yaml   # real-data config
  stage0_stage1_fixtures.yaml     # tiny synthetic config
src/rsrag/
  config.py / schema.py / hashing.py / textnorm.py / pipeline.py / cli.py
  data/   # UCM + RSITMD loaders, splits, manifest IO
  audit/  # integrity, leakage, smoke previews, reports
scripts/
  make_fixtures.py
  run_stage0_stage1.py
tests/
notebooks/
  stage0_stage1_colab.ipynb
data/raw/          # put real datasets here
data/processed/    # generated manifests
reports/           # generated audits
```

## Local setup

```bash
cd /Users/yakshinipatila/Projects/remote-sensing-rag
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

### Real datasets

1. Put UCM-Captions under `data/raw/ucm_captions/`
2. Put RSITMD under `data/raw/rsitmd/`
3. Run:

```bash
python scripts/run_stage0_stage1.py --config configs/stage0_stage1_ucm_rsitmd.yaml --print-summary
```

See `data/raw/README.md` for expected folder layouts.

## Outputs you should expect

Under `data/processed/` (or `data/processed/fixtures/`):

- `ucm_captions_manifest.jsonl`
- `rsitmd_manifest.jsonl`
- CSV companions for quick inspection

Under `reports/` (or `reports/fixtures/`):

- `ucm_integrity.json`
- `rsitmd_integrity.json`
- `leakage_audit.json`
- `stage0_stage1_report.md`
- `stage0_stage1_summary.json`
- optional HTML smoke previews

## Google Colab workflow

1. Upload/clone this repo into Drive
2. Open `notebooks/stage0_stage1_colab.ipynb`
3. Mount Drive
4. Install requirements
5. Either:
   - run fixtures smoke test, or
   - point config paths at Drive-extracted UCM/RSITMD
6. Copy only finished reports/manifest archives back to Drive

Do **not** train by streaming thousands of individual image files from Drive.

## What is intentionally missing

- CLIP embeddings
- FAISS/HNSW retrieval
- GPT-2 captioning
- Bitemp / SAR / Graph / HalluGuard / PEFT extensions

Those start only after Stage 0–1 manifests and leakage findings are accepted.
