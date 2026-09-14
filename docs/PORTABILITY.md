# Portability: GitHub + Google Drive + Colab

The company Mac is not the source of truth.

| What | Where it lives |
|---|---|
| Code, configs, tests, docs | GitHub `yakshini12/remote-sensing-rag` |
| Real UCM-Captions + RSITMD | Google Drive (or any durable disk you control) |
| Generated manifests/reports | Drive `rsrag-artifacts/` (or `/content` then copy) |
| Secrets / tokens | Never in Git |

Stage 1 only. No CLIP / GPT-2 / RAG training in this checkout.

## After this Mac is returned

On a new machine or Colab:

```bash
git clone https://github.com/yakshini12/remote-sensing-rag.git
cd remote-sensing-rag
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install -e ".[dev]"
```

Synthetic smoke test (no real datasets):

```bash
python scripts/make_fixtures.py
python scripts/run_stage0_stage1.py --config configs/stage0_stage1_fixtures.yaml --print-summary
pytest -q
```

## Point at datasets on Google Drive

Committed YAML uses **relative** paths only (`data/raw/...`).
Override them:

1. Environment variables (`RSRAG_UCM_ROOT`, `RSRAG_RSITMD_ROOT`, …)
2. CLI flags (`--ucm-root`, `--rsitmd-root`, …)
3. Untracked `configs/local.yaml` copied from `configs/local.yaml.example`

Priority: CLI > environment > `local.yaml` > committed YAML.

Check resolution without running the audit:

```bash
python scripts/run_stage0_stage1.py --config configs/stage0_stage1_ucm_rsitmd.yaml --print-paths
```

## Colab

Open `notebooks/stage0_stage1_colab.ipynb`.

Private GitHub clone needs a token **you** create; do not paste tokens into the repo.
Prefer making the repo public if Colab clone friction is a problem.

Write manifests to Drive (`RSRAG_PROCESSED_DIR` / `RSRAG_REPORT_DIR`) so a
runtime disconnect does not delete Stage 1 outputs.

Unzip dataset archives **on the Colab VM local disk** (`/content`) if Drive
file-count limits become a problem, but keep the zips on Drive permanently.
