# Raw datasets live here (not committed).

## Expected layout

### UCM-Captions
Place under `data/raw/ucm_captions/`:

```text
data/raw/ucm_captions/
  dataset.json          # or captions.json / ucm_captions.json
  agricultural/*.jpg
  airport/*.jpg
  ...
  splits.json           # optional official splits
```

### RSITMD (knowledge base)
Place under `data/raw/rsitmd/`:

```text
data/raw/rsitmd/
  dataset_RSITMD.json
  images/*.jpg
```

Download links and licenses vary by source. Keep original archives in Drive/local
storage; extract into this folder before running Stage 0–1.

For a no-download smoke test, use:

```bash
python scripts/make_fixtures.py
python scripts/run_stage0_stage1.py --config configs/stage0_stage1_fixtures.yaml --print-summary
```
