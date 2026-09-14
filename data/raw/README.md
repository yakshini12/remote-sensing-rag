# Raw datasets are NOT in Git

Store UCM-Captions and RSITMD on **Google Drive** (or any durable disk).
This Mac may be returned; do not treat `~/...` as the dataset home.

## Recommended Google Drive layout

Keep **one archive plus one extracted folder** per dataset. Do not scatter
thousands of JPEGs in Drive root (Colab mount timeouts).

```text
MyDrive/
  rsrag-data/
    ucm_captions.zip          # original archive (keep)
    ucm_captions/             # extracted
      dataset.json            # or captions.json / ucm_captions.json
      agricultural/
      airport/
      ...
    rsitmd.zip
    rsitmd/
      dataset_RSITMD.json
      images/
  rsrag-artifacts/            # generated manifests + reports (safe to write)
    processed/
    reports/
```

## If you extract locally first

You may still use `data/raw/` inside a clone, but that folder is gitignored
and will vanish if the machine is wiped. Copy the **zip files** to Drive.

## Point the code at Drive (no Mac path required)

```bash
export RSRAG_UCM_ROOT=/content/drive/MyDrive/rsrag-data/ucm_captions
export RSRAG_RSITMD_ROOT=/content/drive/MyDrive/rsrag-data/rsitmd
export RSRAG_PROCESSED_DIR=/content/drive/MyDrive/rsrag-artifacts/processed
export RSRAG_REPORT_DIR=/content/drive/MyDrive/rsrag-artifacts/reports

python scripts/run_stage0_stage1.py \
  --config configs/stage0_stage1_ucm_rsitmd.yaml \
  --print-paths
```

Or pass `--ucm-root` / `--rsitmd-root` instead of env vars.

## Download sources (you download; we do not commit the files)

- UCM-Captions MEGA: https://mega.nz/folder/wCpSzSoS#RXzIlrv--TDt3ENZdKN8JA
- Listed from: https://github.com/201528014227051/RSICD_optimal
- RSITMD Google Drive: https://drive.google.com/file/d/1NJY86TAAUd8BVs7hyteImv8I2_Lh95W6/view
- RSITMD page: https://github.com/xiaoyuan1996/AMFMN/blob/master/RSITMD/README.md

Cite the original papers in the project report. Do not commit the images.
