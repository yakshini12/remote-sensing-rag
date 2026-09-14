# Generated artifacts (not stored in Git)

Stage 2 writes large, reproducible outputs below `artifacts/stage2/<experiment-id>/`:

```text
embeddings/  # normalized .npy matrices and row metadata
indexes/     # exact-search manifests, result arrays, HNSW binary
retrieval/   # auditable per-query ranked results and dHash flags
reports/     # metrics, experiment log, Markdown and qualitative HTML
```

Set `RSRAG_ARTIFACT_ROOT` to a durable Google Drive path in Colab. Git ignores
all generated contents; only this documentation is tracked.
