# Implementation plan

## Audit findings (2026-10-28)

- The repository contains exploration notebooks only; there is no reusable training, evaluation, or inference package yet.
- `data/raw/NF-CSE-CIC-IDS2018-V2.parquet` (about 642 MB) and its feature dictionary are present and ignored by Git. The notebook reports 17,129,715 rows and 43 columns, but these figures and its data-quality conclusions have not yet been independently reproduced.
- The tracked exploration notebook has uncommitted edits. Preserve it; the untracked `Untitled.ipynb` and `jupyter_server_config.py` are also preserved.
- `requirements.txt` is a full pinned Jupyter environment rather than a minimal project dependency declaration. Existing split code in the notebook is in-memory and has not created persisted splits.
- Existing notebook feature exclusions (`SRC_TO_DST_SECOND_BYTES`, `DST_TO_SRC_SECOND_BYTES`, `FTP_COMMAND_RET_CODE`) are provisional; decide through measured diagnostics and train-only experimentation.

## Smallest sensible route

1. Establish ignored artifact/data paths and a lightweight, chunk-capable data inspection layer.
2. Verify schema, target/category frequencies, missing/nonfinite/constant values, and suspicious-field ranges in batches; estimate the memory and time cost before full training.
3. Implement reproducible binary and multiclass experiments with leakage-safe preprocessing and a fixed split strategy. Keep test evaluation as the final step.
4. Compare a simple baseline and one stronger tabular model, including imbalance-aware metrics and rare-class caveats.
5. Package prediction behind a small CLI, add focused tests, then document commands, decisions, measured results, and limitations.

## Current implementation checklist

- [x] Audit current work, Git state, dataset location, schema, and local docs.
- [x] Add bounded Arrow access, feature selector, and repeatable data audit.
- [x] Implement binary and multiclass sampled experiments with validation-based selection and held-out test metrics.
- [x] Persist task-specific model artifacts locally and metrics as JSON.
- [x] Add validated CLI inference and small-fixture end-to-end tests.
- [x] Re-run current trainer version to collect majority baseline and runtime measurements.
- [x] Run a validation-only duration ablation; its binary F1 was unchanged.
- [x] Read the associated NetFlow v2 paper and document its methodology and differences from the local files.
- [x] Document actual sample-based metrics, limitations, model settings, and inference commands.
- [ ] Test a monotonic transform for the extreme rate fields; their formulas remain unverified.
- [ ] Perform final verification and push the results/documentation update.

The first sampled train/test run completed. Its results and limitations are reported in `README.md`; generated models are ignored by Git.
