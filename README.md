# Flow-Based Network Intrusion Detection

An undergraduate project to detect malicious network flows with the NF-CSE-CIC-IDS2018 V2 dataset. The system predicts from flow records; it does not inspect raw packets or capture live traffic.

## Status

Repository audit, bounded-memory dataset validation, sampled training, held-out evaluation, and local inference are implemented. The measured sample-based results are summarized below; they are not full-data training results. See [IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md), [docs/data-audit.md](docs/data-audit.md), [docs/research_basis.md](docs/research_basis.md), and [docs/architecture.md](docs/architecture.md).

## Dataset setup

Place the dataset at `data/raw/NF-CSE-CIC-IDS2018-V2.parquet`. The feature dictionary can be placed beside it as `data/raw/NetFlow v2 Features.csv`. Neither file should be committed. The current local dataset has 17,129,715 rows and 43 columns.

## Environment and data audit

Create a Python 3.11+ environment and install the project's minimal runtime dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-project.txt
PYTHONPATH=src python scripts/audit_data.py
PYTHONPATH=src python -m unittest discover -s tests -v
```

The audit iterates through Arrow batches; it does not load the complete dataset into a pandas DataFrame. The notebook files are exploratory material, not the reproducible training interface.

## Training

Train either task independently, or both in sequence:

```bash
PYTHONPATH=src python scripts/train.py --task binary --seed 43
PYTHONPATH=src python scripts/train.py --task multiclass --seed 43
# or --task all
```

The trainer scans the Parquet file in Arrow batches and keeps at most 125,000 benign plus 17,000 attack samples for binary classification. Multiclass sampling keeps at most 12,000 rows per category and retains every smaller class in full. Each sample is split into 70% train, 15% validation, and 15% test with seed 42. The validation set selects between majority-class, logistic-regression, and histogram-gradient-boosting models. Class-balanced sample weights are applied only during model fitting; source class counts reweight metrics to reflect the source prevalence. The test partition is evaluated once after selection.

Models are written to ignored local files under `models/`. Metric JSON files are written to `reports/experiments/`. Reproduce the validation-only duration ablation with `PYTHONPATH=src python scripts/train.py --task binary --drop-duration --validation-only --run-suffix without_duration --seed 43`; it intentionally does not inspect test metrics.

## Inference

Provide one JSON object containing every feature named in the saved model artifact. The tool checks required names, numeric types, finiteness, and float32 representability, then returns the task, predicted label, per-class probabilities, and maximum probability:

```bash
PYTHONPATH=src python scripts/predict.py --model models/binary_model.joblib --input flow.json
PYTHONPATH=src python scripts/predict.py --model models/multiclass_model.joblib --input flow.json
```

The saved artifact fixes the feature order and includes the fitted estimator. Binary `Label` and multiclass `Attack` predictions use separate artifacts.

## Current sampled experiment (seed 43)

These runs use the paper-aligned feature exclusions, 142,000 binary and 135,126 multiclass sampled rows, and stratified 70/15/15 train/validation/test partitions. The training sample is not the entire dataset. Validation selection scores and the final selected model's test metrics are source-frequency weighted to correct the capped class sampling.

| Task | Validation F1: majority / logistic / HGB | Selected model | Test accuracy | Test F1 | Test precision / recall | ROC-AUC / PR-AUC / FPR |
|---|---|---|---:|---:|---:|---|
| Binary `Label` | attack F1: 0.0000 / 0.8529 / 0.9766 | HistGradientBoosting | 0.9952 | attack F1 0.9792 | 0.9955 / 0.9635 | 0.9893 / 0.9776 / 0.00059 |

| Task | Validation macro-F1: majority / logistic / HGB | Selected model | Test accuracy | Test macro-F1 | Test weighted-F1 |
|---|---|---|---:|---:|---:|
| Multiclass `Attack` | 0.0625 / 0.5680 / 0.6907 | HistGradientBoosting | 0.5737 | 0.6915 | 0.7180 |

The binary test confusion matrix contains 18,739 true benign, 11 false positives, 93 missed attacks, and 2,457 detected attacks (sample counts). Source-weighted FPR is 0.00059. This illustrates the trade-off: fewer false alarms come at the cost of some missed attacks; the operating threshold has not been tuned for a deployment setting.

The multiclass report is weak for several rare categories. SQL Injection has only 65 test examples (out of 432 total); weighted precision/recall/F1 are 0.00155 / 0.5846 / 0.00309. Brute Force -Web: 0.00358 / 0.3271 / 0.00709; Brute Force -XSS: 0.00195 / 0.3525 / 0.00387; Infilteration: 0.0150 / 0.8928 / 0.0296. High recalls paired with very low precision show substantial false alarms into these categories. Treat their estimates as uncertain. Full per-class results and confusion matrices are in `reports/experiments/multiclass_metrics.json`.

The validation-only binary duration ablation selected the same HistGradientBoosting model and produced the same attack F1 (0.976589) with `FLOW_DURATION_MILLISECONDS` excluded. This indicates no score change at that operating point on this sample; it does not prove the duration field is free of dataset artifacts. The ablation did not evaluate the test partition. Full details are in `reports/experiments/binary_without_duration_metrics.json`.

The earlier seed-42 run included source/destination ports; it is retained in Git history only. Current model artifacts exclude those ports to follow the reference paper. The two rate fields are also excluded because their magnitudes exceed float32 range and their exact formulas remain unverified; a log transform has not yet been tested.

## Pipeline decisions

1. Binary flow classification using `Label` (0 benign, 1 attack).
2. Separate multiclass flow classification using `Attack`, retaining all 15 categories and reporting support for rare categories.

The feature selector excludes both target columns from predictors in either experiment. The model omits the all-zero `FTP_COMMAND_RET_CODE`, source/destination ports to follow the reference paper, and the two extreme-magnitude directional bytes-per-second fields, which are beyond float32 range. Those rate fields remain in the raw data; a log transform is an open follow-up. Preprocessing is fit on training data only, and test data is reserved for final evaluation.

## Limitations

This dataset audit does not establish generalization to other networks, time periods, or collection environments. The local Parquet counts and detailed attack labels differ from the UQ/paper reference distribution. The smallest category has 432 observations, limiting the certainty of rare-class metrics. Exact-duplicate and capture-level leakage checks remain incomplete. Results must be described as sample-based dataset evaluation rather than a claim of reliable real-world detection.
