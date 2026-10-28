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
PYTHONPATH=src python scripts/train.py --task binary
PYTHONPATH=src python scripts/train.py --task multiclass
# or --task all
```

The trainer scans the Parquet file in Arrow batches and keeps at most 125,000 benign plus 17,000 attack samples for binary classification. Multiclass sampling keeps at most 12,000 rows per category and retains every smaller class in full. Each sample is split into 70% train, 15% validation, and 15% test with seed 42. The validation set selects between majority-class, logistic-regression, and histogram-gradient-boosting models. Class-balanced sample weights are applied only during model fitting; source class counts reweight metrics to reflect the source prevalence. The test partition is evaluated once after selection.

Models are written to ignored local files under `models/`. Metric JSON files are written to `reports/experiments/`. The flow duration ablation can be run with `--task binary --drop-duration --validation-only --run-suffix without_duration`; it reads the source once and intentionally does not inspect test metrics.

## Inference

Provide one JSON object containing every feature named in the saved model artifact. The tool checks required names, numeric types, finiteness, and float32 representability, then returns the task, predicted label, per-class probabilities, and maximum probability:

```bash
PYTHONPATH=src python scripts/predict.py --model models/binary_model.joblib --input flow.json
PYTHONPATH=src python scripts/predict.py --model models/multiclass_model.joblib --input flow.json
```

The saved artifact fixes the feature order and includes the fitted estimator. Binary `Label` and multiclass `Attack` predictions use separate artifacts.

## Initial measured experiment

The first completed experiment used 142,000 binary samples and 135,126 multiclass samples. Models selected by validation were evaluated on stratified held-out subsets, with source-count reweighting. These figures are sample-based estimates and should not be read as full-data training results.

| Task | Selected model | Validation selection score | Test accuracy | Test F1 | Additional test metrics |
|---|---|---:|---:|---:|---|
| Binary `Label` | HistGradientBoosting | attack F1 0.9760 | 0.9954 | attack F1 0.9800 | precision 0.9975; recall 0.9631; ROC-AUC 0.9898; PR-AUC 0.9780; FPR 0.00032 |
| Multiclass `Attack` | HistGradientBoosting | macro-F1 0.7054 | 0.5698 | macro-F1 0.6995 | weighted-F1 0.7146 |

For multiclass, the test sample had only 65 SQL Injection examples; its estimated precision was 0.0167, recall 0.8615, and F1 0.0327. `Infilteration` also had low weighted precision (0.0142) despite 0.8689 recall. The model overpredicts these categories, and their per-class results are uncertain. See the full confusion matrices and per-class metrics in `reports/experiments/` after training.

The simple logistic baseline did not converge within its initial 300-iteration run. The trainer now uses a higher iteration limit and includes a majority-class baseline; those candidates need a fresh run before their results are reported. The very strong binary result also remains subject to the duration-feature shortcut check described in the plan.

## Planned experiments

1. Binary flow classification using `Label` (0 benign, 1 attack).
2. Separate multiclass flow classification using `Attack`, retaining all 15 categories and reporting support for rare categories.

The feature selector excludes both target columns from predictors in either experiment. The initial feature list omits the all-zero `FTP_COMMAND_RET_CODE` and the two extreme-magnitude directional bytes-per-second fields; these rate fields remain in the raw data and should be compared in a documented alternative. Preprocessing must be fit on training data only, and test data must remain untouched until the final evaluation.

## Limitations

This dataset audit does not establish generalization to other networks, time periods, or collection environments. The local Parquet counts and detailed attack labels differ from the UQ/paper reference distribution. The smallest category has 432 observations, limiting the certainty of rare-class metrics. Results must be described as sample-based dataset evaluation rather than a claim of reliable real-world detection.
