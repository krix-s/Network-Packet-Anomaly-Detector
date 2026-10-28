# Binary Flow Anomaly Detection — Experiment Brief

## Goal

Predict whether a flow is benign (`Label = 0`) or an attack (`Label = 1`) from the local 43-column flow dataset. This is flow-level classification; the project does not inspect packet payloads or capture live traffic.

## Dataset and evaluation

- Local Parquet: 17,129,715 rows and 43 columns (41 feature columns plus `Label` and `Attack`).
- A bounded, seeded sample of 142,000 rows was used: 125,000 benign and 17,000 attack flows. The sample preserves the source class proportions approximately.
- Stratified split: 99,400 training, 21,300 validation, and 21,300 test records.
- Models were fit with balanced class weights. Reported validation and test metrics use source-frequency weights to account for capped sampling.
- Validation selected the model. The held-out test set was evaluated once for the selected model; ablations below use validation only.

## Baselines and selected model

| Candidate | Validation attack F1 | Purpose |
|---|---:|---|
| Majority class | 0.0000 | Simple reference; predicts benign for every flow |
| Logistic regression | 0.8529 | Linear baseline with standardized features |
| Histogram gradient boosting | **0.9766** | Selected nonlinear tree ensemble |

The final model is histogram gradient boosting. The comparison uses attack-class F1 because missing attacks and raising false alarms matter more than accuracy alone in this imbalanced dataset.

## Held-out test result

| Accuracy | Attack precision | Attack recall | Attack F1 | ROC-AUC | PR-AUC | False-positive rate |
|---:|---:|---:|---:|---:|---:|---:|
| 0.9952 | 0.9955 | 0.9635 | **0.9792** | 0.9893 | 0.9776 | 0.000587 |

At the default decision threshold, the sampled test confusion matrix is:

| Actual / predicted | Benign | Attack |
|---|---:|---:|
| Benign | 18,739 | 11 |
| Attack | 93 | 2,457 |

This is a low-false-alarm result with some missed attacks: about 3.65% of attack test examples were missed, while 11 of 18,750 benign examples were flagged. The weighted false-positive rate is about 0.0587%. The confusion matrix shows test sample counts; the summary metrics are source-frequency weighted estimates.

## Feature experiments

Each ablation was selected and scored on validation only. None of the variants below received a test evaluation, so they should not be presented as test comparisons.

| Validation experiment | Logistic attack F1 | HGB attack F1 | Readout |
|---|---:|---:|---|
| Final feature set (reference) | 0.8529 | **0.9766** | Selected HGB |
| Drop `FLOW_DURATION_MILLISECONDS` | 0.8399 | **0.9766** | Same HGB score at shown precision |
| Include two extreme rate fields after `log1p` | 0.8513 | **0.9766** | Safe numeric representation; no HGB score gain |
| Include source and destination ports | 0.8624 | 0.9764 | Tiny HGB decrease; ports omitted conservatively |

The final model excludes ports to limit endpoint-specific shortcuts, excludes two extreme-magnitude rate fields whose formulas are unverified, and omits the constant `FTP_COMMAND_RET_CODE`. Duration remains included; its ablation had no observed HGB score change on this validation sample, which does not establish robustness to duration-related dataset artifacts.

## Relation to the supplied paper

The supplied CNN paper is a methodological reference, not the source of this dataset or these results. It evaluates KDD Cup 99 and NSL-KDD. The project uses the local 43-column flow dataset and currently compares conventional tabular baselines; it does not claim to reproduce the paper’s CNN or its results. See [`docs/research_basis.md`](../../docs/research_basis.md).

## Limits to state during the presentation

- These are sampled, source-weighted estimates, not results from training on all 17.1 million rows.
- The held-out test partition comes from the same dataset and collection process; it does not establish performance on another network or time period.
- Capture-level and exact-duplicate leakage checks are incomplete.
- The decision threshold has not been tuned to an operational false-alarm or missed-attack cost.
- Validation ablations did not inspect the test partition.

## Short presentation script

> I built a binary flow classifier using the local 43-column network-flow dataset. On a seeded 142,000-row sample, I compared a majority baseline, logistic regression, and histogram gradient boosting, selecting by source-weighted attack F1 on validation. Gradient boosting achieved 0.9766 validation attack F1 and 0.9792 weighted attack F1 on the held-out test sample, with 0.9955 precision and 0.9635 recall. The test sample had 11 false alarms and 93 missed attacks. Feature ablations tested duration, transformed rate fields, and ports on validation only; they did not change the selected model or improve HGB attack F1. Results are promising within this dataset, but cross-network generalization and leakage checks remain open.

## Reproduction

```bash
PYTHONPATH=src python scripts/train.py --task binary --seed 43
PYTHONPATH=src python scripts/train.py --task binary --drop-duration --validation-only --run-suffix without_duration --seed 43
PYTHONPATH=src python scripts/train.py --task binary --log-rates --validation-only --run-suffix rates_log1p --seed 43
PYTHONPATH=src python scripts/train.py --task binary --include-ports --validation-only --run-suffix with_ports --seed 43
```

Detailed machine-readable results are in `binary_metrics.json`, `binary_without_duration_metrics.json`, `binary_rates_log1p_metrics.json`, and `binary_with_ports_metrics.json` in this directory.
