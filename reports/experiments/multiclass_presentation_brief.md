# Multiclass Flow Classification — Follow-up Experiment

## Task and sample

Predict the `Attack` category from the local 43-column flow dataset (41 numeric features plus `Label` and `Attack`). The model classifies flow records; it does not inspect raw packets.

- Seed 43; 135,126 sampled flows across all 15 categories.
- At most 12,000 rows per category; categories with fewer source rows were retained in full.
- Stratified split: 94,588 train, 20,269 validation, and 20,269 test rows.
- A per-row source-population weight corrects the equal-per-class cap for evaluation. The follow-up also uses this correction during fitting (`source-prior` weighting).
- Validation selects among majority-class, logistic regression, and histogram gradient boosting. The held-out test metrics below are source-weighted estimates.

## Why change the training weights?

The original balanced model over-predicted several rare classes. For example, on its held-out test sample, `Infilteration` recall was 0.8928 but precision was only 0.0150. The source-prior experiment multiplies each sampled row by its source class count divided by its sample count, then normalizes weights to mean 1. This trains against the local dataset's strongly skewed category frequencies rather than treating each capped class as equally common.

## Validation comparison

| Training weights | Majority macro-F1 | Logistic regression macro-F1 | HGB macro-F1 |
|---|---:|---:|---:|
| Balanced | 0.0625 | 0.5680 | 0.6907 |
| Source-prior | 0.0625 | 0.6118 | **0.7586** |

Both runs use the same seed, sample, split, features, and source-weighted validation macro-F1. The source-prior HGB was selected. These validation-only reports are `multiclass_balanced_validation_metrics.json` and `multiclass_source_prior_metrics.json`.

## Held-out test comparison

| Training weights | Accuracy | Macro-F1 | Weighted-F1 |
|---|---:|---:|---:|
| Balanced | 0.5737 | 0.6915 | 0.7180 |
| Source-prior | **0.9942** | **0.7887** | **0.9932** |

The source-prior result greatly reduces false alarms from dominant classes, especially benign and major DoS/DDoS categories. It is not a uniform improvement for rare categories:

| Category | Balanced P / R / F1 | Source-prior P / R / F1 | Source-prior test support |
|---|---:|---:|---:|
| `Brute Force -Web` | 0.0036 / 0.3271 / 0.0071 | 0.9250 / 0.1153 / 0.2050 | 321 |
| `Brute Force -XSS` | 0.0019 / 0.3525 / 0.0039 | 1.0000 / 0.1223 / 0.2179 | 139 |
| `Infilteration` | 0.0150 / 0.8928 / 0.0296 | 0.6881 / 0.3211 / 0.4379 | 1,800 sampled |
| `SQL Injection` | 0.0015 / 0.5846 / 0.0031 | 0 / 0 / 0 | 65 |

The prior model's poor precision for Web, XSS, and Infilteration is reduced substantially, but recall falls. SQL Injection is not detected by the source-prior model. Its 65 test examples are too few for a stable performance claim. The balanced model's high rare-class recall came with many false positives. Which trade-off is acceptable depends on the intended use and the cost of missed attacks versus false alarms.

## Interpretation and caveats

- `source-prior` uses class counts from the complete local file to correct its deliberately capped sample. This encodes the local dataset prevalence and may not match a future deployment network or time period.
- The test set was evaluated for the earlier balanced baseline and now for the validation-selected source-prior model. The new model choice was made using validation, not test, but the result is a comparison on the same held-out sample rather than a first-touch test.
- Test sample support for `SQL Injection` is 65, `Brute Force -XSS` 139, and `Brute Force -Web` 321. Rare-class estimates have high uncertainty.
- Cross-network and time-based generalization, exact-duplicate leakage, and capture-level leakage remain unresolved.
- The supplied CNN paper uses KDD Cup 99 and NSL-KDD, not this dataset. This is a tabular gradient-boosting experiment, not a reproduction of that CNN.

## Short presentation script

> I compared multiclass models on a seeded sample of the local 43-column dataset. The first balanced model treated capped classes similarly, which helped find rare attacks but produced many false positives. I then used source-prior weights to account for the dataset's original class imbalance. Histogram gradient boosting rose from 0.6907 to 0.7586 validation macro-F1 and scored 0.7887 macro-F1 on the held-out sample. The false-positive trade-off improved sharply for Infilteration and the two brute-force web categories, but their recall dropped, and the model missed all 65 sampled SQL Injection test cases. So the aggregate result is much stronger, while rare-class detection remains unfinished and needs a class-specific strategy.

## Reproduction

```bash
PYTHONPATH=src python scripts/train.py --task multiclass --training-weighting balanced --validation-only --run-suffix balanced_validation --seed 43
PYTHONPATH=src python scripts/train.py --task multiclass --training-weighting source-prior --validation-only --run-suffix source_prior --seed 43
PYTHONPATH=src python scripts/train.py --task multiclass --training-weighting source-prior --run-suffix source_prior --seed 43
```

Final test metrics and the source-prior model are recorded in `multiclass_source_prior_metrics.json` and `models/multiclass_source_prior_model.joblib` (model files are local and ignored by Git).
