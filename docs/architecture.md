# Architecture

```text
Local Parquet flow table
        │
        ├── audit_data.py ──> schema, quality, class-frequency report
        │
        └── train.py
             ├── bounded per-class reservoir sample (Arrow batches)
             ├── target-specific feature selector (always removes Label and Attack)
             ├── stratified train / validation / test split (seed 42)
             ├── Logistic Regression and HistGradientBoosting comparison
             ├── validation-only selection; one final held-out test evaluation
             ├── models/{binary,multiclass}_model.joblib (local, ignored)
             └── reports/experiments/{task}_metrics.json (tracked results)

Saved artifact + one feature record
        └── predict.py ──> validated task-specific prediction and probabilities
```

The streaming sampler caps each class independently and tracks the source population frequencies. The capped sample is split by target stratification. Model fit and validation use the sample; final metrics are reweighted by source-class counts to approximate source prevalence. This does not replace evaluation on a persisted full-data holdout, and it does not test generalization across capture sessions or time. SQL Injection has only 432 source examples, so its held-out multiclass estimate is especially uncertain.

The inference artifact stores the estimator, task/target names, expected feature order, and classes together. Prediction validates all required values and supplies them in the saved order. No live packet capture is implemented.
