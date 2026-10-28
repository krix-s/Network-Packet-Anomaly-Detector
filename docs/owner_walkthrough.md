# Project owner walkthrough

Use this guide from the project folder to learn the work by running it in small steps. Start with inspection and the demo; retraining is optional because local model files are already present on the development machine.

## 1. Open a terminal in the project

```bash
cd /home/krix/Network-Packet-Anomaly-Detector
.venv/bin/python --version
```

The project uses a local virtual environment. Do not activate it if you use the `.venv/bin/python` form shown below. The dataset is expected at `data/raw/NF-CSE-CIC-IDS2018-V2.parquet` and is intentionally not committed to Git.

## 2. Start with the interactive demo

```bash
.venv/bin/python scripts/demo_server.py
```

Open <http://127.0.0.1:8000>. Try the benign, DDoS, infiltration, and SQL Injection examples. For each one, notice that the page shows two different questions:

1. **Binary screening:** does this flow look benign or like an attack?
2. **Multiclass prediction:** which of the 15 attack labels is the closest match?

The SQL Injection example is intentionally difficult: the recorded label is SQL Injection, while the current category model predicts a different class. This is a useful way to explain that model confidence is not a guarantee. The app uses stored examples and does not capture live network traffic. Stop it with Ctrl+C.

If port 8000 is occupied, the demo may already be open; visit the URL instead of launching another copy. Otherwise use `.venv/bin/python scripts/demo_server.py --port 8001` and open <http://127.0.0.1:8001>.

## 3. Run the checks

```bash
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python scripts/audit_data.py
```

The tests check batch reading, sampling, reproducible splits, model save/load, and input validation. The audit scans the local Parquet in batches and reports its schema, labels, missing values, feature ranges, and constant fields. The full local file has 17,129,715 rows and 43 columns: 41 numeric predictors and two targets.

## 4. Reproduce the binary experiment

```bash
.venv/bin/python scripts/train.py --task binary --seed 43
```

This streams the file, samples up to 125,000 benign and 17,000 attack flows, creates stratified train/validation/test partitions, compares a majority baseline, logistic regression, and histogram gradient boosting, then evaluates the validation-selected model on the held-out sample. It writes:

- `models/binary_model.joblib` — local model used by the demo and prediction CLI.
- `reports/experiments/binary_metrics.json` — model settings, validation scores, test metrics, and confusion matrix.

The report should show HGB validation attack F1 near 0.9766 and source-weighted test attack F1 near 0.9792. Small differences can occur if the environment or library versions change.

## 5. Reproduce the multiclass follow-up

```bash
.venv/bin/python scripts/train.py --task multiclass --training-weighting source-prior --run-suffix source_prior --seed 43
```

This keeps up to 12,000 records per attack category, retains smaller categories in full, and compares the three model baselines. The source-prior weights restore the local dataset's class proportions after sampling caps. The HGB variant scored near 0.7586 validation macro-F1 and 0.7887 test macro-F1 in the recorded run.

Important files:

- `reports/experiments/multiclass_balanced_validation_metrics.json` — validation-only balanced control.
- `reports/experiments/multiclass_source_prior_metrics.json` — source-prior validation and test results.
- `models/multiclass_source_prior_model.joblib` — category model used by the demo.

The source-prior model found fewer false positives in several rare classes but missed all 65 sampled SQL Injection test flows. Read each per-class precision and recall; overall accuracy alone hides this limitation. Training again is useful for learning, but do not tune model choices by repeatedly checking the test score. Use validation for experiments and reserve the test set for final evaluation.

## 6. Try a single prediction yourself

Create a JSON input from the bundled DDoS example, then run both models:

```bash
.venv/bin/python - <<'PY'
import json
from pathlib import Path

examples = json.loads(Path("web/demo_examples.json").read_text())
ddos = next(item for item in examples if item["id"] == "ddos")
Path("/tmp/flowwatch-example.json").write_text(json.dumps(ddos["record"]))
print("Dataset reference label:", ddos["known_attack"])
PY

.venv/bin/python scripts/predict.py --model models/binary_model.joblib --input /tmp/flowwatch-example.json
.venv/bin/python scripts/predict.py --model models/multiclass_source_prior_model.joblib --input /tmp/flowwatch-example.json
```

Each artifact carries its own feature list and order. `predict.py` checks for the complete set of 36 numeric inputs before returning a prediction and class scores.

## 7. Read the code in this order

1. `src/nids/config.py` — targets, paths, and feature exclusions.
2. `src/nids/data.py` and `src/nids/sampling.py` — batch access and bounded-memory sampling.
3. `src/nids/experiment.py` — splits, candidate models, weighting, validation selection, and metrics.
4. `src/nids/inference.py` — input checks and single-record prediction.
5. `scripts/train.py`, `scripts/predict.py`, and `scripts/demo_server.py` — command-line and demo entry points.
6. `web/index.html` and `web/demo_examples.json` — the audience-facing page and its sample flows.

## 8. Explain the results in your own words

- **Precision:** of the flows flagged as attacks, what share really had an attack label?
- **Recall:** of the attack flows in the test sample, what share did the model find?
- **F1:** a combined score that is low when either precision or recall is low.
- **Macro-F1:** calculate F1 per category and average the categories, so rare labels matter too.
- **Source-weighted metrics:** adjust the capped sample back toward the local dataset's original label distribution. They estimate performance for this dataset, not for every network.
- **Test split:** records held out from model fitting and model selection. The saved reports have been evaluated for the recorded comparison; do not choose future settings based on test results.

This is a flow-classification project using the local 43-column dataset. The supplied CNN paper is a reference, not a reproduction: it used KDD Cup 99 and NSL-KDD, while this project uses tabular HGB and logistic-regression models. The project is a prototype, not a live security appliance.

## 9. Git workflow

```bash
git status --short
git log --oneline -5
```

Local model artifacts and the raw Parquet dataset are ignored by Git. `web/demo_examples.json`, code, documentation, and reports are tracked. Preserve your notebook and scratch-file changes when committing other work.
