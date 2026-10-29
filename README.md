# Network Traffic Anomaly Detector

A flow-based machine-learning project using the local NF-CSE-CIC-IDS2018-V2 dataset. It predicts from connection summaries; it does not read message contents or monitor a live network.

## Try the demonstration

The browser demo contains four recorded examples and shows both the benign-or-attack screening result and the predicted attack category.

**Hosted demo:** [Open FlowWatch](https://krix-s.github.io/Network-Packet-Anomaly-Detector/)

To run the same static demo from a local checkout, use Python 3:

```bash
python3 -m http.server 8000 --directory web
```

Then open <http://127.0.0.1:8000>. If that port is already in use, open the existing demo or change `8000` to `8001` in both places.

GitHub Pages must be set to **GitHub Actions** as its publishing source for the hosted link to work. The workflow publishes only the demo page and its precomputed results.

## Technical walkthroughs

The notebooks are the deeper technical explanation behind the demo:

1. [`01_dataset_eda.ipynb`](notebooks/01_dataset_eda.ipynb) inspects the local Parquet dataset in batches, checks label consistency and data quality, and plots the highly imbalanced attack categories with Seaborn.
2. [`02_model_experiments.ipynb`](notebooks/02_model_experiments.ipynb) explains the sampling and feature decisions, binary and multiclass tasks, model comparison, class weighting, metrics, ablations, and limitations.

The 642 MB Parquet dataset and trained model files are not included. To run the EDA cells, place `NF-CSE-CIC-IDS2018-V2.parquet` in `data/raw/` and install `pyarrow`, `numpy`, `matplotlib`, and `seaborn`. The model-experiments notebook reads published results from the demo bundle and explains them; it does not retrain models.

## What was evaluated

The local dataset contains 17.1 million flow records and 43 columns: 41 numeric features plus binary `Label` and multiclass `Attack` targets. The models were trained and evaluated on bounded, stratified samples, not on all rows.

| Task | Validation selection | Held-out sample result |
|---|---|---|
| Benign vs attack | Compared a majority baseline, logistic regression, and histogram gradient boosting; selected by attack F1 | Attack F1 **0.9792**, precision 0.9955, recall 0.9635 |
| 15-category prediction | Compared balanced and source-prior training weights; selected by macro-F1 | Macro-F1 **0.7887**, accuracy 0.9942 |

The test metrics use source-frequency weights to adjust for capped sampling. The multiclass accuracy is driven by the dataset's highly uneven class frequencies. The model missed all 65 sampled SQL Injection test examples; the demo includes a misclassified example to make this limitation visible.

## Important limits

This is an educational demonstration, not a live intrusion prevention system. The examples and predictions are precomputed. Results do not establish performance on another network, and rare attack categories remain difficult. The public presentation build does not include the raw dataset or trained model files.
