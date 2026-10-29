"""Compare a small feature-axis 1D CNN with the binary project's baselines."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

SEED = 43
EXCLUDED = {"FTP_COMMAND_RET_CODE", "L4_SRC_PORT", "L4_DST_PORT", "SRC_TO_DST_SECOND_BYTES", "DST_TO_SRC_SECOND_BYTES"}


class FlowFeatureCNN(nn.Module):
    """1D CNN over a fixed ordered flow-feature vector, not over time."""
    def __init__(self, feature_count: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Conv1d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(32, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def sample_binary(path: str | Path, seed: int = SEED, batch_size: int = 100_000):
    pf = pq.ParquetFile(path)
    features = [name for name in pf.schema_arrow.names if name not in ({"Label", "Attack"} | EXCLUDED)]
    caps = {0: 125_000, 1: 17_000}
    seen = dict.fromkeys(caps, 0)
    samples: dict[int, np.ndarray] = {}
    rng = np.random.default_rng(seed)
    for batch in pf.iter_batches(batch_size=batch_size, columns=features + ["Label"]):
        labels = np.asarray(batch.column("Label").to_pylist())
        values = np.column_stack([
            batch.column(name).to_numpy(zero_copy_only=False) for name in features
        ]).astype(np.float32, copy=False)
        if not np.isfinite(values).all():
            raise ValueError("Non-finite value encountered in selected feature matrix")
        for label, cap in caps.items():
            positions = np.flatnonzero(labels == label)
            if not positions.size:
                continue
            old_seen = seen[label]
            seen[label] += len(positions)
            current = samples.get(label, np.empty((0, len(features)), dtype=np.float32))
            fill = min(cap - len(current), len(positions))
            if fill:
                current = np.concatenate((current, values[positions[:fill]]), axis=0)
            remaining = positions[fill:]
            if remaining.size:
                ordinals = old_seen + fill + np.arange(remaining.size)
                slots = rng.integers(0, ordinals + 1)
                eligible = slots < cap
                slots = slots[eligible]
                candidates = values[remaining[eligible]]
                if slots.size:
                    _, reverse_index = np.unique(slots[::-1], return_index=True)
                    last_indices = slots.size - 1 - reverse_index
                    current[slots[last_indices]] = candidates[last_indices]
            samples[label] = current
    X = np.concatenate([samples[label] for label in caps])
    y = np.concatenate([np.full(len(samples[label]), label, dtype=np.int8) for label in caps])
    return features, X, y, seen


def run_comparison(data_path: str | Path, *, epochs: int = 12, seed: int = SEED) -> dict:
    torch.manual_seed(seed)
    torch.set_num_threads(min(4, torch.get_num_threads()))
    features, X, y, population_counts = sample_binary(data_path, seed=seed)
    sample_counts = Counter(y.tolist())
    population_weights = np.asarray([
        population_counts[int(label)] / sample_counts[int(label)] for label in y
    ], dtype=np.float64)
    indices = np.arange(len(y))
    train_validation, _test = train_test_split(indices, test_size=0.15, random_state=seed, stratify=y)
    train, validation = train_test_split(
        train_validation, test_size=0.1764705882, random_state=seed, stratify=y[train_validation]
    )
    X_train, X_validation = X[train], X[validation]
    y_train, y_validation = y[train], y[validation]
    val_weights = population_weights[validation]
    fit_weights = compute_sample_weight("balanced", y_train)

    candidates = {
        "majority_baseline": DummyClassifier(strategy="most_frequent"),
        "logistic_regression": make_pipeline(
            StandardScaler(), LogisticRegression(max_iter=1000, tol=1e-3, random_state=seed)
        ),
        "histogram_gradient_boosting": HistGradientBoostingClassifier(
            max_iter=100, max_leaf_nodes=31, learning_rate=0.1,
            l2_regularization=1.0, random_state=seed,
        ),
    }
    scores = {}
    for name, model in candidates.items():
        if name == "logistic_regression":
            model.fit(X_train, y_train, logisticregression__sample_weight=fit_weights)
        elif name == "histogram_gradient_boosting":
            model.fit(X_train, y_train, sample_weight=fit_weights)
        else:
            model.fit(X_train, y_train)
        pred = model.predict(X_validation)
        scores[name] = float(f1_score(
            y_validation, pred, pos_label=1, sample_weight=val_weights, zero_division=0
        ))

    scaler = StandardScaler().fit(X_train)
    X_train_scaled = scaler.transform(X_train).astype(np.float32)
    X_validation_scaled = scaler.transform(X_validation).astype(np.float32)
    labels = np.bincount(y_train.astype(int), minlength=2)
    class_weights = torch.tensor(len(y_train) / (2 * labels), dtype=torch.float32)
    model = FlowFeatureCNN(X.shape[1])
    loss_fn = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    dataset = TensorDataset(
        torch.from_numpy(X_train_scaled).unsqueeze(1),
        torch.from_numpy(y_train.astype(np.int64)),
    )
    loader = DataLoader(
        dataset, batch_size=1024, shuffle=True,
        generator=torch.Generator().manual_seed(seed),
    )
    best_score, best_epoch, best_weights = -1.0, 0, None
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        losses = []
        for xb, yb in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(xb), yb)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
        model.eval()
        with torch.no_grad():
            logits = model(torch.from_numpy(X_validation_scaled).unsqueeze(1))
            prediction = logits.argmax(1).numpy()
        score = float(f1_score(
            y_validation, prediction, pos_label=1, sample_weight=val_weights, zero_division=0
        ))
        history.append({"epoch": epoch, "training_loss": float(np.mean(losses)), "validation_attack_f1": score})
        if score > best_score:
            best_score, best_epoch = score, epoch
            best_weights = {k: v.detach().clone() for k, v in model.state_dict().items()}
    scores["conv1d_flow_features"] = best_score
    report = {
        "seed": seed,
        "task": "binary attack detection",
        "sample_rows": len(y),
        "sample_class_counts": {str(k): int(v) for k, v in sample_counts.items()},
        "source_class_counts": {str(k): int(v) for k, v in population_counts.items()},
        "split_rows": {"train": len(train), "validation": len(validation), "test_reserved": len(y) - len(train) - len(validation)},
        "validation_attack_f1": scores,
        "cnn_best_epoch": best_epoch,
        "cnn_history": history,
        "cnn_architecture": "Conv1d(1,16,k=3) -> Conv1d(16,32,k=3) -> global average pool -> 2 logits",
        "cnn_input_note": "The CNN applies filters to the fixed feature-column order, not to time. The columns are heterogeneous flow statistics.",
        "evaluation_note": "Validation-only comparison; the test partition was reserved and not scored for this CNN extension.",
    }
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/raw/NF-CSE-CIC-IDS2018-V2.parquet"))
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--output", type=Path, default=Path("live_traffic_demo/flow_cnn_validation.json"))
    args = parser.parse_args()
    report = run_comparison(args.data, epochs=args.epochs)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
