"""Leakage-conscious model comparison for sampled flow records."""

from __future__ import annotations

import json
from pathlib import Path
from time import perf_counter
from typing import Any

import joblib
import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight

from .config import (
    BINARY_TARGET,
    DEFAULT_DATA_PATH,
    EXTREME_RATE_FEATURES,
    INITIAL_EXCLUDED_FEATURES,
    MULTICLASS_TARGET,
    PORT_FEATURES,
)
from .data import parquet_metadata
from .features import feature_columns
from .sampling import sample_parquet


SEED = 42


def _splits(y: np.ndarray, seed: int = SEED) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    indices = np.arange(y.size)
    train_val, test = train_test_split(
        indices, test_size=0.15, random_state=seed, stratify=y
    )
    train, validation = train_test_split(
        train_val,
        test_size=0.1764705882,
        random_state=seed,
        stratify=y[train_val],
    )
    return train, validation, test


def _fit_sample_weights(
    strategy: str,
    y: np.ndarray,
    population_weights: np.ndarray,
) -> np.ndarray:
    """Return fit weights, optionally correcting a capped sample's class priors."""
    if strategy == "balanced":
        return compute_sample_weight("balanced", y)
    if strategy == "source-prior":
        weights = np.asarray(population_weights, dtype=np.float64)
        if weights.shape != y.shape or not np.isfinite(weights).all() or np.any(weights <= 0):
            raise ValueError("source-prior weights must be positive, finite, and match labels")
        return weights / weights.mean()
    raise ValueError("training_weighting must be 'balanced' or 'source-prior'")


def _metrics(
    task: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    probabilities: np.ndarray,
    classes: np.ndarray,
    sample_weight: np.ndarray,
) -> dict[str, Any]:
    report = classification_report(
        y_true,
        y_pred,
        labels=classes,
        output_dict=True,
        zero_division=0,
        sample_weight=sample_weight,
    )
    result: dict[str, Any] = {
        "accuracy_weighted": float(accuracy_score(y_true, y_pred, sample_weight=sample_weight)),
        "classification_report_population_weighted": report,
        "confusion_matrix_sample_counts": confusion_matrix(y_true, y_pred, labels=classes).tolist(),
        "confusion_matrix_population_weighted": confusion_matrix(
            y_true, y_pred, labels=classes, sample_weight=sample_weight
        ).tolist(),
    }
    if task == "binary":
        attack = 1
        result.update({
            "attack_precision_population_weighted": float(precision_score(
                y_true, y_pred, pos_label=attack, sample_weight=sample_weight, zero_division=0
            )),
            "attack_recall_population_weighted": float(recall_score(
                y_true, y_pred, pos_label=attack, sample_weight=sample_weight, zero_division=0
            )),
            "attack_f1_population_weighted": float(f1_score(
                y_true, y_pred, pos_label=attack, sample_weight=sample_weight, zero_division=0
            )),
            "roc_auc_population_weighted": float(roc_auc_score(
                y_true, probabilities[:, list(classes).index(attack)], sample_weight=sample_weight
            )),
            "pr_auc_population_weighted": float(average_precision_score(
                y_true, probabilities[:, list(classes).index(attack)], sample_weight=sample_weight
            )),
        })
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1], sample_weight=sample_weight)
        result["false_positive_rate_population_weighted"] = float(cm[0, 1] / max(cm[0].sum(), 1))
    else:
        result["macro_f1_population_weighted"] = float(f1_score(
            y_true, y_pred, labels=classes, average="macro", sample_weight=sample_weight, zero_division=0
        ))
        result["weighted_f1_population_weighted"] = float(f1_score(
            y_true, y_pred, labels=classes, average="weighted", sample_weight=sample_weight, zero_division=0
        ))
    return result


def run_experiment(
    task: str,
    *,
    data_path: Path = DEFAULT_DATA_PATH,
    output_dir: Path = Path("models"),
    multiclass_cap: int = 12_000,
    batch_size: int = 100_000,
    report_dir: Path = Path("reports/experiments"),
    drop_duration: bool = False,
    validation_only: bool = False,
    run_suffix: str = "",
    seed: int = SEED,
    log_rates: bool = False,
    include_ports: bool = False,
    training_weighting: str = "balanced",
) -> dict[str, Any]:
    if task not in {"binary", "multiclass"}:
        raise ValueError("task must be 'binary' or 'multiclass'")
    target = BINARY_TARGET if task == "binary" else MULTICLASS_TARGET
    metadata = parquet_metadata(data_path)
    excluded = tuple(
        name for name in INITIAL_EXCLUDED_FEATURES
        if not (log_rates and name in EXTREME_RATE_FEATURES)
        and not (include_ports and name in PORT_FEATURES)
    ) + (
        ("FLOW_DURATION_MILLISECONDS",) if drop_duration else ()
    )
    predictors = feature_columns(metadata["columns"], target=target, excluded=excluded)
    if task == "binary":
        caps = {0: 125_000, 1: 17_000}  # close to the full-data class prevalence
        default_cap = None
    else:
        caps = {}
        default_cap = multiclass_cap
    sample_started = perf_counter()
    x, y, population_counts = sample_parquet(
        str(data_path),
        feature_columns=predictors,
        target_column=target,
        class_caps=caps,
        default_cap=default_cap,
        seed=seed,
        batch_size=batch_size,
        log1p_columns=EXTREME_RATE_FEATURES if log_rates else (),
    )
    sampling_seconds = perf_counter() - sample_started
    classes, sample_counts = np.unique(y, return_counts=True)
    counts_by_class = dict(zip(classes.tolist(), sample_counts.tolist()))
    weights = np.asarray([population_counts[value] / counts_by_class[value] for value in y])
    train, validation, test = _splits(y, seed=seed)

    baseline = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            max_iter=1000,
            tol=1e-3,
            random_state=seed,
        ),
    )
    stronger = HistGradientBoostingClassifier(
        max_iter=100,
        max_leaf_nodes=31,
        learning_rate=0.1,
        l2_regularization=1.0,
        random_state=seed,
    )
    candidates = {
        "majority_class": DummyClassifier(strategy="most_frequent"),
        "logistic_regression": baseline,
        "hist_gradient_boosting": stronger,
    }
    validation_scores = {}
    validation_metrics = {}
    fit_seconds = {}
    for name, estimator in candidates.items():
        train_weights = _fit_sample_weights(
            training_weighting, y[train], weights[train]
        )
        started = perf_counter()
        if name == "logistic_regression":
            estimator.fit(x[train], y[train], logisticregression__sample_weight=train_weights)
        elif name == "hist_gradient_boosting":
            estimator.fit(x[train], y[train], sample_weight=train_weights)
        else:
            estimator.fit(x[train], y[train])
        fit_seconds[name] = perf_counter() - started
        pred = estimator.predict(x[validation])
        if task == "binary":
            score = f1_score(
                y[validation], pred, pos_label=1,
                sample_weight=weights[validation], zero_division=0,
            )
        else:
            score = f1_score(
                y[validation], pred, labels=classes, average="macro",
                sample_weight=weights[validation], zero_division=0,
            )
        validation_scores[name] = float(score)
        validation_metrics[name] = _metrics(
            task,
            y[validation],
            pred,
            estimator.predict_proba(x[validation]),
            classes,
            weights[validation],
        )

    selected_name = max(validation_scores, key=validation_scores.get)
    selected = candidates[selected_name]
    fit_indices = np.concatenate((train, validation))
    fit_weights = _fit_sample_weights(
        training_weighting, y[fit_indices], weights[fit_indices]
    )
    refit_started = perf_counter()
    if selected_name == "logistic_regression":
        selected.fit(x[fit_indices], y[fit_indices], logisticregression__sample_weight=fit_weights)
    elif selected_name == "hist_gradient_boosting":
        selected.fit(x[fit_indices], y[fit_indices], sample_weight=fit_weights)
    else:
        selected.fit(x[fit_indices], y[fit_indices])
    refit_seconds = perf_counter() - refit_started

    test_metrics = None
    if not validation_only:
        test_metrics = _metrics(
            task,
            y[test],
            selected.predict(x[test]),
            selected.predict_proba(x[test]),
            classes,
            weights[test],
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"_{run_suffix}" if run_suffix else ""
    artifact_path = output_dir / f"{task}{suffix}_model.joblib"
    joblib.dump({
        "task": task,
        "target": target,
        "features": predictors,
        "log1p_features": list(EXTREME_RATE_FEATURES if log_rates else ()),
        "classes": classes.tolist(),
        "model_name": selected_name,
        "model": selected,
        "seed": seed,
    }, artifact_path)
    summary = {
        "task": task,
        "target": target,
        "seed": seed,
        "sampled_rows": int(len(y)),
        "sample_matrix_bytes": int(x.nbytes),
        "source_rows_scanned": int(metadata["rows"]),
        "sampling_seconds": sampling_seconds,
        "population_class_counts": {str(k): int(v) for k, v in population_counts.items()},
        "sample_class_counts": {str(k): int(v) for k, v in counts_by_class.items()},
        "split_rows": {"train": int(len(train)), "validation": int(len(validation)), "test": int(len(test))},
        "features": predictors,
        "excluded_features": list(excluded),
        "include_ports": include_ports,
        "training_weighting": training_weighting,
        "log1p_features": list(EXTREME_RATE_FEATURES if log_rates else ()),
        "validation_only": validation_only,
        "selection_metric": "attack_f1_population_weighted" if task == "binary" else "macro_f1_population_weighted",
        "model_configurations": {
            "majority_class": {"strategy": "most_frequent"},
            "logistic_regression": {"max_iter": 1000, "tol": 1e-3, "scaler": "StandardScaler", "training_weights": training_weighting},
            "hist_gradient_boosting": {"max_iter": 100, "max_leaf_nodes": 31, "learning_rate": 0.1, "l2_regularization": 1.0, "training_weights": training_weighting},
        },
        "validation_scores": validation_scores,
        "validation_metrics": validation_metrics,
        "fit_seconds": fit_seconds,
        "refit_seconds": refit_seconds,
        "selected_model": selected_name,
        "test_metrics": test_metrics,
        "model_path": str(artifact_path),
    }
    report_dir.mkdir(parents=True, exist_ok=True)
    if test_metrics is not None:
        summary["test_support_sample_counts"] = {
            str(label): int(np.sum(y[test] == label)) for label in classes
        }
        summary["insufficient_test_support"] = {
            str(label): int(np.sum(y[test] == label))
            for label in classes if np.sum(y[test] == label) < 100
        }
    metrics_path = report_dir / f"{task}{suffix}_metrics.json"
    metrics_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
