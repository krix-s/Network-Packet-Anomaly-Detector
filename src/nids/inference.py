"""Validated inference using saved task-specific artifacts."""

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import joblib
import numpy as np


def load_artifact(path: str | Path) -> dict[str, Any]:
    artifact = joblib.load(path)
    required = {"task", "target", "features", "classes", "model"}
    if not isinstance(artifact, dict) or not required.issubset(artifact):
        raise ValueError("Model artifact is missing required metadata")
    if artifact["task"] not in {"binary", "multiclass"}:
        raise ValueError("Model artifact has an unsupported task")
    if not artifact["features"]:
        raise ValueError("Model artifact has no feature names")
    return artifact


def predict_record(artifact: Mapping[str, Any], record: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(record, Mapping):
        raise TypeError("A flow record must be a mapping of feature names to numeric values")
    features = list(artifact["features"])
    missing = [name for name in features if name not in record]
    if missing:
        raise ValueError(f"Missing required flow features: {missing}")
    values = []
    log1p_features = set(artifact.get("log1p_features", ()))
    for name in features:
        value = record[name]
        if isinstance(value, bool):
            raise ValueError(f"Feature {name!r} must be numeric, not boolean")
        try:
            numeric = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Feature {name!r} must be numeric") from exc
        if not np.isfinite(numeric):
            raise ValueError(f"Feature {name!r} must be finite")
        if name in log1p_features:
            if numeric < 0:
                raise ValueError(f"Feature {name!r} must be nonnegative for log1p")
            numeric = float(np.log1p(numeric))
        if abs(numeric) > np.finfo(np.float32).max:
            raise ValueError(f"Feature {name!r} must be representable as float32 after preprocessing")
        values.append(numeric)

    x = np.asarray([values], dtype=np.float32)
    model = artifact["model"]
    prediction = model.predict(x)[0]
    result: dict[str, Any] = {
        "task": artifact["task"],
        "target": artifact["target"],
        "prediction": prediction.item() if hasattr(prediction, "item") else prediction,
    }
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(x)[0]
        labels = artifact.get("classes", model.classes_)
        result["probabilities"] = {
            str(label): float(probability) for label, probability in zip(labels, probabilities)
        }
        result["confidence"] = float(np.max(probabilities))
    return result
