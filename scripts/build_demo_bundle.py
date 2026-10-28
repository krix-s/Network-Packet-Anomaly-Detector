#!/usr/bin/env python3
"""Build a static, no-model-files-required bundle for the presentation site."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nids.inference import load_artifact, predict_record


def main() -> None:
    examples = json.loads((ROOT / "web/demo_examples.json").read_text(encoding="utf-8"))
    binary_artifact = load_artifact(ROOT / "models/binary_model.joblib")
    multiclass_artifact = load_artifact(ROOT / "models/multiclass_source_prior_model.joblib")
    if binary_artifact["features"] != multiclass_artifact["features"]:
        raise ValueError("Binary and multiclass models must use the same demo feature set")

    binary_report = json.loads(
        (ROOT / "reports/experiments/binary_metrics.json").read_text(encoding="utf-8")
    )
    multiclass_report = json.loads(
        (ROOT / "reports/experiments/multiclass_source_prior_metrics.json").read_text(encoding="utf-8")
    )
    binary_metrics = binary_report["test_metrics"]
    multiclass_metrics = multiclass_report["test_metrics"]

    bundle_examples: list[dict[str, Any]] = []
    for example in examples:
        record = example["record"]
        binary_result = predict_record(binary_artifact, record)
        multiclass_result = predict_record(multiclass_artifact, record)
        ranked = sorted(
            multiclass_result["probabilities"].items(), key=lambda item: item[1], reverse=True
        )[:5]
        bundle_examples.append({
            "id": example["id"],
            "title": example["title"],
            "description": example["description"],
            "known_attack": example["known_attack"],
            "known_binary_label": example["known_binary_label"],
            "flow_summary": {
                "duration_ms": record.get("FLOW_DURATION_MILLISECONDS", 0),
                "packets": record.get("IN_PKTS", 0) + record.get("OUT_PKTS", 0),
                "bytes": record.get("IN_BYTES", 0) + record.get("OUT_BYTES", 0),
            },
            "binary": binary_result,
            "multiclass": multiclass_result,
            "top_categories": [
                {"name": name, "probability": score} for name, score in ranked
            ],
        })

    bundle = {
        "binary": {
            "attack_f1": binary_metrics["attack_f1_population_weighted"],
            "attack_precision": binary_metrics["attack_precision_population_weighted"],
            "attack_recall": binary_metrics["attack_recall_population_weighted"],
            "false_positive_rate": binary_metrics["false_positive_rate_population_weighted"],
            "sampled_rows": binary_report["sampled_rows"],
            "test_rows": binary_report["split_rows"]["test"],
        },
        "multiclass": {
            "macro_f1": multiclass_metrics["macro_f1_population_weighted"],
            "accuracy": multiclass_metrics["accuracy_weighted"],
            "class_count": len(multiclass_metrics["classification_report_population_weighted"]) - 3,
            "sql_test_support": multiclass_report["test_support_sample_counts"].get("SQL Injection", 0),
            "sql_f1": multiclass_metrics["classification_report_population_weighted"]["SQL Injection"]["f1-score"],
        },
        "note": "Precomputed predictions from the local models. The site does not collect or analyze live traffic.",
        "examples": bundle_examples,
    }
    output = ROOT / "web/demo_bundle.json"
    output.write_text(json.dumps(bundle, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"Wrote {output.relative_to(ROOT)} with {len(bundle_examples)} examples.")


if __name__ == "__main__":
    main()
