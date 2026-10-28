#!/usr/bin/env python3
"""Serve the local, no-extra-dependency browser demo."""

from __future__ import annotations

import argparse
import json
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

ROOT = Path(__file__).resolve().parents[1]
WEB_DIR = ROOT / "web"
sys.path.insert(0, str(ROOT / "src"))


def _load_json(path: Path) -> dict[str, Any] | list[Any]:
    return json.loads(path.read_text(encoding="utf-8"))


class DemoHandler(BaseHTTPRequestHandler):
    server_version = "FlowWatchDemo/1.0"

    def _send(self, status: int, payload: bytes, content_type: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(payload)

    def _json(self, status: int, payload: Any) -> None:
        encoded = json.dumps(payload, allow_nan=False).encode("utf-8")
        self._send(status, encoded, "application/json; charset=utf-8")

    def do_GET(self) -> None:  # noqa: N802 - stdlib handler API
        path = urlparse(self.path).path
        if path in {"/", "/index.html"}:
            self._send(200, (WEB_DIR / "index.html").read_bytes(), "text/html; charset=utf-8")
            return
        if path == "/api/examples":
            examples = _load_json(WEB_DIR / "demo_examples.json")
            self._json(200, [
                {key: item[key] for key in ("id", "title", "description", "known_attack")}
                for item in examples
            ])
            return
        if path == "/api/flow-summary":
            example_id = parse_qs(urlparse(self.path).query).get("example_id", [""])[0]
            examples = _load_json(WEB_DIR / "demo_examples.json")
            example = next((item for item in examples if item["id"] == example_id), None)
            if example is None:
                self._json(404, {"error": "Example not found"})
                return
            record = example["record"]
            self._json(200, {
                "duration_ms": record.get("FLOW_DURATION_MILLISECONDS", 0),
                "packets": record.get("IN_PKTS", 0) + record.get("OUT_PKTS", 0),
                "bytes": record.get("IN_BYTES", 0) + record.get("OUT_BYTES", 0),
            })
            return
        if path == "/api/summary":
            binary = _load_json(ROOT / "reports/experiments/binary_metrics.json")
            multiclass = _load_json(ROOT / "reports/experiments/multiclass_source_prior_metrics.json")
            binary_test = binary["test_metrics"]
            multi_test = multiclass["test_metrics"]
            self._json(200, {
                "binary": {
                    "attack_f1": binary_test["attack_f1_population_weighted"],
                    "attack_precision": binary_test["attack_precision_population_weighted"],
                    "attack_recall": binary_test["attack_recall_population_weighted"],
                    "false_positive_rate": binary_test["false_positive_rate_population_weighted"],
                    "sampled_rows": binary["sampled_rows"],
                    "test_rows": binary["split_rows"]["test"],
                },
                "multiclass": {
                    "macro_f1": multi_test["macro_f1_population_weighted"],
                    "accuracy": multi_test["accuracy_weighted"],
                    "class_count": len(multiclass["test_metrics"]["classification_report_population_weighted"]) - 3,
                    "sql_test_support": multiclass["test_support_sample_counts"].get("SQL Injection", 0),
                    "sql_f1": multi_test["classification_report_population_weighted"]["SQL Injection"]["f1-score"],
                },
                "note": "Sample-based, source-weighted estimates on this dataset; not a live traffic result.",
            })
            return
        self._json(404, {"error": "Not found"})

    def do_POST(self) -> None:  # noqa: N802 - stdlib handler API
        if urlparse(self.path).path != "/api/predict":
            self._json(404, {"error": "Not found"})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length < 1 or length > 1_024:
                raise ValueError("Choose one of the built-in examples and try again.")
            request = json.loads(self.rfile.read(length))
            example_id = request.get("example_id") if isinstance(request, dict) else None
            examples = _load_json(WEB_DIR / "demo_examples.json")
            example = next((item for item in examples if item["id"] == example_id), None)
            if example is None:
                raise ValueError("That example is not available. Choose one of the cards and try again.")

            from nids.inference import predict_record

            binary_artifact = self.server.artifacts["binary"]
            multi_artifact = self.server.artifacts["multiclass"]
            binary_result = predict_record(binary_artifact, example["record"])
            multi_result = predict_record(multi_artifact, example["record"])
            ranked = sorted(
                multi_result["probabilities"].items(), key=lambda item: item[1], reverse=True
            )[:5]
            self._json(200, {
                "binary": binary_result,
                "multiclass": multi_result,
                "top_categories": [{"name": name, "probability": score} for name, score in ranked],
                "reference": {
                    "known_attack": example["known_attack"],
                    "known_binary_label": example["known_binary_label"],
                },
                "example_id": example_id,
            })
        except (ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
            self._json(400, {"error": str(exc)})
        except FileNotFoundError as exc:
            missing = Path(exc.filename or "a required project file")
            self._json(503, {
                "error": f"Could not find {missing.name}. Train the binary and source-prior multiclass models first."
            })
        except Exception as exc:  # return a helpful app error instead of a traceback to the browser
            self.log_error("Prediction failed: %s", exc)
            self._json(500, {"error": "The demo could not complete the analysis. Check the terminal for details."})

    def log_message(self, fmt: str, *args: Any) -> None:
        print(f"[demo] {self.address_string()} - {fmt % args}", file=sys.stderr)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    required = (
        WEB_DIR / "index.html",
        WEB_DIR / "demo_examples.json",
        ROOT / "models/binary_model.joblib",
        ROOT / "models/multiclass_source_prior_model.joblib",
    )
    missing = [path for path in required if not path.exists()]
    if missing:
        parser.error("Missing demo files: " + ", ".join(str(path) for path in missing))
    from nids.inference import load_artifact

    server = ThreadingHTTPServer(("127.0.0.1", args.port), DemoHandler)
    server.artifacts = {
        "binary": load_artifact(ROOT / "models/binary_model.joblib"),
        "multiclass": load_artifact(ROOT / "models/multiclass_source_prior_model.joblib"),
    }
    print(f"FlowWatch demo is ready at http://127.0.0.1:{args.port}")
    print("Press Ctrl+C to stop. This demo does not capture live network traffic.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping FlowWatch demo.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
