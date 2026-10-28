#!/usr/bin/env python3
"""Predict one flow record from a JSON object."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nids.inference import load_artifact, predict_record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path, help="Saved binary or multiclass model")
    parser.add_argument("--input", type=Path, help="JSON object file; omit to read one object from stdin")
    args = parser.parse_args()
    raw = args.input.read_text(encoding="utf-8") if args.input else sys.stdin.read()
    payload = json.loads(raw)
    if not isinstance(payload, dict):
        parser.error("Input JSON must be one object mapping feature names to values")
    result = predict_record(load_artifact(args.model), payload)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
