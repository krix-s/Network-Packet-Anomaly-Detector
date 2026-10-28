#!/usr/bin/env python3
"""Train and evaluate the bounded-sample flow classifiers."""

import argparse
import json
from pathlib import Path

from nids.config import DEFAULT_DATA_PATH
from nids.experiment import run_experiment


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=("binary", "multiclass", "all"), default="all")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--output-dir", type=Path, default=Path("models"))
    parser.add_argument("--report-dir", type=Path, default=Path("reports/experiments"))
    parser.add_argument("--multiclass-cap", type=int, default=12_000,
                        help="Maximum rows retained per attack category")
    parser.add_argument("--batch-size", type=int, default=100_000)
    parser.add_argument("--drop-duration", action="store_true",
                        help="Run a duration-feature ablation")
    parser.add_argument("--validation-only", action="store_true",
                        help="Do not inspect the held-out test partition")
    parser.add_argument("--run-suffix", default="",
                        help="Optional suffix for model and metric filenames")
    args = parser.parse_args()

    tasks = ("binary", "multiclass") if args.task == "all" else (args.task,)
    for task in tasks:
        summary = run_experiment(
            task,
            data_path=args.data,
            output_dir=args.output_dir,
            report_dir=args.report_dir,
            multiclass_cap=args.multiclass_cap,
            batch_size=args.batch_size,
            drop_duration=args.drop_duration,
            validation_only=args.validation_only,
            run_suffix=args.run_suffix,
        )
        print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
