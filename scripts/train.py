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
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-rates", action="store_true",
                        help="Include both extreme directional rate fields after log1p")
    parser.add_argument("--include-ports", action="store_true",
                        help="Keep source/destination ports in the predictor set")
    parser.add_argument("--training-weighting", choices=("balanced", "source-prior"),
                        default="balanced",
                        help="Use class-balanced weights or restore source class proportions")
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
            seed=args.seed,
            log_rates=args.log_rates,
            include_ports=args.include_ports,
            training_weighting=args.training_weighting,
        )
        print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
