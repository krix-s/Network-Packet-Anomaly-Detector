#!/usr/bin/env python3
"""Reproduce the bounded-memory schema, quality, and target audit."""

from collections import Counter

import numpy as np

from nids.config import DEFAULT_DATA_PATH
from nids.data import iter_parquet_batches, parquet_metadata


def main() -> None:
    meta = parquet_metadata(DEFAULT_DATA_PATH)
    columns = list(meta["columns"])
    features = [name for name in columns if name not in {"Label", "Attack"}]
    nulls: Counter[str] = Counter()
    nonfinite: Counter[str] = Counter()
    attacks: Counter[str] = Counter()
    label_pairs: Counter[tuple[str, int]] = Counter()
    duration_zero_by_label: Counter[int] = Counter()
    duration_exact_saturation_by_label: Counter[int] = Counter()
    extreme_rate_by_label: Counter[int] = Counter()
    extreme_rate_rows = 0
    bounds = {name: [float("inf"), float("-inf")] for name in features}
    distinct = {name: set() for name in features}
    rows = 0

    for batch in iter_parquet_batches(columns=columns, batch_size=200_000):
        rows += batch.num_rows
        for name in columns:
            nulls[name] += batch.column(name).null_count
        attack_values = batch.column("Attack").to_pylist()
        label_values = batch.column("Label").to_pylist()
        attacks.update(attack_values)
        label_pairs.update(zip(attack_values, label_values))
        label_array = np.asarray(label_values)
        duration = batch.column("FLOW_DURATION_MILLISECONDS").to_numpy(zero_copy_only=False)
        for label in (0, 1):
            mask = label_array == label
            duration_zero_by_label[label] += int(np.sum(mask & (duration == 0)))
            duration_exact_saturation_by_label[label] += int(np.sum(mask & (duration == 4_294_967)))
        rate_a = batch.column("SRC_TO_DST_SECOND_BYTES").to_numpy(zero_copy_only=False)
        rate_b = batch.column("DST_TO_SRC_SECOND_BYTES").to_numpy(zero_copy_only=False)
        extreme = (rate_a > 1e9) | (rate_b > 1e9)
        extreme_rate_rows += int(extreme.sum())
        for label in (0, 1):
            extreme_rate_by_label[label] += int(np.sum(extreme & (label_array == label)))
        for name in features:
            values = batch.column(name).to_numpy(zero_copy_only=False)
            if values.dtype.kind not in "iuf":
                continue
            finite = np.isfinite(values)
            nonfinite[name] += int((~finite).sum())
            values = values[finite]
            if not values.size:
                continue
            bounds[name][0] = min(bounds[name][0], float(values.min()))
            bounds[name][1] = max(bounds[name][1], float(values.max()))
            # Exact constant detection needs no more than three distinct values.
            if len(distinct[name]) <= 1:
                distinct[name].update(np.unique(values).tolist())
                if len(distinct[name]) > 2:
                    distinct[name] = {None, False, True}

    print(f"Rows: {rows:,}; columns: {len(columns)}; bytes: {meta['size_bytes']:,}")
    print("Attack counts:")
    for label, count in attacks.most_common():
        print(f"  {label}: {count:,}")
    print("Label/Attack pairs:", dict(label_pairs))
    print("Null counts:", {k: v for k, v in nulls.items() if v})
    print("Non-finite numeric counts:", {k: v for k, v in nonfinite.items() if v})
    print("Constant numeric features:", [k for k, v in distinct.items() if len(v) == 1])
    for name in ("SRC_TO_DST_SECOND_BYTES", "DST_TO_SRC_SECOND_BYTES", "FLOW_DURATION_MILLISECONDS"):
        print(f"{name} range: {bounds[name][0]} .. {bounds[name][1]}")
    print("Duration zero counts by Label:", dict(duration_zero_by_label))
    print("Duration 4,294,967 counts by Label:", dict(duration_exact_saturation_by_label))
    print(">1e9 in either directional bytes/sec field:", extreme_rate_rows)
    print("Extreme-rate rows by Label:", dict(extreme_rate_by_label))


if __name__ == "__main__":
    main()
