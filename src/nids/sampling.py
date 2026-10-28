"""Bounded stratified sampling from a Parquet flow table."""

from collections.abc import Mapping, Sequence

import numpy as np
import pyarrow.parquet as pq


class StratifiedReservoir:
    """Reservoir sample rows independently for each string/integer class."""

    def __init__(
        self,
        caps: Mapping[object, int],
        *,
        seed: int,
        default_cap: int | None = None,
    ) -> None:
        if any(cap < 1 for cap in caps.values()) or (default_cap is not None and default_cap < 1):
            raise ValueError("Every class cap must be positive")
        self.caps = dict(caps)
        self.default_cap = default_cap
        self.seen = {label: 0 for label in caps}
        self.samples: dict[object, np.ndarray] = {}
        self.rng = np.random.default_rng(seed)

    def update(self, features: np.ndarray, labels: Sequence[object]) -> None:
        labels = np.asarray(labels)
        if self.default_cap is not None:
            for label in np.unique(labels).tolist():
                self.caps.setdefault(label, self.default_cap)
                self.seen.setdefault(label, 0)
        for label, cap in self.caps.items():
            positions = np.flatnonzero(labels == label)
            if not positions.size:
                continue
            old_seen = self.seen[label]
            self.seen[label] += positions.size
            current = self.samples.get(label)
            if current is None:
                current = np.empty((0, features.shape[1]), dtype=features.dtype)

            fill_count = min(cap - current.shape[0], positions.size)
            if fill_count > 0:
                current = np.concatenate((current, features[positions[:fill_count]]), axis=0)

            remaining_positions = positions[fill_count:]
            if remaining_positions.size:
                # Algorithm R in vectorized form. Resolve duplicate replacement
                # slots by keeping their final candidate, as sequential R does.
                ordinal = old_seen + fill_count + np.arange(remaining_positions.size)
                slots = self.rng.integers(0, ordinal + 1)
                eligible = slots < cap
                slots = slots[eligible]
                candidates = features[remaining_positions[eligible]]
                if slots.size:
                    reverse_unique, reverse_index = np.unique(slots[::-1], return_index=True)
                    last_indices = slots.size - 1 - reverse_index
                    current[reverse_unique] = candidates[last_indices]
            self.samples[label] = current

    def result(self) -> tuple[np.ndarray, np.ndarray, dict[object, int]]:
        missing = [label for label, cap in self.caps.items() if label not in self.samples]
        if missing:
            raise ValueError(f"No rows found for requested classes: {missing}")
        ordered = list(self.caps)
        x = np.concatenate([self.samples[label] for label in ordered], axis=0)
        y = np.concatenate([
            np.full(self.samples[label].shape[0], label) for label in ordered
        ])
        supports = {label: self.seen[label] for label in ordered}
        return x, y, supports


def sample_parquet(
    path: str,
    *,
    feature_columns: Sequence[str],
    target_column: str,
    class_caps: Mapping[object, int],
    seed: int = 42,
    batch_size: int = 100_000,
    default_cap: int | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[object, int]]:
    """Read selected numeric features and return a bounded per-class sample."""
    if target_column not in {"Label", "Attack"}:
        raise ValueError("target_column must be Label or Attack")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    required = [*feature_columns, target_column]
    parquet = pq.ParquetFile(path)
    missing = set(required) - set(parquet.schema_arrow.names)
    if missing:
        raise ValueError(f"Requested columns are missing: {sorted(missing)}")
    reservoir = StratifiedReservoir(class_caps, seed=seed, default_cap=default_cap)
    for batch in parquet.iter_batches(batch_size=batch_size, columns=required):
        columns = [batch.column(name).to_numpy(zero_copy_only=False) for name in feature_columns]
        with np.errstate(over="ignore", invalid="ignore"):
            x = np.column_stack(columns).astype(np.float32, copy=False)
        if not np.isfinite(x).all():
            raise ValueError("Feature values became non-finite in float32; review feature handling")
        y = batch.column(target_column).to_pylist()
        reservoir.update(x, y)
    return reservoir.result()
