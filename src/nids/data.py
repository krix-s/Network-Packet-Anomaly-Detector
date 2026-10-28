"""Memory-conscious access to the flow dataset."""

from collections.abc import Iterator, Sequence
from pathlib import Path

import pyarrow.parquet as pq

from .config import DEFAULT_DATA_PATH, TARGET_COLUMNS


def parquet_metadata(path: str | Path = DEFAULT_DATA_PATH) -> dict[str, object]:
    """Return file-level facts without loading dataset rows into memory."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Dataset not found: {path}")
    parquet = pq.ParquetFile(path)
    return {
        "path": path,
        "rows": parquet.metadata.num_rows,
        "row_groups": parquet.num_row_groups,
        "columns": parquet.schema_arrow.names,
        "size_bytes": path.stat().st_size,
    }


def iter_parquet_batches(
    path: str | Path = DEFAULT_DATA_PATH,
    *,
    columns: Sequence[str] | None = None,
    batch_size: int = 100_000,
) -> Iterator[object]:
    """Yield Arrow record batches, avoiding a whole-file DataFrame allocation."""
    if batch_size <= 0:
        raise ValueError("batch_size must be a positive integer")
    metadata = parquet_metadata(path)
    available = set(metadata["columns"])
    required = set(TARGET_COLUMNS)
    if not required.issubset(available):
        raise ValueError(f"Dataset is missing target columns: {sorted(required - available)}")
    if columns is not None:
        missing = set(columns) - available
        if missing:
            raise ValueError(f"Requested columns are missing: {sorted(missing)}")
    parquet = pq.ParquetFile(path)
    yield from parquet.iter_batches(batch_size=batch_size, columns=columns)
