"""Feature selection rules shared by both prediction tasks."""

from collections.abc import Sequence

from .config import INITIAL_EXCLUDED_FEATURES, TARGET_COLUMNS


def feature_columns(
    available_columns: Sequence[str],
    *,
    target: str,
    excluded: Sequence[str] = INITIAL_EXCLUDED_FEATURES,
) -> list[str]:
    """Return predictors while guaranteeing neither target leaks into X."""
    if target not in TARGET_COLUMNS:
        raise ValueError(f"Unsupported target: {target!r}")
    if target not in available_columns:
        raise ValueError(f"Target column is missing: {target}")
    forbidden = set(TARGET_COLUMNS) | set(excluded)
    return [name for name in available_columns if name not in forbidden]
