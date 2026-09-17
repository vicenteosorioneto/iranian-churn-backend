"""Read-only data-quality diagnostics."""

from dataclasses import dataclass

import pandas as pd

from app.ml.features import FEATURES, REQUIRED_COLUMNS, TARGET


@dataclass(frozen=True)
class DuplicateScopeSummary:
    """Duplicate counts for one set of comparison columns."""

    duplicate_rows: int
    duplicate_groups: int
    group_sizes: tuple[int, ...]
    unique_rows: int


@dataclass(frozen=True)
class DuplicateAnalysis:
    """Exact, feature-only, and target-conflict duplicate diagnostics."""

    all_columns: DuplicateScopeSummary
    features_only: DuplicateScopeSummary
    conflicting_feature_groups: int
    conflicting_rows: int
    conflict_examples: pd.DataFrame


def _summarize_duplicates(
    dataframe: pd.DataFrame, subset: tuple[str, ...]
) -> DuplicateScopeSummary:
    group_sizes = dataframe.groupby(list(subset), dropna=False, sort=False).size()
    duplicate_group_sizes = group_sizes[group_sizes > 1].sort_values(ascending=False)

    return DuplicateScopeSummary(
        duplicate_rows=int((duplicate_group_sizes - 1).sum()),
        duplicate_groups=int(duplicate_group_sizes.size),
        group_sizes=tuple(int(size) for size in duplicate_group_sizes.tolist()),
        unique_rows=int(dataframe.drop_duplicates(subset=list(subset)).shape[0]),
    )


def analyze_duplicates(
    dataframe: pd.DataFrame, *, example_limit: int = 10
) -> DuplicateAnalysis:
    """Analyze duplicates without removing or mutating any observation."""
    all_columns = _summarize_duplicates(dataframe, REQUIRED_COLUMNS)
    features_only = _summarize_duplicates(dataframe, FEATURES)

    churn_counts = dataframe.groupby(list(FEATURES), dropna=False)[TARGET].nunique()
    conflicting_feature_groups = int((churn_counts > 1).sum())
    conflict_mask = (
        dataframe.groupby(list(FEATURES), dropna=False)[TARGET]
        .transform("nunique")
        .gt(1)
    )
    conflict_rows = dataframe.loc[conflict_mask, REQUIRED_COLUMNS]

    return DuplicateAnalysis(
        all_columns=all_columns,
        features_only=features_only,
        conflicting_feature_groups=conflicting_feature_groups,
        conflicting_rows=int(conflict_mask.sum()),
        conflict_examples=conflict_rows.head(example_limit).copy(),
    )
