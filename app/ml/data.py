"""Dataset loading and schema validation."""

from pathlib import Path

import pandas as pd
from pandas.errors import EmptyDataError

from app.ml.features import FEATURE_DOMAINS, REQUIRED_COLUMNS

EXPECTED_MODELING_ROWS = 2_850


def validate_dataset_columns(dataframe: pd.DataFrame) -> None:
    """Raise a clear error when any model input or target is absent."""
    missing = [column for column in REQUIRED_COLUMNS if column not in dataframe.columns]
    if missing:
        raise ValueError(
            "Dataset is missing required columns: " + ", ".join(missing)
        )


def validate_dataset_domains(dataframe: pd.DataFrame) -> None:
    """Validate categorical, ordinal, and target values without modifying them."""
    validate_dataset_columns(dataframe)
    violations: list[str] = []

    for column, allowed_values in FEATURE_DOMAINS.items():
        observed_values = set(dataframe[column].dropna().unique().tolist())
        unexpected_values = observed_values.difference(allowed_values)
        if unexpected_values:
            ordered_values = sorted(unexpected_values, key=str)
            violations.append(f"{column}: {ordered_values}")

    if violations:
        raise ValueError(
            "Dataset contains values outside the expected domains: "
            + "; ".join(violations)
        )


def read_dataset(path: Path) -> pd.DataFrame:
    """Read the CSV without dropping extra columns or changing values."""
    if not path.is_file():
        raise FileNotFoundError(
            f"Dataset not found at '{path}'. Place the real CSV at this path."
        )

    if path.stat().st_size == 0:
        raise ValueError(f"Dataset file is empty: '{path}'.")

    try:
        dataframe = pd.read_csv(path)
    except EmptyDataError as exc:
        raise ValueError(f"Dataset file has no readable columns: '{path}'.") from exc

    normalized_columns = (
        dataframe.columns
        .astype(str)
        .str.strip()
        .str.replace(r"\s+", " ", regex=True)
    )
    if normalized_columns.duplicated().any():
        duplicates = normalized_columns[normalized_columns.duplicated()].tolist()
        raise ValueError(f"Dataset has duplicate column names: {duplicates}")
    dataframe.columns = normalized_columns
    return dataframe


def load_dataset(path: Path) -> pd.DataFrame:
    dataframe = read_dataset(path)
    validate_dataset_columns(dataframe)
    return dataframe.loc[:, REQUIRED_COLUMNS].copy()


def prepare_modeling_dataset(path: Path) -> pd.DataFrame:
    """Return the modeling rows without changing or persisting the raw dataset."""
    dataframe = load_dataset(path)
    modeling_dataframe = dataframe.drop_duplicates(
        subset=list(REQUIRED_COLUMNS), keep="first"
    ).reset_index(drop=True)

    if len(modeling_dataframe) != EXPECTED_MODELING_ROWS:
        raise ValueError(
            "Unexpected modeling dataset size after exact deduplication: "
            f"expected {EXPECTED_MODELING_ROWS}, got {len(modeling_dataframe)}."
        )

    return modeling_dataframe
