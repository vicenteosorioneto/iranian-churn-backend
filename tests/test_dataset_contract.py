from pathlib import Path

import pandas as pd
import pytest

from app.config import DATASET_PATH
from app.ml.data import (
    read_dataset,
    validate_dataset_columns,
    validate_dataset_domains,
)
from app.ml.features import FEATURE_DOMAINS, FEATURES, REQUIRED_COLUMNS, TARGET


def _valid_contract_frame() -> pd.DataFrame:
    row = {column: 0 for column in REQUIRED_COLUMNS}
    row.update({"Tariff Plan": 1, "Status": 1, "Age Group": 1})
    return pd.DataFrame([row])


def test_contract_contains_thirteen_features_and_target() -> None:
    dataframe = _valid_contract_frame()

    validate_dataset_columns(dataframe)
    assert len(FEATURES) == 13
    assert TARGET in dataframe.columns


@pytest.mark.parametrize(("column", "invalid_value"), [
    ("Complains", 2),
    ("Tariff Plan", 0),
    ("Status", 0),
    ("Charge Amount", 11),
    ("Age Group", 6),
    ("Churn", 2),
])
def test_contract_rejects_values_outside_domain(
    column: str, invalid_value: int
) -> None:
    dataframe = _valid_contract_frame()
    dataframe.loc[0, column] = invalid_value

    with pytest.raises(ValueError, match=column):
        validate_dataset_domains(dataframe)


def test_declared_domains_match_dataset_specification() -> None:
    assert FEATURE_DOMAINS == {
        "Complains": frozenset({0, 1}),
        "Tariff Plan": frozenset({1, 2}),
        "Status": frozenset({1, 2}),
        "Charge Amount": frozenset(range(11)),
        "Age Group": frozenset(range(1, 6)),
        "Churn": frozenset({0, 1}),
    }


@pytest.mark.skipif(
    not DATASET_PATH.is_file() or DATASET_PATH.stat().st_size == 0,
    reason="The real dataset is not available or is empty.",
)
def test_real_dataset_satisfies_contract() -> None:
    dataframe = read_dataset(Path(DATASET_PATH))

    validate_dataset_columns(dataframe)
    validate_dataset_domains(dataframe)
