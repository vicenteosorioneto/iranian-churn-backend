from pathlib import Path

import pandas as pd
import pytest

from app.ml.data import read_dataset, validate_dataset_columns
from app.ml.features import REQUIRED_COLUMNS


def test_validate_dataset_columns_accepts_complete_schema() -> None:
    dataframe = pd.DataFrame(columns=REQUIRED_COLUMNS)

    validate_dataset_columns(dataframe)


def test_validate_dataset_columns_reports_missing_columns() -> None:
    dataframe = pd.DataFrame(columns=REQUIRED_COLUMNS[:-2])

    with pytest.raises(ValueError, match="Age Group, Churn"):
        validate_dataset_columns(dataframe)


def test_read_dataset_normalizes_whitespace_in_headers(tmp_path: Path) -> None:
    csv_path = tmp_path / "headers.csv"
    csv_path.write_text(
        "Call  Failure,Subscription   Length, Charge  Amount \n1,2,3\n",
        encoding="utf-8",
    )

    dataframe = read_dataset(csv_path)

    assert dataframe.columns.tolist() == [
        "Call Failure",
        "Subscription Length",
        "Charge Amount",
    ]
