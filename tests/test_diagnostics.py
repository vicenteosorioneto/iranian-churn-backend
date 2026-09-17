import pandas as pd

from app.ml.diagnostics import analyze_duplicates
from app.ml.features import REQUIRED_COLUMNS


def _row(seed: int, churn: int) -> dict[str, int]:
    row = {column: seed for column in REQUIRED_COLUMNS}
    row["Complains"] = seed % 2
    row["Tariff Plan"] = 1
    row["Status"] = 1
    row["Charge Amount"] = seed
    row["Age Group"] = 1
    row["Churn"] = churn
    return row


def test_analyze_duplicates_distinguishes_exact_and_target_conflicts() -> None:
    first = _row(seed=0, churn=0)
    second = _row(seed=1, churn=0)
    dataframe = pd.DataFrame([first, first, second, {**second, "Churn": 1}])

    analysis = analyze_duplicates(dataframe)

    assert analysis.all_columns.duplicate_rows == 1
    assert analysis.all_columns.duplicate_groups == 1
    assert analysis.all_columns.group_sizes == (2,)
    assert analysis.all_columns.unique_rows == 3
    assert analysis.features_only.duplicate_rows == 2
    assert analysis.features_only.duplicate_groups == 2
    assert analysis.features_only.group_sizes == (2, 2)
    assert analysis.features_only.unique_rows == 2
    assert analysis.conflicting_feature_groups == 1
    assert analysis.conflicting_rows == 2
    assert set(analysis.conflict_examples["Churn"]) == {0, 1}
