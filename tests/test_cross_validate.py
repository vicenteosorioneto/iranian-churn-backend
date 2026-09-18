import pandas as pd

from app.config import DATASET_PATH
from app.ml.cross_validate import METRIC_NAMES, _summarize_metrics, iter_grouped_folds
from app.ml.data import EXPECTED_MODELING_ROWS, prepare_modeling_dataset


def test_every_cross_validation_fold_keeps_feature_groups_isolated() -> None:
    dataframe = prepare_modeling_dataset(DATASET_PATH)
    folds = list(iter_grouped_folds(dataframe))

    assert len(folds) == 5
    for _, train_positions, validation_positions, shared_groups in folds:
        assert shared_groups == 0
        assert len(train_positions) + len(validation_positions) == EXPECTED_MODELING_ROWS


def test_cross_validation_summary_reports_expected_statistics() -> None:
    fold_metrics = [
        {"fold": fold, **{metric: float(fold) for metric in METRIC_NAMES}}
        for fold in range(1, 6)
    ]

    summary = _summarize_metrics(fold_metrics)

    assert set(summary) == set(METRIC_NAMES)
    for metric_summary in summary.values():
        assert metric_summary == {
            "mean": 3.0,
            "std": 2**0.5,
            "min": 1.0,
            "max": 5.0,
        }
