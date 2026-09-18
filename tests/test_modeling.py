from hashlib import sha256
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from app.config import DATASET_PATH
from app.ml.data import EXPECTED_MODELING_ROWS, prepare_modeling_dataset
from app.ml.features import BINARY_FEATURES, FEATURES, ORDINAL_FEATURES, REQUIRED_COLUMNS
from app.ml.modeling import ModelingSplit, create_grouped_split
from app.ml.preprocessing import (
    build_logistic_regression_pipeline,
    build_random_forest_pipeline,
)
from app.ml.train import fit_and_evaluate_models


def _file_digest(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _synthetic_frame(rows: int = 30) -> pd.DataFrame:
    records: list[dict[str, int | float]] = []
    for index in range(rows):
        row = {feature: float(index + offset) for offset, feature in enumerate(FEATURES)}
        row.update({feature: index % 2 for feature in BINARY_FEATURES})
        row["Tariff Plan"] = 1 + index % 2
        row["Status"] = 1 + index % 2
        row["Charge Amount"] = index % 11
        row["Age Group"] = 1 + index % 5
        row["Churn"] = index % 2
        records.append(row)
    return pd.DataFrame(records, columns=REQUIRED_COLUMNS)


def test_modeling_preparation_preserves_raw_file_and_removes_exact_duplicates() -> None:
    before = _file_digest(DATASET_PATH)

    dataframe = prepare_modeling_dataset(DATASET_PATH)

    assert _file_digest(DATASET_PATH) == before
    assert len(dataframe) == EXPECTED_MODELING_ROWS
    assert not dataframe.duplicated(subset=list(REQUIRED_COLUMNS)).any()


def test_grouped_split_has_no_shared_feature_groups() -> None:
    dataframe = prepare_modeling_dataset(DATASET_PATH)

    split = create_grouped_split(dataframe)

    assert not split.shared_groups
    assert len(split.x_train) + len(split.x_test) == EXPECTED_MODELING_ROWS


class _RecordingPipeline:
    def __init__(self) -> None:
        self.fit_x_index: list[int] = []
        self.fit_y_index: list[int] = []
        self.predict_x_index: list[int] = []

    def fit(self, x: pd.DataFrame, y: pd.Series) -> "_RecordingPipeline":
        self.fit_x_index = x.index.tolist()
        self.fit_y_index = y.index.tolist()
        return self

    def predict(self, x: pd.DataFrame) -> np.ndarray:
        self.predict_x_index = x.index.tolist()
        return np.zeros(len(x), dtype=int)

    def predict_proba(self, x: pd.DataFrame) -> np.ndarray:
        return np.column_stack((np.full(len(x), 0.8), np.full(len(x), 0.2)))


def test_both_models_receive_the_same_split() -> None:
    dataframe = _synthetic_frame()
    split = create_grouped_split(dataframe)
    pipelines = {
        "logistic_regression": _RecordingPipeline(),
        "random_forest": _RecordingPipeline(),
    }

    fit_and_evaluate_models(pipelines, split)

    first, second = pipelines.values()
    assert first.fit_x_index == second.fit_x_index == split.x_train.index.tolist()
    assert first.fit_y_index == second.fit_y_index == split.y_train.index.tolist()
    assert first.predict_x_index == second.predict_x_index == split.x_test.index.tolist()


def test_model_pipelines_fit_and_predict() -> None:
    dataframe = _synthetic_frame()
    split = create_grouped_split(dataframe)

    for pipeline in (
        build_logistic_regression_pipeline(),
        build_random_forest_pipeline(),
    ):
        pipeline.fit(split.x_train, split.y_train)
        predictions = pipeline.predict(split.x_test)
        assert len(predictions) == len(split.y_test)
