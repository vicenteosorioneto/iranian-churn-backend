"""Grouped cross-validation for the two untuned baseline classifiers."""

from collections.abc import Iterator
import json
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from app.config import DATASET_PATH
from app.ml.data import prepare_modeling_dataset
from app.ml.evaluate import calculate_classification_metrics
from app.ml.features import FEATURES, TARGET
from app.ml.modeling import N_SPLITS, RANDOM_STATE, create_feature_group_ids
from app.ml.train import build_model_pipelines

METRIC_NAMES = (
    "accuracy",
    "balanced_accuracy",
    "precision",
    "recall",
    "f1",
    "roc_auc",
    "average_precision",
)


def iter_grouped_folds(
    dataframe: pd.DataFrame,
) -> Iterator[tuple[int, np.ndarray, np.ndarray, int]]:
    """Yield deterministic folds after explicitly checking group isolation."""
    x = dataframe.loc[:, FEATURES]
    y = dataframe.loc[:, TARGET]
    groups = create_feature_group_ids(dataframe)
    splitter = StratifiedGroupKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    for fold, (train_positions, validation_positions) in enumerate(
        splitter.split(x, y, groups), start=1
    ):
        train_groups = set(groups.iloc[train_positions])
        validation_groups = set(groups.iloc[validation_positions])
        shared_group_count = len(train_groups.intersection(validation_groups))
        if shared_group_count:
            raise ValueError(
                f"Fold {fold} shares {shared_group_count} feature groups "
                "between training and validation."
            )
        yield fold, train_positions, validation_positions, shared_group_count


def _summarize_metrics(
    fold_metrics: list[dict[str, float | int]],
) -> dict[str, dict[str, float]]:
    summary: dict[str, dict[str, float]] = {}
    for metric in METRIC_NAMES:
        values = np.asarray([result[metric] for result in fold_metrics], dtype=float)
        summary[metric] = {
            "mean": float(values.mean()),
            "std": float(values.std(ddof=0)),
            "min": float(values.min()),
            "max": float(values.max()),
        }
    return summary


def run_grouped_cross_validation(
    dataframe: pd.DataFrame,
) -> dict[str, Any]:
    """Fit fresh pipelines in every fold and aggregate validation metrics."""
    x = dataframe.loc[:, FEATURES]
    y = dataframe.loc[:, TARGET]
    fold_checks: list[dict[str, int]] = []
    results: dict[str, list[dict[str, float | int]]] = {
        "logistic_regression": [],
        "random_forest": [],
    }

    for fold, train_positions, validation_positions, shared_groups in iter_grouped_folds(
        dataframe
    ):
        x_train = x.iloc[train_positions]
        x_validation = x.iloc[validation_positions]
        y_train = y.iloc[train_positions]
        y_validation = y.iloc[validation_positions]
        fold_checks.append(
            {
                "fold": fold,
                "train_size": len(train_positions),
                "validation_size": len(validation_positions),
                "shared_feature_groups": shared_groups,
            }
        )

        # New estimators and preprocessors are created and fitted in every fold.
        for model_name, pipeline in build_model_pipelines().items():
            pipeline.fit(x_train, y_train)
            predictions = pipeline.predict(x_validation)
            probabilities = pipeline.predict_proba(x_validation)[:, 1]
            metrics = calculate_classification_metrics(
                y_validation, predictions, probabilities
            )
            results[model_name].append(
                {
                    "fold": fold,
                    **{metric: float(metrics[metric]) for metric in METRIC_NAMES},
                }
            )

    return {
        "configuration": {
            "splitter": "StratifiedGroupKFold",
            "n_splits": N_SPLITS,
            "shuffle": True,
            "random_state": RANDOM_STATE,
        },
        "folds": fold_checks,
        "models": {
            model_name: {
                "fold_metrics": fold_metrics,
                "summary": _summarize_metrics(fold_metrics),
            }
            for model_name, fold_metrics in results.items()
        },
    }


def main() -> None:
    dataframe = prepare_modeling_dataset(DATASET_PATH)
    results = run_grouped_cross_validation(dataframe)
    print(json.dumps(results, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
