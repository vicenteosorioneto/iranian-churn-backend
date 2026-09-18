"""Train, evaluate, and persist two classifiers on one leakage-safe split."""

from typing import Any

import joblib

from app.config import DATASET_PATH, METRICS_PATH, MODEL_PATHS, MODELS_DIR
from app.ml.data import prepare_modeling_dataset
from app.ml.evaluate import calculate_classification_metrics, save_metrics
from app.ml.modeling import ModelingSplit, create_grouped_split
from app.ml.preprocessing import (
    build_logistic_regression_pipeline,
    build_random_forest_pipeline,
)

def build_model_pipelines() -> dict[str, Any]:
    return {
        "logistic_regression": build_logistic_regression_pipeline(),
        "random_forest": build_random_forest_pipeline(),
    }


def fit_and_evaluate_models(
    pipelines: dict[str, Any], split: ModelingSplit
) -> dict[str, dict[str, Any]]:
    """Fit every model with the exact same prepared train/test split."""
    metrics: dict[str, dict[str, Any]] = {}

    for name, pipeline in pipelines.items():
        pipeline.fit(split.x_train, split.y_train)
        predictions = pipeline.predict(split.x_test)
        probabilities = pipeline.predict_proba(split.x_test)[:, 1]
        metrics[name] = calculate_classification_metrics(
            split.y_test, predictions, probabilities
        )

    return metrics


def _class_distribution(values: Any) -> str:
    percentages = values.value_counts(normalize=True).sort_index().mul(100)
    return ", ".join(
        f"Churn={int(label)}: {percentage:.2f}%"
        for label, percentage in percentages.items()
    )


def train_models() -> tuple[dict[str, dict[str, Any]], ModelingSplit]:
    dataframe = prepare_modeling_dataset(DATASET_PATH)
    split = create_grouped_split(dataframe)
    pipelines = build_model_pipelines()
    metrics = fit_and_evaluate_models(pipelines, split)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    for name, pipeline in pipelines.items():
        joblib.dump(pipeline, MODEL_PATHS[name])

    save_metrics(metrics, METRICS_PATH)
    return metrics, split


def main() -> None:
    metrics, split = train_models()
    print(f"Training complete. Metrics saved to: {METRICS_PATH}")
    print(f"Split: train={len(split.x_train)}, test={len(split.x_test)}")
    print(f"Train distribution: {_class_distribution(split.y_train)}")
    print(f"Test distribution: {_class_distribution(split.y_test)}")
    print(f"Shared feature groups: {len(split.shared_groups)}")
    for model_name, model_metrics in metrics.items():
        print(
            f"{model_name}: F1={model_metrics['f1']:.4f}, "
            f"ROC-AUC={model_metrics['roc_auc']:.4f}, "
            f"Average Precision={model_metrics['average_precision']:.4f}, "
            f"Confusion Matrix={model_metrics['confusion_matrix']}"
        )


if __name__ == "__main__":
    main()
