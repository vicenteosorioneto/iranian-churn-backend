"""Train, evaluate, and persist the two initial churn classifiers."""

from typing import Any

import joblib
from sklearn.model_selection import train_test_split

from app.config import DATASET_PATH, METRICS_PATH, MODEL_PATHS, MODELS_DIR
from app.ml.data import load_dataset
from app.ml.evaluate import calculate_classification_metrics, save_metrics
from app.ml.features import FEATURES, TARGET
from app.ml.preprocessing import (
    build_logistic_regression_pipeline,
    build_random_forest_pipeline,
)

TEST_SIZE = 0.20
RANDOM_STATE = 42


def train_models() -> dict[str, dict[str, Any]]:
    dataframe = load_dataset(DATASET_PATH)
    x = dataframe.loc[:, FEATURES]
    y = dataframe[TARGET]

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    pipelines = {
        "logistic_regression": build_logistic_regression_pipeline(),
        "random_forest": build_random_forest_pipeline(),
    }
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    metrics: dict[str, dict[str, Any]] = {}

    for name, pipeline in pipelines.items():
        pipeline.fit(x_train, y_train)
        predictions = pipeline.predict(x_test)
        probabilities = pipeline.predict_proba(x_test)[:, 1]
        metrics[name] = calculate_classification_metrics(
            y_test, predictions, probabilities
        )
        joblib.dump(pipeline, MODEL_PATHS[name])

    save_metrics(metrics, METRICS_PATH)
    return metrics


def main() -> None:
    metrics = train_models()
    print(f"Training complete. Metrics saved to: {METRICS_PATH}")
    for model_name, model_metrics in metrics.items():
        print(f"{model_name}: F1={model_metrics['f1']:.4f}")


if __name__ == "__main__":
    main()
