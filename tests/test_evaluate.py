import json
from pathlib import Path

from app.ml.evaluate import calculate_classification_metrics, save_metrics


EXPECTED_METRICS = {
    "accuracy",
    "balanced_accuracy",
    "precision",
    "recall",
    "f1",
    "roc_auc",
    "average_precision",
    "confusion_matrix",
}


def test_metrics_json_contains_expected_metrics(tmp_path: Path) -> None:
    model_metrics = calculate_classification_metrics(
        y_true=[0, 0, 1, 1],
        y_pred=[0, 1, 0, 1],
        y_probability=[0.1, 0.7, 0.4, 0.9],
    )
    metrics = {
        "logistic_regression": model_metrics,
        "random_forest": model_metrics,
    }
    path = tmp_path / "metrics.json"

    save_metrics(metrics, path)
    payload = json.loads(path.read_text(encoding="utf-8"))

    assert set(payload) == {"logistic_regression", "random_forest"}
    assert set(payload["logistic_regression"]) == EXPECTED_METRICS
    assert set(payload["random_forest"]) == EXPECTED_METRICS
