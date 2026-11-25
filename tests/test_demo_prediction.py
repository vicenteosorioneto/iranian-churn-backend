"""Behavioral API tests for adjustable churn thresholds and real example loading."""
from unittest.mock import patch

import pandas as pd
from fastapi.testclient import TestClient

from app.api.routes import analytics
from app.main import app
from app.ml.features import FEATURES, TARGET

client = TestClient(app)


def _payload(threshold: float) -> dict[str, float | str]:
    row = {name: 0 for name in FEATURES}
    row.update({"Tariff Plan": 1, "Status": 1, "Age Group": 1})
    return {**row, "model_name": "random_forest", "threshold": threshold}


def test_threshold_adjusts_classification_without_changing_probability() -> None:
    with patch("app.api.routes.prediction.predict_churn", return_value=(0, 0.42)):
        low = client.post("/api/predict", json=_payload(0.3))
        high = client.post("/api/predict", json=_payload(0.8))
    assert low.status_code == high.status_code == 200
    assert low.json()["prediction"] == 1
    assert high.json()["prediction"] == 0
    assert low.json()["churn_probability"] == high.json()["churn_probability"] == 0.42
    assert low.json()["decision_threshold"] == 0.3
    assert high.json()["decision_threshold"] == 0.8


def test_threshold_must_be_between_zero_and_one() -> None:
    assert client.post("/api/predict", json=_payload(-0.1)).status_code == 422
    assert client.post("/api/predict", json=_payload(1.1)).status_code == 422


def test_examples_are_real_rows_without_target_in_features(monkeypatch) -> None:
    records = []
    for label in (0, 1):
        row = {name: 3.0 + label for name in FEATURES}
        row.update({"Tariff Plan": 1, "Status": 1, "Age Group": 1, TARGET: label})
        records.append(row)
    dataframe = pd.DataFrame(records)
    monkeypatch.setattr(analytics, "prepare_modeling_dataset", lambda path: dataframe)
    response = client.get("/api/analytics/examples")
    assert response.status_code == 200
    data = response.json()
    assert [row["observed_churn"] for row in data] == [0, 1]
    for row in data:
        assert set(row["features"]) == set(FEATURES)
        assert "Churn" not in row["features"]


def test_examples_skip_rows_with_conflicting_labels(monkeypatch) -> None:
    def make_row(value: float, label: int) -> dict:
        row = {name: value for name in FEATURES}
        row.update({"Tariff Plan": 1, "Status": 1, "Age Group": 1, TARGET: label})
        return row

    # The first churn row shares its features with a non-churn row, so the
    # endpoint must skip it and return the unambiguous churn row instead.
    dataframe = pd.DataFrame([
        make_row(3.0, 1),
        make_row(3.0, 0),
        make_row(5.0, 0),
        make_row(7.0, 1),
    ])
    monkeypatch.setattr(analytics, "prepare_modeling_dataset", lambda path: dataframe)
    data = client.get("/api/analytics/examples").json()
    by_label = {row["observed_churn"]: row["features"] for row in data}
    assert by_label[0]["Age"] == 5.0
    assert by_label[1]["Age"] == 7.0


def test_prediction_rejects_invalid_dataset_domains() -> None:
    invalid_cases = [
        ("Complains", 2),
        ("Tariff Plan", 3),
        ("Status", 0),
        ("Charge Amount", 11),
        ("Age Group", 6),
        ("Age", -5),
    ]
    for field, invalid in invalid_cases:
        payload = _payload(0.5)
        payload[field] = invalid
        response = client.post("/api/predict", json=payload)
        assert response.status_code == 422, (field, response.text)
