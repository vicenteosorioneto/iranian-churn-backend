"""End-to-end API demo smoke test using the official CSV and trained artifacts."""

from fastapi.testclient import TestClient

from app.config import DATASET_PATH
from app.main import app
from app.ml.data import prepare_modeling_dataset
from app.ml.features import FEATURES


def main() -> None:
    if not DATASET_PATH.is_file():
        raise SystemExit("Missing data/raw/iranian_churn.csv")
    dataframe = prepare_modeling_dataset(DATASET_PATH)
    client = TestClient(app)

    metrics = client.get("/api/models/metrics")
    assert metrics.status_code == 200, metrics.text
    metrics_body = metrics.json()
    assert len(metrics_body["random_forest"]["roc_curve"]) > 1

    summary = client.get("/api/analytics/summary")
    assert summary.status_code == 200, summary.text
    assert summary.json()["total_clients"] == len(dataframe)

    importance = client.get("/api/models/feature-importance")
    assert importance.status_code == 200, importance.text
    assert len(importance.json()["features"]) == len(FEATURES)

    record = dataframe.iloc[0]
    payload = {key: float(record[key]) for key in FEATURES}
    payload["model_name"] = "random_forest"
    response = client.post("/api/predict", json=payload)
    assert response.status_code == 200, response.text
    result = response.json()
    assert result["prediction"] in [0, 1]
    assert result["model_name"] == "random_forest"
    assert 0.0 <= result["churn_probability"] <= 1.0

    print("Real-data demo checks passed.")
    print("Total records analyzed:", summary.json()["total_clients"])
    print("Random Forest F1:", metrics_body["random_forest"]["f1"])
    print("Sample prediction:", result)


if __name__ == "__main__":
    main()
