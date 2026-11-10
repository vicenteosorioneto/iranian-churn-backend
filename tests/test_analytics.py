"""Integration coverage for the browser-facing analytics API."""
import pandas as pd
from fastapi.testclient import TestClient

from app.main import app
from app.api.routes import analytics
from app.ml.features import TARGET

client = TestClient(app)


def test_cors_allows_local_frontend_preflight() -> None:
    response = client.options(
        "/api/analytics/summary",
        headers={
            "Origin": "http://localhost:3000",
            "Access-Control-Request-Method": "GET",
        },
    )
    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == "http://localhost:3000"


def test_analytics_returns_real_aggregates(monkeypatch) -> None:
    observations = pd.DataFrame(
        {
            "Age Group": [1, 1, 2, 2],
            "Complains": [0, 1, 0, 1],
            "Status": [1, 1, 2, 2],
            TARGET: [0, 1, 0, 1],
        }
    )
    monkeypatch.setattr(
        analytics, "prepare_modeling_dataset", lambda path: observations
    )
    response = client.get("/api/analytics/summary")
    assert response.status_code == 200
    body = response.json()
    assert body["total_clients"] == 4
    assert body["churn_clients"] == 2
    assert body["retained_clients"] == 2
    assert body["churn_rate"] == 0.5
    assert body["distributions"]["Complains"][1] == {
        "value": 1,
        "total": 2,
        "churn": 2,
        "churn_rate": 1.0,
    }


def test_analytics_does_not_invent_data_without_dataset(monkeypatch) -> None:
    def missing(_):
        raise FileNotFoundError("missing")
    monkeypatch.setattr(analytics, "prepare_modeling_dataset", missing)
    response = client.get("/api/analytics/summary")
    assert response.status_code == 503
