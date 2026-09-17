"""Model metadata endpoints."""

import json
from json import JSONDecodeError

from fastapi import APIRouter, HTTPException, status

from app.api.schemas.metrics import MetricsResponse
from app.config import METRICS_PATH

router = APIRouter(prefix="/models", tags=["models"])


@router.get("/metrics", response_model=MetricsResponse)
def get_model_metrics() -> MetricsResponse:
    if not METRICS_PATH.is_file():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Metrics not found. Run 'python -m app.ml.train' first.",
        )

    try:
        payload = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
        return MetricsResponse.model_validate(payload)
    except (OSError, JSONDecodeError, ValueError) as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="The metrics file could not be read.",
        ) from exc
