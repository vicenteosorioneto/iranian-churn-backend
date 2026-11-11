"""Model metadata endpoints."""

import json
from json import JSONDecodeError

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field

from app.api.schemas.metrics import MetricsResponse
from app.config import METRICS_PATH, FEATURE_IMPORTANCE_PATH

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


class FeatureImportanceItem(BaseModel):
    feature: str
    importance_mean: float
    importance_std: float = Field(ge=0)


class FeatureImportanceResponse(BaseModel):
    model_name: str
    scoring: str
    features: list[FeatureImportanceItem]


@router.get("/feature-importance", response_model=FeatureImportanceResponse)
def get_feature_importance() -> FeatureImportanceResponse:
    if not FEATURE_IMPORTANCE_PATH.is_file():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Feature importance not found. Run 'python -m app.ml.train' first.",
        )
    try:
        payload = json.loads(FEATURE_IMPORTANCE_PATH.read_text(encoding="utf-8"))
        return FeatureImportanceResponse.model_validate(payload)
    except (OSError, JSONDecodeError, ValueError) as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Feature importance could not be read.",
        ) from exc
