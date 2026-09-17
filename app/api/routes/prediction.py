"""Churn prediction endpoint."""

from fastapi import APIRouter, HTTPException, status

from app.api.schemas.prediction import PredictionRequest, PredictionResponse
from app.services.prediction_service import ModelNotAvailableError, predict_churn

router = APIRouter(tags=["predictions"])


@router.post("/predict", response_model=PredictionResponse)
def create_prediction(request: PredictionRequest) -> PredictionResponse:
    try:
        prediction, probability = predict_churn(
            model_name=request.model_name,
            features=request.feature_frame(),
        )
    except ModelNotAvailableError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        ) from exc

    return PredictionResponse(
        model_name=request.model_name,
        prediction=prediction,
        churn_probability=probability,
    )
