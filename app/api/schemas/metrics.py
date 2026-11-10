"""Schemas for persisted model metrics."""

from pydantic import BaseModel, Field


class RocPoint(BaseModel):
    fpr: float = Field(ge=0.0, le=1.0)
    tpr: float = Field(ge=0.0, le=1.0)



class ModelMetrics(BaseModel):
    accuracy: float = Field(ge=0.0, le=1.0)
    balanced_accuracy: float = Field(ge=0.0, le=1.0)
    precision: float = Field(ge=0.0, le=1.0)
    recall: float = Field(ge=0.0, le=1.0)
    f1: float = Field(ge=0.0, le=1.0)
    roc_auc: float = Field(ge=0.0, le=1.0)
    average_precision: float = Field(ge=0.0, le=1.0)
    confusion_matrix: list[list[int]]
    roc_curve: list[RocPoint] = Field(default_factory=list)


class MetricsResponse(BaseModel):
    logistic_regression: ModelMetrics
    random_forest: ModelMetrics
