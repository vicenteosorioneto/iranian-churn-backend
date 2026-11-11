"""Schemas for model inference."""

from typing import Literal

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from app.ml.features import FEATURES

ModelName = Literal["logistic_regression", "random_forest"]


class PredictionRequest(BaseModel):
    """One customer observation, using dataset column names as JSON aliases."""

    model_config = ConfigDict(populate_by_name=True)

    model_name: ModelName = "random_forest"
    threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    call_failure: float = Field(alias="Call Failure", ge=0, allow_inf_nan=False)
    subscription_length: float = Field(alias="Subscription Length", ge=0, allow_inf_nan=False)
    seconds_of_use: float = Field(alias="Seconds of Use", ge=0, allow_inf_nan=False)
    frequency_of_use: float = Field(alias="Frequency of use", ge=0, allow_inf_nan=False)
    frequency_of_sms: float = Field(alias="Frequency of SMS", ge=0, allow_inf_nan=False)
    distinct_called_numbers: float = Field(alias="Distinct Called Numbers", ge=0, allow_inf_nan=False)
    age: float = Field(alias="Age", ge=0, allow_inf_nan=False)
    customer_value: float = Field(alias="Customer Value", ge=0, allow_inf_nan=False)
    complains: int = Field(alias="Complains", ge=0, le=1)
    tariff_plan: int = Field(alias="Tariff Plan", ge=1, le=2)
    status: int = Field(alias="Status", ge=1, le=2)
    charge_amount: int = Field(alias="Charge Amount", ge=0, le=10)
    age_group: int = Field(alias="Age Group", ge=1, le=5)

    def feature_frame(self) -> pd.DataFrame:
        values = self.model_dump(by_alias=True, exclude={"model_name", "threshold"})
        return pd.DataFrame([values], columns=FEATURES)


class PredictionResponse(BaseModel):
    model_name: ModelName
    prediction: int
    churn_probability: float | None
    decision_threshold: float = Field(ge=0.0, le=1.0)
