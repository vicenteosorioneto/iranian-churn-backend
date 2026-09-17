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
    call_failure: float = Field(alias="Call Failure")
    subscription_length: float = Field(alias="Subscription Length")
    seconds_of_use: float = Field(alias="Seconds of Use")
    frequency_of_use: float = Field(alias="Frequency of use")
    frequency_of_sms: float = Field(alias="Frequency of SMS")
    distinct_called_numbers: float = Field(alias="Distinct Called Numbers")
    age: float = Field(alias="Age")
    customer_value: float = Field(alias="Customer Value")
    complains: int = Field(alias="Complains")
    tariff_plan: int = Field(alias="Tariff Plan")
    status: int = Field(alias="Status")
    charge_amount: int = Field(alias="Charge Amount")
    age_group: int = Field(alias="Age Group")

    def feature_frame(self) -> pd.DataFrame:
        values = self.model_dump(by_alias=True, exclude={"model_name"})
        return pd.DataFrame([values], columns=FEATURES)


class PredictionResponse(BaseModel):
    model_name: ModelName
    prediction: int
    churn_probability: float | None
