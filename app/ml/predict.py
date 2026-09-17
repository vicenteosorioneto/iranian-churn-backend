"""Low-level inference helpers."""

from typing import Any

import numpy as np
import pandas as pd


def predict_with_pipeline(pipeline: Any, features: pd.DataFrame) -> tuple[int, float | None]:
    prediction = int(pipeline.predict(features)[0])
    probability: float | None = None

    if hasattr(pipeline, "predict_proba"):
        probabilities = pipeline.predict_proba(features)[0]
        classes = np.asarray(pipeline.classes_)
        positive_indices = np.flatnonzero(classes == 1)
        if positive_indices.size:
            probability = float(probabilities[positive_indices[0]])

    return prediction, probability
