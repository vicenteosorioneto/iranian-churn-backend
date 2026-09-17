"""Load persisted pipelines and coordinate inference."""

from functools import lru_cache
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from app.config import MODEL_PATHS
from app.ml.predict import predict_with_pipeline


class ModelNotAvailableError(RuntimeError):
    """Raised when a trained pipeline is not available for inference."""


@lru_cache(maxsize=len(MODEL_PATHS))
def _load_pipeline(path: Path) -> Any:
    if not path.is_file():
        raise ModelNotAvailableError(
            f"Model '{path.name}' not found. Run 'python -m app.ml.train' first."
        )
    try:
        return joblib.load(path)
    except (OSError, ValueError) as exc:
        raise ModelNotAvailableError(f"Model '{path.name}' could not be loaded.") from exc


def predict_churn(model_name: str, features: pd.DataFrame) -> tuple[int, float | None]:
    try:
        model_path = MODEL_PATHS[model_name]
    except KeyError as exc:
        raise ValueError(f"Unknown model: {model_name}") from exc
    return predict_with_pipeline(_load_pipeline(model_path), features)
