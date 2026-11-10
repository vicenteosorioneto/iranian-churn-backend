"""Public aggregate analytics from the same deduplicated data used for training."""

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field

from app.config import DATASET_PATH
from app.ml.data import prepare_modeling_dataset
from app.ml.features import TARGET

router = APIRouter(prefix="/analytics", tags=["analytics"])

GROUP_FIELDS = ("Age Group", "Complains", "Status")


class CategorySummary(BaseModel):
    value: int
    total: int = Field(ge=0)
    churn: int = Field(ge=0)
    churn_rate: float = Field(ge=0, le=1)


class AnalyticsSummary(BaseModel):
    total_clients: int = Field(ge=0)
    churn_clients: int = Field(ge=0)
    retained_clients: int = Field(ge=0)
    churn_rate: float = Field(ge=0, le=1)
    distributions: dict[str, list[CategorySummary]]


@router.get("/summary", response_model=AnalyticsSummary)
def get_analytics_summary() -> AnalyticsSummary:
    """Return real aggregated observations (not predictions or invented figures)."""
    try:
        dataframe = prepare_modeling_dataset(DATASET_PATH)
    except (FileNotFoundError, ValueError, OSError) as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="The dataset is unavailable or invalid. Place data/raw/iranian_churn.csv first.",
        ) from exc

    total = int(len(dataframe))
    churn = int(dataframe[TARGET].sum())
    distributions: dict[str, list[CategorySummary]] = {}
    for column in GROUP_FIELDS:
        grouped = dataframe.groupby(column, dropna=False)[TARGET].agg(["count", "sum"])
        distributions[column] = [
            CategorySummary(
                value=int(value),
                total=int(row["count"]),
                churn=int(row["sum"]),
                churn_rate=float(row["sum"] / row["count"]),
            )
            for value, row in grouped.sort_index().iterrows()
        ]

    return AnalyticsSummary(
        total_clients=total,
        churn_clients=churn,
        retained_clients=total - churn,
        churn_rate=churn / total if total else 0,
        distributions=distributions,
    )
