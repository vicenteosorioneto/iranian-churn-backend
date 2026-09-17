"""Application paths shared by the API and machine-learning modules."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_PATH = PROJECT_ROOT / "data" / "raw" / "iranian_churn.csv"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
METRICS_PATH = REPORTS_DIR / "metrics.json"

MODEL_PATHS = {
    "logistic_regression": MODELS_DIR / "logistic_regression.joblib",
    "random_forest": MODELS_DIR / "random_forest.joblib",
}
