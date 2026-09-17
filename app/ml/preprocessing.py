"""Leakage-safe preprocessing and estimator pipelines."""

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from app.ml.features import BINARY_FEATURES, NUMERIC_FEATURES, ORDINAL_FEATURES


def _numeric_transformer(*, scale: bool) -> Pipeline:
    steps: list[tuple[str, object]] = [("imputer", SimpleImputer(strategy="median"))]
    if scale:
        steps.append(("scaler", StandardScaler()))
    return Pipeline(steps=steps)


def _categorical_transformer() -> Pipeline:
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )


def _ordinal_transformer() -> Pipeline:
    return Pipeline(steps=[("imputer", SimpleImputer(strategy="most_frequent"))])


def build_preprocessor(*, scale_numeric: bool) -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("numeric", _numeric_transformer(scale=scale_numeric), NUMERIC_FEATURES),
            ("binary", _categorical_transformer(), BINARY_FEATURES),
            ("ordinal", _ordinal_transformer(), ORDINAL_FEATURES),
        ],
        remainder="drop",
    )


def build_logistic_regression_pipeline() -> Pipeline:
    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessor(scale_numeric=True)),
            (
                "classifier",
                LogisticRegression(max_iter=1_000, class_weight="balanced", random_state=42),
            ),
        ]
    )


def build_random_forest_pipeline() -> Pipeline:
    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessor(scale_numeric=False)),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=300,
                    class_weight="balanced",
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
        ]
    )
