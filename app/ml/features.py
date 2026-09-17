"""Canonical feature groups for the Iranian Churn Dataset."""

# Continuous or count measurements. These are scaled for Logistic Regression.
NUMERIC_FEATURES: tuple[str, ...] = (
    "Call Failure",
    "Subscription Length",
    "Seconds of Use",
    "Frequency of use",
    "Frequency of SMS",
    "Distinct Called Numbers",
    "Age",
    "Customer Value",
)

# Discrete labels without a meaningful numeric distance; they are one-hot encoded.
BINARY_FEATURES: tuple[str, ...] = (
    "Complains",
    "Tariff Plan",
    "Status",
)

# Ordered discrete levels. Their numeric order is intentionally preserved.
# The source documentation describes Charge Amount as 0-9, but the official CSV
# contains 7 observations at level 10. We preserve the raw observations and use
# the empirically observed ordered domain 0-10 throughout the pipeline.
ORDINAL_FEATURES: tuple[str, ...] = (
    "Charge Amount",
    "Age Group",
)

FEATURES: tuple[str, ...] = NUMERIC_FEATURES + BINARY_FEATURES + ORDINAL_FEATURES
TARGET = "Churn"
REQUIRED_COLUMNS: tuple[str, ...] = FEATURES + (TARGET,)

# Allowed values documented by the dataset specification.
FEATURE_DOMAINS: dict[str, frozenset[int]] = {
    "Complains": frozenset({0, 1}),
    "Tariff Plan": frozenset({1, 2}),
    "Status": frozenset({1, 2}),
    "Charge Amount": frozenset(range(11)),
    "Age Group": frozenset(range(1, 6)),
    TARGET: frozenset({0, 1}),
}
