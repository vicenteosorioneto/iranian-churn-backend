from app.ml.features import (
    BINARY_FEATURES,
    FEATURES,
    NUMERIC_FEATURES,
    ORDINAL_FEATURES,
    TARGET,
)


def test_feature_groups_are_complete_and_disjoint() -> None:
    groups = [set(NUMERIC_FEATURES), set(BINARY_FEATURES), set(ORDINAL_FEATURES)]

    assert len(FEATURES) == 13
    assert len(set(FEATURES)) == 13
    assert all(groups[index].isdisjoint(groups[other]) for index in range(3) for other in range(index + 1, 3))
    assert TARGET == "Churn"
