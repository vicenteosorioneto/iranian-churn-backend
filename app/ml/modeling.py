"""Leakage-safe preparation of a shared train/test split."""

from dataclasses import dataclass

import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from app.ml.features import FEATURES, TARGET

N_SPLITS = 5
RANDOM_STATE = 42


@dataclass(frozen=True)
class ModelingSplit:
    x_train: pd.DataFrame
    x_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    train_groups: pd.Series
    test_groups: pd.Series

    @property
    def shared_groups(self) -> set[int]:
        return set(self.train_groups).intersection(self.test_groups)


def create_feature_group_ids(dataframe: pd.DataFrame) -> pd.Series:
    """Assign the same identifier to rows with identical model features."""
    feature_index = pd.MultiIndex.from_frame(dataframe.loc[:, FEATURES])
    group_ids, _ = pd.factorize(feature_index, sort=False)
    return pd.Series(group_ids, index=dataframe.index, name="feature_group")


def create_grouped_split(dataframe: pd.DataFrame) -> ModelingSplit:
    """Use one stratified group fold for test and the other four for training."""
    x = dataframe.loc[:, FEATURES]
    y = dataframe.loc[:, TARGET]
    groups = create_feature_group_ids(dataframe)
    splitter = StratifiedGroupKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )
    train_positions, test_positions = next(splitter.split(x, y, groups))

    split = ModelingSplit(
        x_train=x.iloc[train_positions].copy(),
        x_test=x.iloc[test_positions].copy(),
        y_train=y.iloc[train_positions].copy(),
        y_test=y.iloc[test_positions].copy(),
        train_groups=groups.iloc[train_positions].copy(),
        test_groups=groups.iloc[test_positions].copy(),
    )
    if split.shared_groups:
        raise ValueError("Feature groups are shared between training and test sets.")
    return split
