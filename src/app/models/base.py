from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def _preprocessor(num_cols: list[str], cat_cols: list[str], scale: bool) -> ColumnTransformer:
    num_steps: list = [("impute", SimpleImputer(strategy="median"))]
    if scale:
        num_steps.append(("scale", StandardScaler()))
    return ColumnTransformer(
        [
            ("num", Pipeline(num_steps), num_cols),
            (
                "cat",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="most_frequent")),
                        ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                    ]
                ),
                cat_cols,
            ),
        ],
        sparse_threshold=0.0,
    )


class MajorityBaseline:
    """Predicts the constant train-set positive rate for every sample."""

    name = "majority"
    objective = "classification"

    def __init__(
        self,
        num_cols: list[str] | None = None,
        cat_cols: list[str] | None = None,
        class_weight: str | None = None,
    ):
        self.pos_rate_ = 0.5

    def fit(self, X: pd.DataFrame, y: pd.Series) -> MajorityBaseline:
        self.pos_rate_ = float(np.mean(y))
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        p = np.full(len(X), self.pos_rate_)
        return np.column_stack([1 - p, p])


class LogisticModel:
    name = "logistic"
    objective = "classification"

    def __init__(
        self,
        num_cols: list[str],
        cat_cols: list[str],
        class_weight: str | None = "balanced",
    ):
        self.pipeline = Pipeline(
            [
                ("prep", _preprocessor(num_cols, cat_cols, scale=True)),
                (
                    "clf",
                    LogisticRegression(max_iter=2000, C=0.5, class_weight=class_weight),
                ),
            ]
        )

    def fit(self, X: pd.DataFrame, y: pd.Series) -> LogisticModel:
        self.pipeline.fit(X, y)
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self.pipeline.predict_proba(X)


class LightGBMModel:
    """LightGBM gradient-boosted trees.

    Tree models are scale-invariant, so numeric features are imputed but
    NOT standardized — standardization is only applied where appropriate
    (the logistic model). Preprocessing lives inside the sklearn Pipeline,
    so it is refit from scratch on every training fold and can never see
    validation/test data.
    """

    name = "lightgbm"
    objective = "classification"

    def __init__(
        self,
        num_cols: list[str],
        cat_cols: list[str],
        class_weight: str | None = "balanced",
    ):
        import lightgbm as lgb

        self.pipeline = Pipeline(
            [
                ("prep", _preprocessor(num_cols, cat_cols, scale=False)),
                (
                    "clf",
                    lgb.LGBMClassifier(
                        n_estimators=300,
                        learning_rate=0.05,
                        num_leaves=31,
                        min_child_samples=50,
                        subsample=0.9,
                        colsample_bytree=0.9,
                        class_weight=class_weight,
                        verbosity=-1,
                    ),
                ),
            ]
        )

    def fit(self, X: pd.DataFrame, y: pd.Series) -> LightGBMModel:
        self.pipeline.fit(X, y)
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self.pipeline.predict_proba(X)


class BoostedTreesModel:
    """sklearn HistGradientBoosting fallback when LightGBM is unavailable."""

    name = "hgb"
    objective = "classification"

    def __init__(
        self,
        num_cols: list[str],
        cat_cols: list[str],
        class_weight: str | None = "balanced",
    ):
        from sklearn.ensemble import HistGradientBoostingClassifier

        self.pipeline = Pipeline(
            [
                ("prep", _preprocessor(num_cols, cat_cols, scale=False)),
                (
                    "clf",
                    HistGradientBoostingClassifier(
                        max_iter=200, learning_rate=0.05, class_weight=class_weight
                    ),
                ),
            ]
        )

    def fit(self, X: pd.DataFrame, y: pd.Series) -> BoostedTreesModel:
        self.pipeline.fit(X, y)
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self.pipeline.predict_proba(X)
