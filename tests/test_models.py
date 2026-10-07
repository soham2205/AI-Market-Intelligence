from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.evaluation.metrics import classification_metrics, summarize_folds
from app.models.registry import create_model


def test_perfect_classifier_auc():
    y = np.array([0, 0, 0, 1, 1, 1])
    p = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
    m = classification_metrics(y, p)
    assert m["roc_auc"] == 1.0
    assert m["brier"] < 0.05


def test_single_class_auc_is_none():
    y = np.array([1, 1, 1])
    p = np.array([0.8, 0.9, 0.7])
    assert classification_metrics(y, p)["roc_auc"] is None


def test_summarize_folds_aggregates():
    per_fold = [
        {"roc_auc": 0.5, "n_samples": 10},
        {"roc_auc": 0.7, "n_samples": 10},
    ]
    s = summarize_folds(per_fold)
    assert s["roc_auc_mean"] == pytest.approx(0.6)
    assert s["roc_auc_std"] == pytest.approx(0.1)
    assert s["n_samples_total"] == 20


def test_registry_unknown_model_raises():
    with pytest.raises(KeyError):
        create_model("does_not_exist", ["f1"], ["ticker"])


def test_majority_baseline_constant_proba():
    model = create_model("majority", [], [])
    X = pd.DataFrame({"a": [1, 2, 3]})
    y = pd.Series([1, 0, 1])
    model.fit(X, y)
    proba = model.predict_proba(X)
    assert np.allclose(proba[:, 1], 2 / 3)
    assert np.allclose(proba.sum(axis=1), 1.0)


def test_logistic_fits_and_predicts_shape(make_panel):
    panel = make_panel(tickers=("AAPL", "MSFT"), n=120)
    from app.features.build import build_features, is_feature_column

    feats = build_features(panel).dropna(subset=["label_up_1d"])
    num_cols = [c for c in feats.columns if is_feature_column(c) and c != "ticker"]
    model = create_model("logistic", num_cols, ["ticker"])
    X = feats[num_cols + ["ticker"]]
    y = feats["label_up_1d"].astype(int)
    model.fit(X.iloc[:150], y.iloc[:150])
    proba = model.predict_proba(X.iloc[150:])
    assert proba.shape == (len(feats) - 150, 2)
    assert np.all((proba >= 0) & (proba <= 1))
