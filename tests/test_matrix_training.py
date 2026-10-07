"""Training against the pooled combined matrix.

The matrix carries three columns that are NOT predictors -- ticker, date and
`source` -- plus a `has_news` flag. The legacy 9-ticker path inferred its
feature list and one-hot encoded ticker, which is wrong here twice over: a
one-hot over hundreds of tickers cannot generalise to an unseen ticker, and
`source` is a provenance string that encodes which tickers were re-sourced
during the data audit, not anything about future returns.

These tests pin that the pooled path feeds the model exactly the matrix's own
feature list, that provenance can never leak in, and that the walk-forward
folds stay chronological with the purge gap intact.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from app.evaluation.metrics import summarize_folds
from app.features.matrix import METADATA_COLUMNS, model_feature_columns
from app.models.train import NON_FEATURE_METADATA, feature_columns, run_training
from app.splits.purged_cv import purge_gap

MATRIX = Path("data/curated/matrix_h5_v4.parquet")
HORIZON = 5


@pytest.fixture(scope="module")
def matrix() -> pd.DataFrame:
    if not MATRIX.exists():
        pytest.skip("matrix_h5_v4 not built")
    return pd.read_parquet(MATRIX)


def synthetic_matrix(n_tickers: int = 6, n_days: int = 600) -> pd.DataFrame:
    """A frame with the matrix's schema, including the metadata columns."""
    rng = np.random.default_rng(0)
    dates = pd.bdate_range("2015-01-02", periods=n_days)
    rows = []
    for i in range(n_tickers):
        fwd = rng.normal(0, 0.03, n_days)
        rows.append(
            pd.DataFrame(
                {
                    "ticker": f"T{i}",
                    "date": dates,
                    "source": "fnspid" if i % 2 else "yfinance",
                    "ret_1d": rng.normal(0, 0.02, n_days),
                    "ret_5d": rng.normal(0, 0.04, n_days),
                    "rsi_14": rng.uniform(20, 80, n_days),
                    "article_count": rng.integers(0, 5, n_days),
                    "mean_sentiment": rng.normal(0, 0.2, n_days),
                    "has_news": rng.integers(0, 2, n_days).astype(bool),
                    f"fwd_ret_{HORIZON}d": fwd,
                    f"label_up_{HORIZON}d": (fwd > 0).astype(float),
                }
            )
        )
    return pd.concat(rows, ignore_index=True)


# --------------------------------------------------------------------------
# Provenance and identity must never become features
# --------------------------------------------------------------------------


def test_source_is_registered_as_non_feature_metadata():
    assert "source" in NON_FEATURE_METADATA
    assert "ticker" in NON_FEATURE_METADATA
    assert "date" in NON_FEATURE_METADATA


def test_inferred_columns_exclude_provenance():
    """Even on the legacy path, the provenance string must not be fed in: it
    would break the numeric imputer and encode data lineage."""
    df = synthetic_matrix()
    num, _ = feature_columns(df)
    assert "source" not in num
    assert "date" not in num


def test_inferred_columns_drop_non_numeric_rather_than_imputing_them():
    df = synthetic_matrix()
    df["some_text"] = "abc"
    num, _ = feature_columns(df)
    assert "some_text" not in num


def test_explicit_matrix_columns_match_the_matrix_definition(matrix):
    """Training and inference must read ONE definition of what feeds the
    model, or they will skew apart."""
    expected = model_feature_columns(matrix)
    num, cat = feature_columns(matrix, expected, [])
    assert num == expected
    assert cat == []


def test_ticker_is_not_a_feature_in_pooled_mode(matrix):
    num, cat = feature_columns(matrix, model_feature_columns(matrix), [])
    assert "ticker" not in num
    assert cat == [], "a one-hot over hundreds of tickers cannot generalise"


def test_no_target_column_can_reach_the_features(matrix):
    num, cat = feature_columns(matrix, model_feature_columns(matrix), [])
    leaked = [c for c in (*num, *cat) if c.startswith(("fwd_ret_", "label_up_"))]
    assert leaked == []


def test_metadata_columns_are_all_excluded(matrix):
    num, cat = feature_columns(matrix, model_feature_columns(matrix), [])
    assert not set(METADATA_COLUMNS) & set(num + cat)


def test_requesting_an_absent_column_fails_loudly():
    df = synthetic_matrix()
    with pytest.raises(KeyError, match="absent from the matrix"):
        run_training(
            df,
            model_name="majority",
            n_folds=3,
            embargo_days=5,
            horizon=HORIZON,
            feature_cols=["ret_1d", "a_column_that_does_not_exist"],
            cat_cols=[],
        )


# --------------------------------------------------------------------------
# Walk-forward geometry
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def trained() -> dict:
    df = synthetic_matrix()
    return run_training(
        df,
        model_name="logistic",
        n_folds=3,
        embargo_days=5,
        horizon=HORIZON,
        feature_cols=["ret_1d", "ret_5d", "rsi_14", "article_count", "mean_sentiment"],
        cat_cols=[],
    )


def test_every_fold_reports_its_validation_period_and_size(trained):
    for fold in trained["per_fold"]:
        for key in ("val_start", "val_end", "train_start", "train_end"):
            assert fold[key], f"fold {fold['fold']} missing {key}"
        assert fold["n_samples"] > 0
        assert fold["n_train"] > 0


def test_validation_always_follows_training_in_time(trained):
    """No fold may train on observations that postdate its validation block."""
    for fold in trained["per_fold"]:
        assert fold["train_end"] < fold["val_start"], (
            f"fold {fold['fold']} trains up to {fold['train_end']} but validates "
            f"from {fold['val_start']}"
        )


def test_the_purge_gap_separates_train_from_validation(trained):
    """The gap must be at least the label horizon, else a training row's label
    is built from validation-period prices."""
    gap = purge_gap(HORIZON, 5)
    assert trained["purge_gap"] == gap
    for fold in trained["per_fold"]:
        sessions = pd.bdate_range(fold["train_end"], fold["val_start"])
        assert len(sessions) - 1 >= HORIZON, (
            f"fold {fold['fold']}: only {len(sessions) - 1} sessions between "
            f"train end and val start; need at least the horizon ({HORIZON})"
        )


def test_folds_expand_and_do_not_overlap(trained):
    folds = trained["per_fold"]
    for earlier, later in zip(folds, folds[1:], strict=False):
        assert later["n_train"] > earlier["n_train"], "window must expand"
        assert later["val_start"] > earlier["val_end"], "validation blocks overlap"


def test_the_final_model_is_fit_short_of_the_test_block(trained):
    """The saved model must not train on rows whose labels come from test
    prices."""
    assert trained["n_train_final"] < trained["n_rows"]
    assert trained["test_start"] is not None


def test_run_metadata_records_what_was_trained(trained):
    assert trained["target"] == f"label_up_{HORIZON}d"
    assert trained["num_cols"], "feature list must travel with the run"
    assert trained["cat_cols"] == []
    assert trained["horizon"] == HORIZON


# --------------------------------------------------------------------------
# Fold aggregation
# --------------------------------------------------------------------------


def test_summarize_folds_ignores_descriptive_non_numeric_fields():
    """Date strings are not averageable; they must be skipped, not coerced."""
    per_fold = [
        {"roc_auc": 0.52, "n_samples": 100.0, "val_start": "2015-01-02", "fold": 1},
        {"roc_auc": 0.54, "n_samples": 120.0, "val_start": "2016-01-04", "fold": 2},
    ]
    summary = summarize_folds(per_fold)
    assert summary["roc_auc_mean"] == pytest.approx(0.53)
    assert summary["n_samples_total"] == pytest.approx(220.0)
    assert not any(k.startswith("val_start") for k in summary)


def test_summarize_folds_still_skips_none_metrics():
    per_fold = [
        {"roc_auc": None, "brier": 0.25, "n_samples": 10.0},
        {"roc_auc": None, "brier": 0.26, "n_samples": 10.0},
    ]
    summary = summarize_folds(per_fold)
    assert "roc_auc_mean" not in summary
    assert summary["brier_mean"] == pytest.approx(0.255)


# --------------------------------------------------------------------------
# The CLI resolves the feature set explicitly
# --------------------------------------------------------------------------


def test_auto_detects_the_pooled_matrix():
    from app.cli import _resolve_feature_set

    df = synthetic_matrix()
    cols, cat, resolved = _resolve_feature_set(df, "auto")
    assert resolved == "matrix"
    assert cat == []
    assert cols is not None and "source" not in cols


def test_auto_falls_back_to_legacy_without_the_metadata_columns():
    from app.cli import _resolve_feature_set

    df = synthetic_matrix().drop(columns=["source"])
    cols, cat, resolved = _resolve_feature_set(df, "legacy")
    assert resolved == "legacy"
    assert cols is None and cat is None


def test_matrix_mode_can_be_forced(matrix):
    from app.cli import _resolve_feature_set

    cols, cat, resolved = _resolve_feature_set(matrix, "matrix")
    assert resolved == "matrix"
    assert cols == model_feature_columns(matrix)
    assert cat == []
