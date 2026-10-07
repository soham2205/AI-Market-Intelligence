from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.evaluation.metrics import classification_metrics
from app.features.build import build_features, is_feature_column
from app.labeling.targets import fwd_ret_col, is_target_column, label_col
from app.models.registry import MODEL_REGISTRY, create_model
from app.models.train import feature_columns, run_training
from app.splits.purged_cv import purge_gap, split_frame

HORIZONS = (1, 5, 21, 63, 126, 252)
EMBARGO = 10


@pytest.fixture(scope="module")
def real_features() -> pd.DataFrame:
    path = "data/curated/features.parquet"
    return pd.read_parquet(path)


# --------------------------------------------------------------------------
# purge / embargo geometry
# --------------------------------------------------------------------------


def test_purge_gap_is_horizon_derived_not_constant():
    """The leakage-critical term is the horizon itself; the embargo is a
    separate, horizon-independent hygiene constant."""
    for k in HORIZONS:
        assert purge_gap(k, EMBARGO) == k + EMBARGO
    # A 252-day label needs 252 sessions of purge, not a fixed 10.
    assert purge_gap(252, 10) == 262
    assert purge_gap(1, 10) == 11
    # The embargo term alone never varies with the horizon.
    assert purge_gap(252, 0) - purge_gap(1, 0) == 251


def test_purge_gap_rejects_invalid_input():
    with pytest.raises(ValueError):
        purge_gap(0, 10)
    with pytest.raises(ValueError):
        purge_gap(5, -1)


@pytest.mark.parametrize("horizon", HORIZONS)
def test_no_future_data_enters_training(real_features, horizon):
    """For every fold the latest training session must precede the earliest
    validation session by more than the horizon-derived purge gap."""
    target = label_col(horizon)
    df = (
        real_features.dropna(subset=[target])
        .sort_values(["date", "ticker"])
        .reset_index(drop=True)
    )
    folds, test_idx, _ = split_frame(
        df, n_folds=5, embargo_days=EMBARGO, horizon=horizon
    )

    unique_dates = np.sort(df["date"].unique())
    rank = {d: i for i, d in enumerate(unique_dates)}
    test_dates = set(df.loc[test_idx, "date"])

    for fold in folds:
        train_dates = set(df.loc[fold.train_idx, "date"])
        val_dates = set(df.loc[fold.val_idx, "date"])
        assert not (train_dates & val_dates), "train/val share sessions"
        assert not (train_dates & test_dates), "training saw the test period"
        gap = min(rank[d] for d in val_dates) - max(rank[d] for d in train_dates)
        # Strictly greater than the horizon: a training label spanning
        # [T, T+k] must not reach val_start.
        assert gap > horizon, f"h={horizon} fold {fold.fold}: label overlap gap={gap}"
        assert gap > purge_gap(horizon, EMBARGO) - 1


@pytest.mark.parametrize("horizon", HORIZONS)
def test_validation_labels_never_reach_the_test_block(real_features, horizon):
    """Fold N's validation must stop short of the test block, or its labels
    would be built from test-period prices."""
    target = label_col(horizon)
    df = (
        real_features.dropna(subset=[target])
        .sort_values(["date", "ticker"])
        .reset_index(drop=True)
    )
    folds, test_idx, _ = split_frame(
        df, n_folds=5, embargo_days=EMBARGO, horizon=horizon
    )
    rank = {d: i for i, d in enumerate(np.sort(df["date"].unique()))}

    first_test_rank = min(rank[d] for d in set(df.loc[test_idx, "date"]))
    last_val_rank = max(rank[d] for f in folds for d in set(df.loc[f.val_idx, "date"]))
    assert first_test_rank - last_val_rank > horizon


@pytest.mark.parametrize("horizon", HORIZONS)
def test_final_fit_is_purged_from_the_test_block(real_features, horizon):
    """The final model -- saved, promoted, scored on test -- must not train on
    rows whose label window reaches into the test period."""
    target = label_col(horizon)
    df = (
        real_features.dropna(subset=[target])
        .sort_values(["date", "ticker"])
        .reset_index(drop=True)
    )
    folds, test_idx, final_fit_idx = split_frame(
        df, n_folds=5, embargo_days=EMBARGO, horizon=horizon
    )
    rank = {d: i for i, d in enumerate(np.sort(df["date"].unique()))}

    first_test_rank = min(rank[d] for d in set(df.loc[test_idx, "date"]))
    last_fit_rank = max(rank[d] for d in set(df.loc[final_fit_idx, "date"]))
    assert first_test_rank - last_fit_rank > horizon, (
        f"h={horizon}: final fit ends {first_test_rank - last_fit_rank} sessions "
        f"before test, but labels span {horizon}"
    )
    assert not set(final_fit_idx) & set(test_idx)


def test_cv_folds_are_chronologically_correct(real_features):
    """Folds must move forward in time and training windows must expand."""
    df = (
        real_features.dropna(subset=["label_up_1d"])
        .sort_values("date")
        .reset_index(drop=True)
    )
    folds, _, _ = split_frame(df, n_folds=5, embargo_days=EMBARGO, horizon=1)

    val_starts = [df.loc[f.val_idx, "date"].min() for f in folds]
    val_ends = [df.loc[f.val_idx, "date"].max() for f in folds]
    assert val_starts == sorted(val_starts)
    assert val_ends == sorted(val_ends)
    for earlier, later in zip(folds[:-1], folds[1:], strict=False):
        assert (
            df.loc[later.val_idx, "date"].min() > df.loc[earlier.val_idx, "date"].max()
        )

    train_sizes = [len(f.train_idx) for f in folds]
    assert train_sizes == sorted(train_sizes), "expanding window violated"


@pytest.mark.parametrize("horizon", [1, 5, 21, 63])
def test_embargo_purges_overlapping_label_windows(horizon):
    """A k-day label built from T+k must never overlap a validation window
    starting inside [T+1, T+k]. Synthetic panel, exact rank arithmetic."""
    n_days = 900
    dates = pd.bdate_range("2020-01-02", periods=n_days)
    df = pd.DataFrame(
        {
            "date": np.repeat(dates, 2),
            "ticker": ["A", "B"] * n_days,
            "close": np.linspace(100, 200, n_days * 2),
            fwd_ret_col(horizon): 0.01,
        }
    )
    folds, test_idx, _ = split_frame(
        df, n_folds=4, embargo_days=EMBARGO, horizon=horizon
    )

    rank = {d: i for i, d in enumerate(np.sort(dates))}
    for fold in folds:
        last_train_day = max(rank[d] for d in set(df.loc[fold.train_idx, "date"]))
        first_val_day = min(rank[d] for d in set(df.loc[fold.val_idx, "date"]))
        assert first_val_day - last_train_day > horizon + EMBARGO - 1

    first_test_day = min(rank[d] for d in set(df.loc[test_idx, "date"]))
    max_val_day = max(
        rank[d] for fold in folds for d in set(df.loc[fold.val_idx, "date"])
    )
    assert first_test_day > max_val_day + horizon


def test_infeasible_geometry_raises_rather_than_reducing_folds():
    """Folds are never silently reduced: an impossible request must raise."""
    n_days = 300
    dates = pd.bdate_range("2023-01-02", periods=n_days)
    df = pd.DataFrame({"date": dates, "ticker": "A", "close": 100.0})
    with pytest.raises(ValueError, match="horizon=252"):
        split_frame(df, n_folds=5, embargo_days=EMBARGO, horizon=252)


# --------------------------------------------------------------------------
# cross-horizon leakage
# --------------------------------------------------------------------------


def test_no_horizon_target_ever_becomes_a_feature(real_features):
    """THE cross-horizon leak guard: with all six horizons materialised, not
    one of the twelve target columns may enter the feature matrix."""
    num_cols, cat_cols = feature_columns(real_features)
    all_targets = {label_col(k) for k in HORIZONS} | {fwd_ret_col(k) for k in HORIZONS}
    assert all_targets <= set(real_features.columns), "fixture lacks all horizons"
    leaked = all_targets & set(num_cols + cat_cols)
    assert not leaked, f"targets leaked into features: {sorted(leaked)}"


def test_target_exclusion_is_prefix_based_not_enumerated():
    """A horizon nobody anticipated must still be excluded automatically."""
    assert is_target_column("label_up_9999d")
    assert is_target_column("fwd_ret_9999d")
    assert not is_feature_column("label_up_9999d")
    # ...while genuine backward-looking features survive the filter.
    assert is_feature_column("ret_1d")
    assert is_feature_column("ret_21d")
    assert is_feature_column("rsi_14")


def test_feature_columns_identical_across_horizons(real_features):
    """Every horizon trains on the SAME feature matrix; only the label moves.
    A column set that shifted per horizon would mean a target is leaking."""
    num_cols, cat_cols = feature_columns(real_features)
    for k in HORIZONS:
        df = real_features.dropna(subset=[label_col(k)])
        n, c = feature_columns(df)
        assert n == num_cols and c == cat_cols


def test_leak_assertion_fires_on_a_target_column():
    """The runtime guard must raise rather than train a quietly leaky model."""
    from app.models.train import _assert_no_target_leakage

    with pytest.raises(AssertionError, match="leaked into the feature matrix"):
        _assert_no_target_leakage(["ret_1d", "label_up_5d"], ["ticker"])


# --------------------------------------------------------------------------
# label semantics
# --------------------------------------------------------------------------


@pytest.mark.parametrize("horizon", HORIZONS)
def test_cumulative_label_matches_raw_price_definition(horizon, make_panel):
    """label_up_kd[T] == (Close[T+k]/Close[T] - 1 > 0), from raw prices."""
    panel = make_panel(tickers=("AAPL",), n=400)
    feats = (
        build_features(panel, horizons=HORIZONS)
        .sort_values("date")
        .reset_index(drop=True)
    )
    close = panel.sort_values("date")["close"].reset_index(drop=True)

    expected_ret = close.shift(-horizon) / close - 1
    expected_lab = (expected_ret > 0).astype(float).where(expected_ret.notna())

    np.testing.assert_allclose(
        feats[fwd_ret_col(horizon)].to_numpy(dtype=float),
        expected_ret.to_numpy(dtype=float),
        rtol=1e-12,
        equal_nan=True,
    )
    pd.testing.assert_series_equal(
        feats[label_col(horizon)], expected_lab, check_names=False
    )


@pytest.mark.parametrize("horizon", HORIZONS)
def test_label_tail_is_nan_for_last_k_rows_per_ticker(horizon, make_panel):
    """The trailing k rows of EACH ticker have no T+k and must stay NaN."""
    panel = make_panel(tickers=("AAPL", "MSFT"), n=400)
    feats = build_features(panel, horizons=HORIZONS)
    for ticker in ("AAPL", "MSFT"):
        g = feats[feats["ticker"] == ticker].sort_values("date")
        col = g[label_col(horizon)]
        assert col.iloc[-horizon:].isna().all()
        assert col.iloc[:-horizon].notna().all()


def test_horizon_labels_do_not_bleed_across_tickers(make_panel):
    """The shift must run inside each ticker group: ticker A's tail must never
    borrow ticker B's head."""
    panel = make_panel(tickers=("AAPL", "MSFT"), n=300)
    feats = build_features(panel, horizons=(21,))
    alone = build_features(panel[panel["ticker"] == "AAPL"], horizons=(21,))
    a_from_panel = (
        feats[feats["ticker"] == "AAPL"].sort_values("date")["fwd_ret_21d"].to_numpy()
    )
    a_alone = alone.sort_values("date")["fwd_ret_21d"].to_numpy()
    np.testing.assert_allclose(a_from_panel, a_alone, rtol=1e-12, equal_nan=True)


# --------------------------------------------------------------------------
# preprocessing isolation (unchanged guarantees)
# --------------------------------------------------------------------------


def test_preprocessing_is_fit_on_training_folds_only():
    """Standardization statistics must come from the training fold alone."""
    rng = np.random.default_rng(7)
    n_train, n_val = 300, 80
    x_train = pd.DataFrame(
        {"f_num": rng.normal(0, 1, n_train), "ticker": ["A"] * n_train}
    )
    y_train = pd.Series(rng.integers(0, 2, n_train))
    x_val = pd.DataFrame({"f_num": rng.normal(100, 5, n_val), "ticker": ["A"] * n_val})
    y_val = pd.Series(rng.integers(0, 2, n_val))

    model = create_model("logistic", ["f_num"], ["ticker"])
    model.fit(x_train, y_train)

    num_pipe = model.pipeline.named_steps["prep"].named_transformers_["num"]
    scaler_mean = float(num_pipe.named_steps["scale"].mean_[0])

    assert scaler_mean == pytest.approx(x_train["f_num"].mean())
    assert abs(scaler_mean) < 0.2

    full_mean = float(np.concatenate([x_train["f_num"], x_val["f_num"]]).mean())
    assert abs(scaler_mean - full_mean) > 20

    metrics = classification_metrics(y_val.to_numpy(), model.predict_proba(x_val)[:, 1])
    assert metrics["n_samples"] == n_val


def test_preprocessing_refit_per_fold_not_shared():
    """Each fold's pipeline must be built fresh: no state carried between folds."""
    rng = np.random.default_rng(11)
    stats = []
    for shift in (0.0, 25.0):
        X = pd.DataFrame({"f_num": rng.normal(shift, 1, 200), "ticker": ["A"] * 200})
        y = pd.Series(rng.integers(0, 2, 200))
        model = create_model("logistic", ["f_num"], ["ticker"])
        model.fit(X, y)
        num_pipe = model.pipeline.named_steps["prep"].named_transformers_["num"]
        stats.append(float(num_pipe.named_steps["scale"].mean_[0]))
    assert abs(stats[1] - stats[0]) > 20


# --------------------------------------------------------------------------
# training end-to-end, per horizon
# --------------------------------------------------------------------------


@pytest.mark.parametrize("model_name", ["logistic", "lightgbm", "majority"])
def test_models_train_on_real_dataset(real_features, model_name):
    result = run_training(
        real_features, model_name=model_name, n_folds=3, embargo_days=7, horizon=1
    )
    summary = result["summary"]
    assert np.isfinite(summary["roc_auc_mean"])
    assert np.isfinite(summary["pr_auc_mean"])
    assert 0.0 <= summary["roc_auc_mean"] <= 1.0
    for key in ("precision_mean", "recall_mean", "f1_mean"):
        assert key in summary and 0.0 <= summary[key] <= 1.0

    preds = result["predictions"]
    assert {"val_fold_1", "test"}.issubset(set(preds["split"]))
    assert preds["proba"].between(0, 1).all()


@pytest.mark.parametrize("horizon", HORIZONS)
def test_every_horizon_trains_and_reports(real_features, horizon):
    result = run_training(
        real_features,
        model_name="logistic",
        n_folds=5,
        embargo_days=EMBARGO,
        horizon=horizon,
    )
    assert result["horizon"] == horizon
    assert result["purge_gap"] == horizon + EMBARGO
    assert np.isfinite(result["summary"]["roc_auc_mean"])
    assert 0.0 <= result["summary"]["roc_auc_mean"] <= 1.0
    # The base rate travels with the metrics so accuracy stays readable.
    assert "pos_rate_mean" in result["summary"]
    assert result["test_metrics"], "test period must be scored"


@pytest.mark.parametrize("horizon", HORIZONS)
def test_predictions_carry_exactly_one_horizon(real_features, horizon):
    """Horizon is explicit in the prediction schema so a 21-day signal can
    never be mistaken for a 1-day one."""
    result = run_training(
        real_features,
        model_name="logistic",
        n_folds=5,
        embargo_days=EMBARGO,
        horizon=horizon,
    )
    preds = result["predictions"]
    assert list(preds.columns) == [
        "date",
        "ticker",
        "horizon",
        "objective",
        "proba",
        "split",
    ]
    assert preds["horizon"].unique().tolist() == [horizon]
    assert preds["objective"].unique().tolist() == ["classification"]


def test_run_training_rejects_unknown_horizon(real_features):
    with pytest.raises(KeyError, match="horizon 7"):
        run_training(
            real_features, model_name="logistic", n_folds=3, embargo_days=7, horizon=7
        )


def test_baseline_comparison_present(real_features):
    result = run_training(
        real_features, model_name="logistic", n_folds=2, embargo_days=7, horizon=1
    )
    baseline = result["baseline_summary"]
    assert baseline is not None
    assert baseline["roc_auc_mean"] == pytest.approx(0.5)


def test_lightgbm_registered(real_features):
    assert "lightgbm" in MODEL_REGISTRY
    from app.models.base import LightGBMModel

    assert isinstance(create_model("lightgbm", [], []), LightGBMModel)


def test_regression_objective_is_a_clean_seam():
    """Regression is deliberately not implemented yet, but the seam exists and
    fails with an explicit message rather than a confusing TypeError."""
    with pytest.raises(NotImplementedError, match="regression objective"):
        create_model("logistic", ["f"], ["ticker"], objective="regression")
    with pytest.raises(ValueError, match="unknown objective"):
        create_model("logistic", ["f"], ["ticker"], objective="nonsense")


def test_feature_columns_exclude_labels_and_metadata():
    df = pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=5),
            "ticker": "A",
            "f1": range(5),
            "fwd_ret_1d": [0.1] * 5,
            "label_up_1d": [1.0] * 5,
            "fwd_ret_252d": [0.5] * 5,
            "label_up_252d": [1.0] * 5,
        }
    )
    num_cols, cat_cols = feature_columns(df)
    assert num_cols == ["f1"]
    assert cat_cols == ["ticker"]


# --------------------------------------------------------------------------
# per-horizon champions
# --------------------------------------------------------------------------


def test_champions_are_isolated_per_horizon(tmp_path):
    from app.storage import all_champions, get_champion, set_champion

    db = tmp_path / "meta.sqlite"
    assert set_champion(
        db, run_id="r1", model_name="logistic", metric_value=0.55, horizon=1
    )
    assert set_champion(
        db, run_id="r21", model_name="hgb", metric_value=0.52, horizon=21
    )

    # A strong 21-day model must not displace the 1-day champion.
    assert get_champion(db, horizon=1)["run_id"] == "r1"
    assert get_champion(db, horizon=21)["run_id"] == "r21"

    # Promotion still requires beating the SAME horizon's incumbent.
    assert not set_champion(
        db, run_id="r1b", model_name="hgb", metric_value=0.50, horizon=1
    )
    assert get_champion(db, horizon=1)["run_id"] == "r1"
    assert set_champion(
        db, run_id="r1c", model_name="hgb", metric_value=0.60, horizon=1
    )
    assert get_champion(db, horizon=1)["run_id"] == "r1c"

    assert {c["horizon"] for c in all_champions(db)} == {1, 21}


def test_champion_absent_for_untrained_horizon(tmp_path):
    from app.storage import get_champion, set_champion

    db = tmp_path / "meta.sqlite"
    set_champion(db, run_id="r1", model_name="logistic", metric_value=0.55, horizon=1)
    assert get_champion(db, horizon=252) is None
