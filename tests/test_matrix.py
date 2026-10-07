"""Combined training matrix: joins, no-news handling, scaling, labels, leakage.

Synthetic fixtures keep these fast; tests against the real artifact skip when
it has not been built.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from app.data.seams import FNSPID_SEAM_DATES, SOURCE_COLUMN
from app.features.matrix import (
    METADATA_COLUMNS,
    NEWS_FEATURES,
    build_matrix,
    model_feature_columns,
)
from app.features.technical import FEATURE_LOOKBACK, SCALE_DEPENDENT
from app.labeling.targets import fwd_ret_col, label_col

HORIZON = 5
LABEL = label_col(HORIZON)
FWD = fwd_ret_col(HORIZON)
MATRIX = Path("data/curated/matrix_h5_v1.parquet")


def make_panel(tickers=("AAA", "BBB"), n=300, start="2015-01-02", price=100.0, source="yfinance"):
    frames = []
    for i, t in enumerate(tickers):
        rng = np.random.default_rng(i)
        dates = pd.bdate_range(start, periods=n)
        close = price * (1 + np.cumsum(rng.normal(0.0005, 0.01, n)))
        df = pd.DataFrame(
            {
                "date": dates,
                "ticker": t,
                "open": close * (1 + rng.normal(0, 0.001, n)),
                "high": close * (1 + abs(rng.normal(0, 0.004, n))),
                "low": close * (1 - abs(rng.normal(0, 0.004, n))),
                "close": close,
                "volume": 1e6 * (1 + rng.normal(0, 0.1, n)),
            }
        )
        df[SOURCE_COLUMN] = source
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def make_news(panel, every=3):
    """News on every Nth session for the first ticker only."""
    t = sorted(panel.ticker.unique())[0]
    d = panel[panel.ticker == t].sort_values("date")["date"].iloc[::every]
    rng = np.random.default_rng(0)
    pos = rng.uniform(0, 0.6, len(d))
    neg = rng.uniform(0, 0.3, len(d))
    return pd.DataFrame(
        {
            "ticker": t,
            "trade_date": d.values,
            "mean_sentiment": pos - neg,
            "positive_ratio": pos,
            "negative_ratio": neg,
            "neutral_ratio": 1 - pos - neg,
            "article_count": rng.integers(1, 20, len(d)),
        }
    )


@pytest.fixture
def panel():
    return make_panel()


@pytest.fixture
def news(panel):
    return make_news(panel)


# --------------------------------------------------------------------------
# Ticker identity and feature selection
# --------------------------------------------------------------------------


def test_ticker_is_not_a_predictive_feature(panel, news):
    m, stats = build_matrix(panel, news, horizon=HORIZON)
    assert "ticker" not in model_feature_columns(m)
    assert stats["ticker_is_a_feature"] is False


def test_ticker_and_date_retained_as_metadata(panel, news):
    m, _ = build_matrix(panel, news, horizon=HORIZON)
    for col in METADATA_COLUMNS:
        assert col in m.columns
    assert m["ticker"].notna().all()
    assert m["date"].notna().all()


def test_no_scale_dependent_feature_reaches_the_model(panel, news):
    m, _ = build_matrix(panel, news, horizon=HORIZON)
    feats = model_feature_columns(m)
    leaked = set(feats) & set(SCALE_DEPENDENT)
    assert not leaked, f"scale-dependent features in matrix: {sorted(leaked)}"


def test_targets_are_not_features(panel, news):
    m, _ = build_matrix(panel, news, horizon=HORIZON)
    feats = model_feature_columns(m)
    assert LABEL not in feats
    assert FWD not in feats
    assert not any(c.startswith(("fwd_ret_", "label_up_")) for c in feats)


def test_feature_order_is_stable(panel, news):
    m, _ = build_matrix(panel, news, horizon=HORIZON)
    assert model_feature_columns(m) == model_feature_columns(m)


def test_every_scale_free_indicator_has_a_lookback(panel, news):
    """Seam masking needs a lookback for every feature it must invalidate."""
    m, _ = build_matrix(panel, news, horizon=HORIZON)
    technical = [c for c in model_feature_columns(m) if c not in NEWS_FEATURES]
    assert all(c in FEATURE_LOOKBACK for c in technical)


# --------------------------------------------------------------------------
# Scaling
# --------------------------------------------------------------------------


def test_scale_free_features_are_comparable_across_price_levels():
    """A $5 stock and a $1,500 stock must yield similar feature magnitudes."""
    cheap = make_panel(tickers=("CHEAP",), price=5.0)
    rich = make_panel(tickers=("RICH",), price=1500.0)
    both = pd.concat([cheap, rich], ignore_index=True)
    m, _ = build_matrix(both, make_news(both, every=1000), horizon=HORIZON)

    feats = [c for c in model_feature_columns(m) if c not in NEWS_FEATURES]
    feats = [c for c in feats if c != "day_of_week"]
    med = m.groupby("ticker")[feats].apply(lambda g: g.abs().median())
    for c in feats:
        lo, hi = med.loc["CHEAP", c], med.loc["RICH", c]
        if min(abs(lo), abs(hi)) < 1e-9:
            continue
        ratio = max(abs(lo), abs(hi)) / min(abs(lo), abs(hi))
        assert ratio < 20, f"{c} differs {ratio:.0f}x between price levels"


def test_raw_levels_would_have_failed_that_test():
    """Guard the guard: the removed features really were scale-dependent."""
    from app.features.build import build_features

    cheap = make_panel(tickers=("CHEAP",), price=5.0)
    rich = make_panel(tickers=("RICH",), price=1500.0)
    f = build_features(pd.concat([cheap, rich], ignore_index=True), horizons=(HORIZON,))
    med = f.groupby("ticker")[["sma_20", "sma_50"]].apply(lambda g: g.abs().median())
    assert med.loc["RICH", "sma_20"] / med.loc["CHEAP", "sma_20"] > 100


# --------------------------------------------------------------------------
# News join and no-news handling
# --------------------------------------------------------------------------


def test_news_joins_on_ticker_and_date(panel, news):
    m, stats = build_matrix(panel, news, horizon=HORIZON)
    joined = m[m.has_news]
    src = news.set_index(["ticker", "trade_date"])
    for r in joined.head(25).itertuples():
        want = src.loc[(r.ticker, r.date)]
        assert r.mean_sentiment == pytest.approx(want.mean_sentiment)
        assert r.article_count == want.article_count
    assert stats["news_rows_joined"] <= stats["news_rows_available"]


def test_news_does_not_leak_to_the_other_ticker(panel, news):
    """News exists only for ticker AAA; BBB must be all zeros."""
    m, _ = build_matrix(panel, news, horizon=HORIZON)
    other = m[m.ticker == "BBB"]
    assert len(other) > 0
    assert (other[NEWS_FEATURES] == 0).all().all()
    assert not other["has_news"].any()


def test_no_news_sessions_are_retained_and_zero_filled(panel, news):
    m, stats = build_matrix(panel, news, horizon=HORIZON)
    quiet = m[~m.has_news]
    assert len(quiet) > 0, "fixture should contain quiet sessions"
    for col in NEWS_FEATURES:
        assert (quiet[col] == 0).all(), f"{col} not zero-filled"
    assert stats["sessions_without_news"] > 0


def test_news_features_never_nan(panel, news):
    m, _ = build_matrix(panel, news, horizon=HORIZON)
    assert m[NEWS_FEATURES].notna().all().all()


def test_has_news_flag_matches_article_count(panel, news):
    m, _ = build_matrix(panel, news, horizon=HORIZON)
    assert (m["has_news"] == (m["article_count"] > 0)).all()


def test_join_cannot_duplicate_rows(panel, news):
    """validate='one_to_one' must reject a duplicated news key."""
    dup = pd.concat([news, news.head(1)], ignore_index=True)
    with pytest.raises((pd.errors.MergeError, AssertionError, ValueError)):
        build_matrix(panel, dup, horizon=HORIZON)


def test_missing_news_column_raises(panel, news):
    with pytest.raises(KeyError, match="missing"):
        build_matrix(panel, news.drop(columns=["article_count"]), horizon=HORIZON)


# --------------------------------------------------------------------------
# Label alignment and leakage
# --------------------------------------------------------------------------


def test_label_matches_five_day_forward_return(panel, news):
    m, _ = build_matrix(panel, news, horizon=HORIZON)
    close = panel[panel.ticker == "AAA"].set_index("date")["close"]
    sub = m[m.ticker == "AAA"].set_index("date")
    checked = 0
    for d in sub.index[:40]:
        pos = close.index.get_loc(d)
        if pos + HORIZON >= len(close):
            continue
        expected = close.iloc[pos + HORIZON] / close.iloc[pos] - 1
        assert sub.loc[d, FWD] == pytest.approx(expected, rel=1e-9)
        assert sub.loc[d, LABEL] == float(expected > 0)
        checked += 1
    assert checked > 20


def test_label_uses_the_requested_horizon_only(panel, news):
    m, _ = build_matrix(panel, news, horizon=HORIZON)
    assert LABEL in m.columns and FWD in m.columns
    assert "label_up_1d" not in m.columns
    assert "label_up_21d" not in m.columns


def test_features_depend_only_on_past_prices(news):
    """Perturb prices strictly after a cutoff: earlier feature rows must not
    change. The definitive no-lookahead check."""
    base = make_panel(tickers=("AAA",), n=300)
    cut = base["date"].iloc[220]
    pert = base.copy()
    mask = pert["date"] > cut
    pert.loc[mask, ["open", "high", "low", "close"]] *= 1.5
    pert.loc[mask, "volume"] *= 3

    n = make_news(base, every=3)
    a, _ = build_matrix(base, n, horizon=HORIZON)
    b, _ = build_matrix(pert, n, horizon=HORIZON)

    feats = [c for c in model_feature_columns(a) if c not in NEWS_FEATURES]
    # Exclude rows whose label window reaches past the cutoff.
    keep = a["date"] < cut - pd.Timedelta(days=12)
    ka, kb = a[keep], b[b["date"].isin(a.loc[keep, "date"])]
    assert len(ka) > 50
    np.testing.assert_allclose(
        ka[feats].to_numpy(dtype=float), kb[feats].to_numpy(dtype=float), rtol=1e-9
    )


def test_seam_rows_are_excluded_for_fnspid_tickers():
    """Seam-invalidated features become NaN and must be dropped, counted."""
    seam = FNSPID_SEAM_DATES[0]
    before = pd.bdate_range(end=seam - pd.Timedelta(days=1), periods=200)
    after = pd.bdate_range(start=seam, periods=200)
    idx = before.append(after)
    close = 100 + np.linspace(0, 10, len(idx))
    rng = np.random.default_rng(0)
    p = pd.DataFrame(
        {
            "date": idx,
            "ticker": "SEAMY",
            "open": close * 0.999,
            "high": close * 1.004,
            "low": close * 0.996,
            "close": close,
            "volume": 1e6 * (1 + rng.normal(0, 0.05, len(idx))),
            SOURCE_COLUMN: "fnspid",
        }
    )
    m, stats = build_matrix(p, make_news(p, every=1000), horizon=HORIZON)
    assert stats["excluded_feature_nan"] > 0
    # Nothing in the masked window survives.
    assert seam not in set(m["date"])


# --------------------------------------------------------------------------
# Exclusions are counted, not silent
# --------------------------------------------------------------------------


def test_row_accounting_reconciles(panel, news):
    m, stats = build_matrix(panel, news, horizon=HORIZON)
    assert (
        stats["price_sessions_in"]
        - stats["excluded_no_label"]
        - stats["excluded_feature_nan"]
        == stats["rows_out"]
        == len(m)
    )


def test_horizon_boundary_rows_excluded(panel, news):
    """The last `horizon` sessions per ticker have no label."""
    m, stats = build_matrix(panel, news, horizon=HORIZON)
    assert stats["excluded_no_label"] >= HORIZON * panel.ticker.nunique()
    for t in panel.ticker.unique():
        last_px = panel[panel.ticker == t]["date"].max()
        assert m[m.ticker == t]["date"].max() < last_px


def test_output_has_no_nan_in_features_or_label(panel, news):
    m, _ = build_matrix(panel, news, horizon=HORIZON)
    assert m[model_feature_columns(m)].notna().all().all()
    assert m[LABEL].notna().all()


def test_mismatched_horizon_raises(panel, news):
    with pytest.raises(ValueError, match="must appear in horizons"):
        build_matrix(panel, news, horizon=5, horizons=(1, 21))


# --------------------------------------------------------------------------
# Determinism
# --------------------------------------------------------------------------


def test_deterministic_output(panel, news):
    a, sa = build_matrix(panel, news, horizon=HORIZON)
    b, sb = build_matrix(panel, news, horizon=HORIZON)
    pd.testing.assert_frame_equal(a, b)
    assert sa == sb


def test_input_frames_not_mutated(panel, news):
    pb, nb = panel.copy(), news.copy()
    build_matrix(panel, news, horizon=HORIZON)
    pd.testing.assert_frame_equal(panel, pb)
    pd.testing.assert_frame_equal(news, nb)


# --------------------------------------------------------------------------
# The real artifact
# --------------------------------------------------------------------------


@pytest.mark.skipif(not MATRIX.exists(), reason="matrix artifact not built")
def test_real_matrix_shape_and_schema():
    m = pd.read_parquet(MATRIX)
    assert len(m) > 1_000_000
    assert m["ticker"].nunique() == 447
    for col in METADATA_COLUMNS:
        assert col in m.columns
    assert LABEL in m.columns
    feats = model_feature_columns(m)
    assert "ticker" not in feats
    assert not set(feats) & set(SCALE_DEPENDENT)
    assert m[feats].notna().all().all()
    assert m[LABEL].isin([0.0, 1.0]).all()


@pytest.mark.skipif(not MATRIX.exists(), reason="matrix artifact not built")
def test_real_matrix_no_news_rows_are_zero():
    m = pd.read_parquet(MATRIX)
    quiet = m[~m.has_news]
    assert len(quiet) > 0
    assert (quiet[NEWS_FEATURES] == 0).all().all()


@pytest.mark.skipif(not MATRIX.exists(), reason="matrix artifact not built")
def test_real_matrix_is_sorted_and_unique():
    m = pd.read_parquet(MATRIX)
    assert not m.duplicated(subset=["ticker", "date"]).any()
    assert m.equals(m.sort_values(["ticker", "date"]).reset_index(drop=True))
