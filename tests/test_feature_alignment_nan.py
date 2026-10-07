from __future__ import annotations

import numpy as np
import pandas as pd

from app.features.build import EXCLUDED_FROM_FEATURES, build_features

INDICATOR_WARMUP = {
    "ret_1d": 1,
    "ret_5d": 5,
    "ret_21d": 21,
    "vol_21d": 21,
    "vol_63d": 63,
    "sma_20": 19,
    "sma_50": 49,
    "sma_ratio_20_50": 49,
    "bb_zscore_20": 19,
    "atr_14": 13,
    "volume_zscore_21": 20,
    "obv_delta_5d": 5,
}


def test_warmup_nan_counts_match_windows(make_panel):
    """Rolling indicators must produce NaN only during their warmup window."""
    panel = make_panel(tickers=("AAPL",), n=200)
    feats = build_features(panel)
    for col, expected_warmup in INDICATOR_WARMUP.items():
        first_valid = feats[col].first_valid_index()
        assert first_valid == expected_warmup, (
            f"{col}: expected first valid row {expected_warmup}, got {first_valid}"
        )


def test_no_infinite_values_in_features(make_panel):
    panel = make_panel(n=300)
    feats = build_features(panel)
    feature_cols = [c for c in feats.columns if c not in EXCLUDED_FROM_FEATURES]
    numeric = feats[feature_cols].select_dtypes(include=[np.number])
    assert not np.isinf(numeric.to_numpy(dtype=float)).any()


def test_label_nan_only_on_last_row_per_ticker(make_panel):
    panel = make_panel(tickers=("AAPL", "MSFT"), n=100)
    feats = build_features(panel)
    nan_labels = feats[feats["label_up_1d"].isna()]
    assert len(nan_labels) == 2
    assert set(nan_labels["ticker"]) == {"AAPL", "MSFT"}
    last_dates = panel.groupby("ticker")["date"].max()
    for _, row in nan_labels.iterrows():
        assert row["date"] == last_dates[row["ticker"]]


def test_feature_target_row_alignment(make_panel):
    """Row T pairs features known at close of day T with the label for the
    move Close_T -> Close_{T+1}."""
    panel = make_panel(tickers=("AAPL",), n=120)
    feats = build_features(panel).sort_values("date").reset_index(drop=True)

    close = panel.sort_values("date").set_index("date")["close"]
    dates = close.index

    for t in range(1, len(dates) - 1):
        row = feats.iloc[t]
        expected_feat_ret = close.iloc[t] / close.iloc[t - 1] - 1
        assert row["ret_1d"] == pytest_approx(expected_feat_ret)
        expected_label = float(close.iloc[t + 1] > close.iloc[t])
        assert row["label_up_1d"] == expected_label


def pytest_approx(value):
    import pytest

    return pytest.approx(value)


def test_append_future_rows_does_not_change_past_features(make_panel):
    """Append-only stability: existing feature rows must be byte-identical
    when new future days arrive. Labels may materialize (NaN -> value) on the
    previously-last row but an existing label value must never flip."""
    panel = make_panel(tickers=("AAPL", "MSFT"), n=150)
    before = build_features(panel)

    extra_dates = pd.bdate_range(
        panel["date"].max() + pd.Timedelta(days=1), periods=20
    )
    rng = np.random.default_rng(999)
    extras = []
    for ticker in ("AAPL", "MSFT"):
        last_close = panel[panel["ticker"] == ticker]["close"].iloc[-1]
        closes = last_close * np.cumprod(1 + rng.normal(0, 0.01, 20))
        extras.append(
            pd.DataFrame(
                {
                    "date": extra_dates,
                    "ticker": ticker,
                    "open": closes * 0.999,
                    "high": closes * 1.005,
                    "low": closes * 0.995,
                    "close": closes,
                    "volume": rng.uniform(1e6, 4e6, 20),
                }
            )
        )
    extended = pd.concat([panel, *extras], ignore_index=True)
    after = build_features(extended)

    merged = after.merge(
        before,
        on=["date", "ticker"],
        suffixes=("_new", "_old"),
    )
    assert len(merged) == len(before)

    indicator_cols = ["ret_1d", "ret_5d", "vol_21d", "sma_50", "rsi_14"]
    for col in indicator_cols:
        old = merged[f"{col}_old"].astype(float)
        new = merged[f"{col}_new"].astype(float)
        same = np.isclose(old, new, equal_nan=True)
        assert same.all(), f"indicator {col} changed for historical rows"

    old_lab = merged["label_up_1d_old"]
    new_lab = merged["label_up_1d_new"]
    materialized_or_same = (old_lab == new_lab) | (old_lab.isna() & new_lab.notna())
    flipped = (~materialized_or_same).sum()
    assert flipped == 0, f"{flipped} historical labels changed value"


def test_multi_ticker_groups_do_not_bleed(make_panel):
    """Shift/roll operations must reset at ticker boundaries: each ticker's
    first session must have NaN indicators and a valid label from day 2."""
    panel = make_panel(tickers=("AAPL", "MSFT"), n=60)
    feats = build_features(panel)
    for ticker in ("AAPL", "MSFT"):
        g = feats[feats["ticker"] == ticker].sort_values("date").reset_index(drop=True)
        assert pd.isna(g.loc[0, ["ret_1d", "sma_20", "vol_21d"]]).all()
        assert pd.notna(g.loc[1, "ret_1d"])
        assert g["label_up_1d"].iloc[:-1].notna().all()


def test_target_matches_explicit_next_day_definition(make_panel):
    """y_t = 1 iff Close_{t+1} > Close_t, computed straight from raw prices."""
    panel = make_panel(tickers=("AAPL",), n=80)
    feats = build_features(panel, positive_threshold=0.0)

    px = panel.sort_values("date").set_index("date")["close"]
    next_close = px.shift(-1)
    expected = (next_close > px).astype(float).where(next_close.notna())

    joined = feats.set_index("date")["label_up_1d"]
    aligned = expected.reindex(joined.index)
    compared = (joined.isna() & aligned.isna()) | (joined == aligned)
    assert compared.all()


def test_perturbing_future_rows_never_touches_past_rows(make_panel):
    """The definitive no-lookahead check: modify prices strictly after a
    cutoff; every feature row before the cutoff must be unchanged."""
    panel = make_panel(tickers=("AAPL", "MSFT"), n=200)
    before = build_features(panel)

    cutoff = panel["date"].max() - pd.Timedelta(days=15)
    perturbed = panel.copy()
    future_mask = perturbed["date"] >= cutoff
    perturbed.loc[future_mask, ["open", "high", "low", "close"]] *= 1.5
    perturbed.loc[future_mask, "volume"] += 10**6
    after = build_features(perturbed)

    b = before[before["date"] < cutoff].sort_values(["ticker", "date"]).reset_index(drop=True)
    a = after[after["date"] < cutoff].sort_values(["ticker", "date"]).reset_index(drop=True)
    assert len(b) == len(a)
    cols = [c for c in b.columns if c != "fwd_ret_1d"]
    np.testing.assert_allclose(
        b[cols].select_dtypes(float).to_numpy(),
        a[cols].select_dtypes(float).to_numpy(),
        rtol=1e-12,
        atol=1e-12,
    )
