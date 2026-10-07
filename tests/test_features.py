from __future__ import annotations

import numpy as np
import pandas as pd

from app.features.build import build_features

INDICATOR_COLS = [
    "ret_1d",
    "ret_5d",
    "ret_21d",
    "vol_21d",
    "rsi_14",
    "macd",
    "sma_ratio_20_50",
    "bb_zscore_20",
    "atr_14",
]


def test_indicators_shifted_no_lookahead(make_panel):
    panel = make_panel(n=120)
    feats = build_features(panel)

    cutoff = feats["date"].max() - pd_offset(10)
    past = feats[feats["date"] < cutoff]

    perturbed = panel.copy()
    future_mask = perturbed["date"] >= cutoff
    perturbed.loc[future_mask, "close"] = perturbed.loc[future_mask, "close"] * 1.5
    perturbed.loc[future_mask, "volume"] = perturbed.loc[future_mask, "volume"] + 10**6
    feats2 = build_features(perturbed)
    past2 = feats2[feats2["date"] < cutoff]

    assert len(past) == len(past2)
    np.testing.assert_allclose(
        past[INDICATOR_COLS].to_numpy(dtype=float),
        past2[INDICATOR_COLS].to_numpy(dtype=float),
        rtol=1e-12,
        atol=1e-12,
    )


def pd_offset(days):
    import pandas as pd

    return pd.Timedelta(days=days)


def test_row_t_contains_only_info_through_day_t(make_panel):
    """Row T's features may use data up to and including close of day T."""
    panel = make_panel(tickers=("AAPL",), n=60)
    feats = build_features(panel).sort_values("date").reset_index(drop=True)

    close = panel.sort_values("date")["close"].to_numpy()
    got = feats["ret_1d"].to_numpy()
    assert np.allclose(got[1:], close[1:] / close[:-1] - 1)
    assert pd.isna(got[0])


def test_label_present_but_separable(make_panel):
    panel = make_panel(n=80)
    feats = build_features(panel)
    assert "label_up_1d" in feats.columns
    labeled = feats.dropna(subset=["label_up_1d"])
    assert set(labeled["label_up_1d"].unique()).issubset({0.0, 1.0})


def test_multi_ticker_pooled_rows_match_prices(make_panel):
    panel = make_panel(tickers=("AAPL", "MSFT", "GOOGL"), n=90)
    feats = build_features(panel)
    assert set(feats["ticker"].unique()) == {"AAPL", "MSFT", "GOOGL"}
    assert len(feats) == len(panel)


def test_sentiment_joined_same_day_bucketed_availability(make_panel):
    import pandas as pd

    panel = make_panel(tickers=("AAPL",), n=40)
    dates = sorted(pd.to_datetime(panel["date"]).unique())
    sent = pd.DataFrame(
        {
            "ticker": ["AAPL"] * 10,
            "trade_date": pd.to_datetime(dates[10:20]),
            "sent_mean": [0.5] * 10,
            "sent_pos_share": [0.8] * 10,
            "sent_neg_share": [0.1] * 10,
            "sent_dispersion": [0.1] * 10,
            "n_articles": [3] * 10,
        }
    )
    feats = build_features(panel, sentiment=sent)
    col = "sent_mean"
    assert col in feats.columns
    at_first_bucket = feats[feats["date"] == pd.Timestamp(dates[10])]
    assert at_first_bucket[col].iloc[0] == 0.5
    before = feats[feats["date"] < pd.Timestamp(dates[10])]
    assert before[col].isna().all()
