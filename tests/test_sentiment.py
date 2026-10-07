from __future__ import annotations

import pandas as pd
import pytest

from app.nlp.aggregate import add_ewma, aggregate_daily
from app.nlp.bucketing import first_tradable_session


def ts_utc(s: str) -> pd.Timestamp:
    return pd.Timestamp(s)


def test_premarket_maps_to_same_day():
    assert first_tradable_session(ts_utc("2024-01-03 13:00:00+00:00")) == pd.Timestamp("2024-01-03")


def test_intraday_maps_to_next_day():
    assert first_tradable_session(ts_utc("2024-01-03 15:00:00+00:00")) == pd.Timestamp("2024-01-04")


def test_after_close_maps_to_next_day():
    assert first_tradable_session(ts_utc("2024-01-05 22:00:00+00:00")) == pd.Timestamp("2024-01-08")


def test_saturday_morning_maps_to_monday():
    assert first_tradable_session(ts_utc("2024-01-06 13:00:00+00:00")) == pd.Timestamp("2024-01-08")


def test_naive_timestamp_treated_as_utc():
    assert first_tradable_session(pd.Timestamp("2024-01-03 13:00:00")) == pd.Timestamp("2024-01-03")


def test_aggregate_daily_stats():
    news = pd.DataFrame(
        {
            "ticker": ["A", "A", "B"],
            "trade_date": [pd.Timestamp("2024-01-02")] * 2 + [pd.Timestamp("2024-01-02")],
            "score": [0.5, -0.5, 0.8],
            "headline": ["x", "y", "z"],
        }
    )
    agg = aggregate_daily(news)
    a = agg[(agg["ticker"] == "A")].iloc[0]
    assert a["n_articles"] == 2
    assert a["sent_mean"] == pytest.approx(0.0)
    assert a["sent_pos_share"] == pytest.approx(0.5)
    assert a["sent_neg_share"] == pytest.approx(0.5)
    b = agg[agg["ticker"] == "B"].iloc[0]
    assert b["sent_mean"] == pytest.approx(0.8)


def test_ewma_carries_forward_through_gaps():
    daily = pd.DataFrame(
        {
            "ticker": ["A"] * 4,
            "trade_date": pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"]),
            "n_articles": [2, 0, 1, 0],
            "sent_mean": [0.6, 0.0, -0.2, 0.0],
        }
    )
    out = add_ewma(daily, span=5)
    vals = out.sort_values("trade_date")["ewma_sent"].tolist()
    assert vals[0] == pytest.approx(0.6)
    assert vals[1] == pytest.approx(0.6)
    assert vals[2] < vals[1]
    assert vals[3] == pytest.approx(vals[2])
