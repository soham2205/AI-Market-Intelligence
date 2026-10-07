"""Daily ticker-level aggregation of article-level FinBERT sentiment.

Covers the arithmetic, the session assignment, and -- most importantly -- that
no article can be attributed to a session that had already opened when it was
published.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from app.nlp.bucketing import ET
from app.nlp.daily_sentiment import (
    DAILY_COLUMNS,
    LABELS,
    aggregate_daily_sentiment,
    assign_trade_date,
    session_calendar,
)

ARTIFACT = Path("data/curated/news_sentiment_daily.parquet")
ARTICLES = Path("data/curated/news_sentiment.parquet")


@pytest.fixture
def sessions() -> pd.DatetimeIndex:
    """A small real-shaped calendar: business days with a holiday removed."""
    days = pd.bdate_range("2024-01-02", "2024-01-31")
    # 2024-01-15 was Martin Luther King Jr. Day -- not a session.
    return pd.DatetimeIndex([d for d in days if d != pd.Timestamp("2024-01-15")])


def art(ticker, published, label, score, aid=None):
    return {
        "article_id": aid or f"{ticker}-{published}",
        "ticker": ticker,
        "published_at": pd.Timestamp(published),
        "label": label,
        "score": score,
        "positive": 0.0,
        "negative": 0.0,
        "neutral": 0.0,
    }


# --------------------------------------------------------------------------
# Aggregation correctness
# --------------------------------------------------------------------------


def test_single_article_day(sessions):
    df = pd.DataFrame([art("AAPL", "2024-01-03T12:00:00Z", "positive", 0.8)])
    out, _ = aggregate_daily_sentiment(df, sessions)
    assert len(out) == 1
    row = out.iloc[0]
    assert row["mean_sentiment"] == pytest.approx(0.8)
    assert row["positive_ratio"] == 1.0
    assert row["negative_ratio"] == 0.0
    assert row["neutral_ratio"] == 0.0
    assert row["article_count"] == 1


def test_multiple_articles_same_ticker_day_aggregate(sessions):
    """Four articles, one session: mean and ratios computed over all four."""
    rows = [
        art("AAPL", "2024-01-03T12:00:00Z", "positive", 1.0, "a1"),
        art("AAPL", "2024-01-03T12:30:00Z", "positive", 0.5, "a2"),
        art("AAPL", "2024-01-03T13:00:00Z", "negative", -0.5, "a3"),
        art("AAPL", "2024-01-03T13:30:00Z", "neutral", 0.0, "a4"),
    ]
    out, _ = aggregate_daily_sentiment(pd.DataFrame(rows), sessions)
    assert len(out) == 1
    row = out.iloc[0]
    assert row["article_count"] == 4
    assert row["mean_sentiment"] == pytest.approx((1.0 + 0.5 - 0.5 + 0.0) / 4)
    assert row["positive_ratio"] == pytest.approx(0.5)
    assert row["negative_ratio"] == pytest.approx(0.25)
    assert row["neutral_ratio"] == pytest.approx(0.25)


def test_tickers_and_days_are_separate_groups(sessions):
    rows = [
        art("AAPL", "2024-01-03T12:00:00Z", "positive", 1.0, "a1"),
        art("AAPL", "2024-01-04T12:00:00Z", "negative", -1.0, "a2"),
        art("MSFT", "2024-01-03T12:00:00Z", "neutral", 0.0, "a3"),
    ]
    out, _ = aggregate_daily_sentiment(pd.DataFrame(rows), sessions)
    assert len(out) == 3
    assert set(out["ticker"]) == {"AAPL", "MSFT"}
    aapl = out[out["ticker"] == "AAPL"].sort_values("trade_date")
    assert aapl["mean_sentiment"].tolist() == pytest.approx([1.0, -1.0])


def test_article_counts_are_correct(sessions):
    rows = [art("AAPL", "2024-01-03T12:00:00Z", "positive", 0.1, f"a{i}") for i in range(7)]
    rows += [art("MSFT", "2024-01-03T12:00:00Z", "neutral", 0.0, f"m{i}") for i in range(3)]
    out, stats = aggregate_daily_sentiment(pd.DataFrame(rows), sessions)
    counts = dict(zip(out["ticker"], out["article_count"], strict=True))
    assert counts == {"AAPL": 7, "MSFT": 3}
    assert stats["articles_represented"] == 10


# --------------------------------------------------------------------------
# Ratios
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "labels",
    [
        ["positive"],
        ["positive", "negative"],
        ["positive", "negative", "neutral"],
        ["neutral"] * 5,
        ["negative", "negative", "positive"],
    ],
)
def test_ratios_sum_to_one(sessions, labels):
    rows = [
        art("AAPL", "2024-01-03T12:00:00Z", lbl, 0.0, f"a{i}")
        for i, lbl in enumerate(labels)
    ]
    out, _ = aggregate_daily_sentiment(pd.DataFrame(rows), sessions)
    total = (
        out["positive_ratio"] + out["negative_ratio"] + out["neutral_ratio"]
    )
    assert (total - 1.0).abs().max() < 1e-12


def test_ratios_are_bounded(sessions):
    rows = [
        art("AAPL", "2024-01-03T12:00:00Z", lbl, 0.0, f"a{i}")
        for i, lbl in enumerate(["positive", "negative", "neutral", "positive"])
    ]
    out, _ = aggregate_daily_sentiment(pd.DataFrame(rows), sessions)
    for col in ("positive_ratio", "negative_ratio", "neutral_ratio"):
        assert out[col].between(0, 1).all()


def test_unknown_label_is_rejected(sessions):
    rows = [art("AAPL", "2024-01-03T12:00:00Z", "bullish", 0.5)]
    with pytest.raises(ValueError, match="unexpected FinBERT labels"):
        aggregate_daily_sentiment(pd.DataFrame(rows), sessions)


# --------------------------------------------------------------------------
# Session assignment
# --------------------------------------------------------------------------


def test_premarket_news_lands_on_the_same_session(sessions):
    """08:00 ET is before the 09:30 open -> actionable that day."""
    df = pd.DataFrame([art("AAPL", "2024-01-03T13:00:00Z", "positive", 0.5)])
    out, _ = aggregate_daily_sentiment(df, sessions)
    assert out.iloc[0]["trade_date"] == pd.Timestamp("2024-01-03")


def test_intraday_news_moves_to_the_next_session(sessions):
    """15:00 UTC = 10:00 ET, the session is already open."""
    df = pd.DataFrame([art("AAPL", "2024-01-03T15:00:00Z", "positive", 0.5)])
    out, _ = aggregate_daily_sentiment(df, sessions)
    assert out.iloc[0]["trade_date"] == pd.Timestamp("2024-01-04")


def test_after_close_news_moves_to_the_next_session(sessions):
    """22:00 UTC = 17:00 ET, after the close."""
    df = pd.DataFrame([art("AAPL", "2024-01-03T22:00:00Z", "positive", 0.5)])
    out, _ = aggregate_daily_sentiment(df, sessions)
    assert out.iloc[0]["trade_date"] == pd.Timestamp("2024-01-04")


def test_weekend_news_lands_on_monday(sessions):
    """Saturday 2024-01-06 -> Monday 2024-01-08."""
    df = pd.DataFrame([art("AAPL", "2024-01-06T14:00:00Z", "positive", 0.5)])
    out, _ = aggregate_daily_sentiment(df, sessions)
    assert out.iloc[0]["trade_date"] == pd.Timestamp("2024-01-08")


def test_holiday_is_skipped(sessions):
    """Friday 2024-01-12 after close -> MLK Day is not a session -> Tuesday."""
    df = pd.DataFrame([art("AAPL", "2024-01-12T22:00:00Z", "positive", 0.5)])
    out, _ = aggregate_daily_sentiment(df, sessions)
    assigned = out.iloc[0]["trade_date"]
    assert assigned == pd.Timestamp("2024-01-16")
    assert assigned in set(sessions)


def test_every_assigned_date_is_a_real_session(sessions):
    stamps = pd.date_range("2024-01-02", "2024-01-25", freq="5h", tz="UTC")
    rows = [art("AAPL", s, "neutral", 0.0, f"a{i}") for i, s in enumerate(stamps)]
    out, _ = aggregate_daily_sentiment(pd.DataFrame(rows), sessions)
    assert set(out["trade_date"]) <= set(sessions)


# --------------------------------------------------------------------------
# Leakage
# --------------------------------------------------------------------------


def test_session_never_precedes_publication(sessions):
    """The core invariant: an article cannot inform a session that already
    happened."""
    stamps = pd.date_range("2024-01-02", "2024-01-25", freq="3h", tz="UTC")
    rows = [art("AAPL", s, "neutral", 0.0, f"a{i}") for i, s in enumerate(stamps)]
    assigned, _ = assign_trade_date(pd.DataFrame(rows), sessions)
    local_day = (
        assigned["published_at"].dt.tz_convert(ET).dt.normalize().dt.tz_localize(None)
    )
    assert (assigned["trade_date"] >= local_day).all()


def test_news_after_the_open_never_joins_that_session(sessions):
    """Anything at or after 09:30 ET must roll forward, never stay put."""
    for hour_utc in (15, 18, 21, 23):  # 10:00, 13:00, 16:00, 18:00 ET
        df = pd.DataFrame(
            [art("AAPL", f"2024-01-03T{hour_utc:02d}:00:00Z", "neutral", 0.0)]
        )
        out, _ = aggregate_daily_sentiment(df, sessions)
        assert out.iloc[0]["trade_date"] > pd.Timestamp("2024-01-03")


def test_articles_before_the_calendar_are_dropped_not_collapsed(sessions):
    """Without this guard every pre-calendar article snaps onto the opening
    session and invents a sentiment spike there."""
    rows = [
        art("AAPL", "2019-05-01T12:00:00Z", "positive", 1.0, "old1"),
        art("AAPL", "2020-05-01T12:00:00Z", "positive", 1.0, "old2"),
        art("AAPL", "2024-01-03T12:00:00Z", "negative", -1.0, "new1"),
    ]
    out, stats = aggregate_daily_sentiment(pd.DataFrame(rows), sessions)
    assert stats["dropped_before_calendar"] == 2
    assert len(out) == 1
    assert out.iloc[0]["trade_date"] == pd.Timestamp("2024-01-03")
    assert out.iloc[0]["article_count"] == 1
    assert out.iloc[0]["mean_sentiment"] == pytest.approx(-1.0)


def test_articles_after_the_calendar_are_dropped(sessions):
    rows = [
        art("AAPL", "2024-01-03T12:00:00Z", "positive", 1.0, "in"),
        art("AAPL", "2030-01-03T12:00:00Z", "positive", 1.0, "future"),
    ]
    out, stats = aggregate_daily_sentiment(pd.DataFrame(rows), sessions)
    assert stats["dropped_after_calendar"] == 1
    assert len(out) == 1


# --------------------------------------------------------------------------
# Schema, determinism, errors
# --------------------------------------------------------------------------


def test_output_schema_and_dtypes(sessions):
    rows = [art("AAPL", "2024-01-03T12:00:00Z", "positive", 0.5)]
    out, _ = aggregate_daily_sentiment(pd.DataFrame(rows), sessions)
    assert list(out.columns) == DAILY_COLUMNS
    for col in ("mean_sentiment", "positive_ratio", "negative_ratio", "neutral_ratio"):
        assert out[col].dtype.kind == "f"
    assert out["article_count"].dtype.kind in "iu"
    assert len(DAILY_COLUMNS) == 7, "five features plus the two keys"


def test_output_is_sorted_and_unique(sessions):
    stamps = pd.date_range("2024-01-02", "2024-01-20", freq="7h", tz="UTC")
    rows = []
    for i, s in enumerate(stamps):
        rows.append(art("MSFT", s, "neutral", 0.0, f"m{i}"))
        rows.append(art("AAPL", s, "positive", 0.2, f"a{i}"))
    out, _ = aggregate_daily_sentiment(pd.DataFrame(rows), sessions)
    assert out.equals(out.sort_values(["ticker", "trade_date"]).reset_index(drop=True))
    assert not out.duplicated(subset=["ticker", "trade_date"]).any()


def test_deterministic(sessions):
    stamps = pd.date_range("2024-01-02", "2024-01-20", freq="6h", tz="UTC")
    rows = [
        art("AAPL", s, LABELS[i % 3], (i % 5) / 10, f"a{i}")
        for i, s in enumerate(stamps)
    ]
    df = pd.DataFrame(rows)
    a, sa = aggregate_daily_sentiment(df, sessions)
    b, sb = aggregate_daily_sentiment(df, sessions)
    pd.testing.assert_frame_equal(a, b)
    assert sa == sb


def test_input_is_not_mutated(sessions):
    df = pd.DataFrame([art("AAPL", "2024-01-03T12:00:00Z", "positive", 0.5)])
    before = df.copy()
    aggregate_daily_sentiment(df, sessions)
    pd.testing.assert_frame_equal(df, before)


def test_missing_columns_raise(sessions):
    df = pd.DataFrame([art("AAPL", "2024-01-03T12:00:00Z", "positive", 0.5)])
    with pytest.raises(KeyError, match="missing columns"):
        aggregate_daily_sentiment(df.drop(columns=["score"]), sessions)


def test_empty_calendar_raises():
    df = pd.DataFrame([art("AAPL", "2024-01-03T12:00:00Z", "positive", 0.5)])
    with pytest.raises(ValueError, match="calendar is empty"):
        aggregate_daily_sentiment(df, pd.DatetimeIndex([]))


# --------------------------------------------------------------------------
# The produced artifact
# --------------------------------------------------------------------------


@pytest.mark.skipif(not ARTIFACT.exists(), reason="daily artifact not built")
def test_artifact_schema_and_invariants():
    daily = pd.read_parquet(ARTIFACT)
    assert list(daily.columns) == DAILY_COLUMNS
    total = daily["positive_ratio"] + daily["negative_ratio"] + daily["neutral_ratio"]
    assert (total - 1.0).abs().max() < 1e-9
    assert (daily["article_count"] > 0).all()
    assert daily["mean_sentiment"].between(-1, 1).all()
    assert not daily.duplicated(subset=["ticker", "trade_date"]).any()
    assert daily[DAILY_COLUMNS[2:]].notna().all().all()


@pytest.mark.skipif(
    not (ARTIFACT.exists() and ARTICLES.exists()), reason="artifacts not built"
)
def test_artifact_sessions_are_real_trading_days():
    daily = pd.read_parquet(ARTIFACT)
    sessions = set(session_calendar())
    assert set(daily["trade_date"]) <= sessions


@pytest.mark.skipif(
    not (ARTIFACT.exists() and ARTICLES.exists()), reason="artifacts not built"
)
def test_artifact_conserves_article_counts():
    """Every represented article is accounted for exactly once."""
    daily = pd.read_parquet(ARTIFACT)
    articles = pd.read_parquet(ARTICLES)
    sessions = session_calendar()
    assigned, stats = assign_trade_date(articles, sessions)
    assert int(daily["article_count"].sum()) == len(assigned)
    assert int(daily["article_count"].sum()) == stats["articles_assigned"]
