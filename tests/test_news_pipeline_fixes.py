"""Regression tests for the five news/sentiment pipeline defects.

Each test pins one defect so it cannot silently return.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.data.sources.news_source import YFinanceNewsSource, dedupe_news
from app.features.build import SENTIMENT_COLS, build_features
from app.nlp.aggregate import add_ewma, aggregate_daily
from app.nlp.bucketing import first_tradable_session

# --------------------------------------------------------------------------
# Defect 1: yfinance timestamp parser
# --------------------------------------------------------------------------

# Recorded from the live yfinance payload: `title` is populated while
# `headline`, `pubIsoDate`, `displayTimeIso` and `providerPublishTime` are all
# None -- the shape that silently dropped every article.
CURRENT_PAYLOAD = [
    {
        "id": "abc123",
        "content": {
            "id": "abc123",
            "title": "Apple Signals Big September Hardware Reveal",
            "headline": None,
            "pubDate": "2026-08-27T11:08:50Z",
            "displayTime": "2026-08-27T11:08:50Z",
            "pubIsoDate": None,
            "displayTimeIso": None,
            "providerPublishTime": None,
            "canonicalUrl": {"url": "https://example.com/a"},
            "clickThroughUrl": {"url": "https://example.com/a"},
            "provider": {"displayName": "Example Wire"},
        },
    },
    {
        "id": "def456",
        "content": {
            "id": "def456",
            "title": "Chipmaker Beats Quarterly Estimates",
            "pubDate": "2026-08-26T21:30:00Z",
            "displayTime": "2026-08-26T21:30:00Z",
            "canonicalUrl": {"url": "https://example.com/b"},
            "provider": {"displayName": "Example Wire"},
        },
    },
]

# The older payload shape must keep working.
LEGACY_PAYLOAD = [
    {
        "id": "old1",
        "content": {
            "headline": "Legacy Shaped Article",
            "pubIsoDate": "2024-01-02T13:00:00Z",
            "clickThroughUrl": {"url": "https://example.com/legacy"},
            "provider": {"displayName": "Legacy Wire"},
        },
    }
]

EPOCH_PAYLOAD = [
    {
        "id": "ep1",
        "title": "Epoch Only Article",
        "providerPublishTime": 1704200400,
        "content": {},
    }
]


def _fetch_with(monkeypatch, payload):
    """Drive YFinanceNewsSource.fetch against a recorded payload."""
    import yfinance as yf

    class _FakeTicker:
        def __init__(self, symbol):
            self.news = payload

    monkeypatch.setattr(yf, "Ticker", _FakeTicker)
    return YFinanceNewsSource().fetch("AAPL")


def test_current_yfinance_payload_parses_timestamps(monkeypatch):
    """pubDate/displayTime must be honoured -- previously every row was
    dropped because only the *Iso keys were consulted."""
    df = _fetch_with(monkeypatch, CURRENT_PAYLOAD)
    assert len(df) == 2, "current-shape payload produced no rows"
    assert df["published_at"].notna().all()
    assert df["published_at"].iloc[0] == pd.Timestamp("2026-08-27T11:08:50Z")
    assert df["headline"].iloc[0] == "Apple Signals Big September Hardware Reveal"
    assert df["ticker"].unique().tolist() == ["AAPL"]


def test_legacy_payload_still_parses(monkeypatch):
    """The old keys remain as fallbacks."""
    df = _fetch_with(monkeypatch, LEGACY_PAYLOAD)
    assert len(df) == 1
    assert df["published_at"].iloc[0] == pd.Timestamp("2024-01-02T13:00:00Z")


def test_epoch_fallback_still_parses(monkeypatch):
    """providerPublishTime remains the last-resort fallback."""
    df = _fetch_with(monkeypatch, EPOCH_PAYLOAD)
    assert len(df) == 1
    assert df["published_at"].iloc[0].year == 2024


def test_articles_without_any_timestamp_are_skipped(monkeypatch):
    df = _fetch_with(monkeypatch, [{"content": {"title": "No timestamp anywhere"}}])
    assert df.empty
    assert list(df.columns) == [
        "article_id",
        "headline",
        "source",
        "url",
        "published_at",
        "ticker",
    ]


def test_dedupe_still_collapses_repeats(monkeypatch):
    df = _fetch_with(monkeypatch, CURRENT_PAYLOAD + CURRENT_PAYLOAD)
    assert len(df) == 4
    assert len(dedupe_news(df)) == 2


# --------------------------------------------------------------------------
# Shared fixtures for the feature-join defects
# --------------------------------------------------------------------------


@pytest.fixture
def price_panel() -> pd.DataFrame:
    dates = pd.bdate_range("2024-01-02", periods=12)
    return pd.DataFrame(
        {
            "date": dates,
            "ticker": "AAPL",
            "open": 100.0,
            "high": 101.0,
            "low": 99.0,
            "close": np.linspace(100, 111, len(dates)),
            "volume": 1e6,
        }
    )


@pytest.fixture
def sentiment_daily(price_panel) -> pd.DataFrame:
    """A sentiment table built exactly the way ingest-news builds it."""
    dates = price_panel["date"].tolist()
    news = pd.DataFrame(
        {
            "ticker": ["AAPL"] * 4,
            "trade_date": [dates[2], dates[2], dates[5], dates[5]],
            "score": [0.5, -0.2, 0.1, 0.4],
        }
    )
    return add_ewma(aggregate_daily(news))


# --------------------------------------------------------------------------
# Defect 2: trade_date/date KeyError on the CLI path
# --------------------------------------------------------------------------


def test_cli_style_sentiment_join_does_not_raise(price_panel, sentiment_daily):
    """The CLI reads sentiment_daily.parquet and passes it straight through.
    It used to rename trade_date -> date first, which raised KeyError."""
    feats = build_features(price_panel, sentiment=sentiment_daily, horizons=(1,))
    assert len(feats) == len(price_panel)
    assert "sent_mean" in feats.columns


def test_cli_featurize_path_end_to_end(tmp_path, price_panel, sentiment_daily, monkeypatch):
    """Drive the real _cmd_featurize against a curated dir on disk -- the
    exact path that regressed, which no test previously covered."""
    from app.cli import _cmd_featurize
    from app.config import PipelineConfig

    curated = tmp_path / "curated"
    (curated / "prices").mkdir(parents=True)
    price_panel.to_parquet(curated / "prices" / "ticker=AAPL.parquet", index=False)
    sentiment_daily.to_parquet(curated / "sentiment_daily.parquet", index=False)

    cfg = PipelineConfig.model_validate(
        {
            "universe": {"tickers": ["AAPL"], "start": "2024-01-01"},
            "paths": {"curated_dir": str(curated), "artifacts_dir": str(tmp_path / "art")},
            "train": {"horizons": [1]},
        }
    )
    assert _cmd_featurize(cfg) == 0, "featurize returned a failure code"

    feats = pd.read_parquet(curated / "features.parquet")
    assert "sent_mean" in feats.columns
    assert "trade_date" not in feats.columns
    assert len(feats) == len(price_panel)


# --------------------------------------------------------------------------
# Defect 3: ewma_sent silently dropped
# --------------------------------------------------------------------------


def test_ewma_sent_reaches_the_feature_matrix(price_panel, sentiment_daily):
    assert "ewma_sent" in sentiment_daily.columns, "fixture should carry ewma_sent"
    feats = build_features(price_panel, sentiment=sentiment_daily, horizons=(1,))
    assert "ewma_sent" in feats.columns, "ewma_sent was dropped during the join"
    assert feats["ewma_sent"].notna().any(), "ewma_sent joined but is entirely NaN"


def test_ewma_sent_is_a_model_feature(price_panel, sentiment_daily):
    """Reaching the matrix is not enough -- it must survive feature selection
    rather than being mistaken for metadata or a target."""
    from app.models.train import feature_columns

    feats = build_features(price_panel, sentiment=sentiment_daily, horizons=(1,))
    num_cols, _ = feature_columns(feats)
    assert "ewma_sent" in num_cols
    for col in SENTIMENT_COLS:
        assert col in num_cols, f"{col} is not being offered to the model"


def test_missing_ewma_column_does_not_crash(price_panel, sentiment_daily):
    """A sentiment table built without add_ewma must still join."""
    without = sentiment_daily.drop(columns=["ewma_sent"])
    feats = build_features(price_panel, sentiment=without, horizons=(1,))
    assert "sent_mean" in feats.columns
    assert "ewma_sent" not in feats.columns


# --------------------------------------------------------------------------
# Defect 4: n_articles NaN instead of 0 on no-news days
# --------------------------------------------------------------------------


def test_no_news_days_get_zero_articles(price_panel, sentiment_daily):
    feats = build_features(price_panel, sentiment=sentiment_daily, horizons=(1,))
    assert feats["n_articles"].notna().all(), "no-news days left n_articles NaN"

    covered = set(sentiment_daily["trade_date"])
    for _, row in feats.iterrows():
        if row["date"] in covered:
            assert row["n_articles"] > 0
        else:
            assert row["n_articles"] == 0, f"{row['date'].date()} should be 0"


def test_sentiment_measures_stay_nan_on_no_news_days(price_panel, sentiment_daily):
    """Only counts are zero-filled. A missing *measurement* must stay NaN so
    the per-fold imputer handles it -- zero would be a fabricated reading."""
    feats = build_features(price_panel, sentiment=sentiment_daily, horizons=(1,))
    covered = set(sentiment_daily["trade_date"])
    uncovered = feats[~feats["date"].isin(covered)]
    assert len(uncovered) > 0
    assert uncovered["sent_mean"].isna().all()
    assert uncovered["sent_dispersion"].isna().all()


def test_n_articles_is_never_median_imputed(price_panel, sentiment_daily):
    """Guards against a regression that re-introduces imputation for counts."""
    feats = build_features(price_panel, sentiment=sentiment_daily, horizons=(1,))
    values = set(feats["n_articles"].unique())
    assert 0.0 in values
    assert not any(0 < v < 1 for v in values), "fractional counts imply imputation"


# --------------------------------------------------------------------------
# Defect 5: holiday-aware bucketing
# --------------------------------------------------------------------------


@pytest.fixture
def real_sessions() -> pd.DatetimeIndex:
    """Real NYSE sessions taken from the curated price panel."""
    px = pd.read_parquet("data/curated/prices/ticker=AAPL.parquet")
    return pd.DatetimeIndex(pd.to_datetime(px["date"]).dt.normalize().unique()).sort_values()


def test_july_third_after_close_skips_july_fourth(real_sessions):
    """BDay(1) lands on July 4th, which is not a session."""
    got = first_tradable_session(pd.Timestamp("2024-07-03T21:00:00Z"), real_sessions)
    assert got == pd.Timestamp("2024-07-05")
    assert got in set(real_sessions)
    naive = first_tradable_session(pd.Timestamp("2024-07-03T21:00:00Z"))
    assert naive == pd.Timestamp("2024-07-04")
    assert naive not in set(real_sessions), "fixture assumption broken"


def test_december_twentyfourth_after_close_skips_christmas(real_sessions):
    got = first_tradable_session(pd.Timestamp("2024-12-24T22:00:00Z"), real_sessions)
    assert got == pd.Timestamp("2024-12-26")
    assert got in set(real_sessions)
    assert first_tradable_session(pd.Timestamp("2024-12-24T22:00:00Z")) == pd.Timestamp(
        "2024-12-25"
    )


def test_calendar_aware_preserves_existing_rules(real_sessions):
    """Every rule that already held must still hold with a real calendar."""
    cases = [
        ("2024-03-05T13:00:00Z", "2024-03-05"),  # 08:00 ET, pre-open -> same day
        ("2024-03-05T16:00:00Z", "2024-03-06"),  # 11:00 ET, intraday -> next
        ("2024-03-05T22:00:00Z", "2024-03-06"),  # 17:00 ET, after close -> next
        ("2024-03-09T13:00:00Z", "2024-03-11"),  # Saturday -> Monday
        ("2024-03-10T23:00:00Z", "2024-03-11"),  # Sunday evening -> Monday
        ("2024-03-08T21:00:00Z", "2024-03-11"),  # Friday after close -> Monday
    ]
    for ts, expected in cases:
        got = first_tradable_session(pd.Timestamp(ts), real_sessions)
        assert got == pd.Timestamp(expected), f"{ts} -> {got}, expected {expected}"
        assert got in set(real_sessions)


def test_every_bucketed_date_is_a_real_session(real_sessions):
    """Sweep a year: no output may fall outside the trading calendar."""
    stamps = pd.date_range("2024-01-01", "2024-12-31", freq="7h", tz="UTC")
    got = pd.DatetimeIndex(
        [first_tradable_session(t, real_sessions) for t in stamps]
    ).dropna()
    assert len(got) > 0
    assert set(got) <= set(real_sessions), "bucketing produced a non-session date"


def test_beyond_calendar_returns_nat(real_sessions):
    """News after the last known session has no session to attribute it to."""
    beyond = real_sessions.max() + pd.Timedelta(days=30)
    assert pd.isna(first_tradable_session(beyond.tz_localize("UTC"), real_sessions))


def test_sessions_none_preserves_legacy_behaviour():
    """Existing callers that pass no calendar keep the old semantics."""
    assert first_tradable_session(pd.Timestamp("2024-03-05T13:00:00Z")) == pd.Timestamp(
        "2024-03-05"
    )
    assert first_tradable_session(pd.Timestamp("2024-03-05T16:00:00Z")) == pd.Timestamp(
        "2024-03-06"
    )
    assert first_tradable_session(pd.Timestamp("2024-03-09T13:00:00Z")) == pd.Timestamp(
        "2024-03-11"
    )


def test_naive_timestamp_still_treated_as_utc(real_sessions):
    naive = first_tradable_session(pd.Timestamp("2024-03-05 13:00:00"), real_sessions)
    aware = first_tradable_session(pd.Timestamp("2024-03-05T13:00:00Z"), real_sessions)
    assert naive == aware
