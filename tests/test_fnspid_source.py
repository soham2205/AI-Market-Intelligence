"""FNSPID ingestion: ticker attribution, timestamps, and bucketing.

Unit tests run against a synthetic FNSPID-shaped fixture so they never
depend on the 5.73 GB download. Tests that need the real extracted artifact
skip cleanly when it is absent.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from app.data.sources.base import NEWS_SCHEMA
from app.data.sources.fnspid_source import (
    FNSPID_DATE_FORMAT,
    USED_COLUMNS,
    FnspidNewsSource,
    extract_universe,
)
from app.nlp.bucketing import first_tradable_session

UNIVERSE = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "JPM", "JNJ", "XOM", "SPY"]
ARTIFACT = Path("data/curated/news_fnspid.parquet")


def _raw_row(date: str, title: str, symbol: str, url: str = "", pub: str = "Benzinga"):
    return {
        "Date": date,
        "Article_title": title,
        "Stock_symbol": symbol,
        "Url": url or f"https://example.com/{abs(hash(title)) % 10**8}",
        "Publisher": pub,
    }


@pytest.fixture
def raw_csv(tmp_path) -> Path:
    """A synthetic FNSPID CSV with the verified real column layout."""
    rows = [
        _raw_row("2015-03-02 12:30:00 UTC", "Apple unveils new product line", "AAPL"),
        _raw_row("2015-03-02 21:05:00 UTC", "Apple closes higher after rally", "AAPL"),
        _raw_row("2018-07-03 21:00:00 UTC", "Apple gains before the holiday", "AAPL"),
        _raw_row("2020-06-05 06:30:54 UTC", "Microsoft beats estimates", "MSFT"),
        _raw_row("2021-11-11 14:00:00 UTC", "Nvidia surges on earnings", "NVDA"),
        _raw_row("2023-12-15 18:00:00 UTC", "Exxon announces buyback", "XOM"),
        # Rows that must NOT survive the universe filter:
        _raw_row("2020-01-02 12:00:00 UTC", "Agilent reports results", "A"),
        _raw_row("2020-01-02 12:00:00 UTC", "Tesla announcement", "TSLA"),
        # Invalid timestamp -- must be dropped, never fabricated:
        _raw_row("not-a-timestamp", "Broken timestamp article", "AAPL"),
        # Exact duplicate of the first row -- must dedupe:
        _raw_row("2015-03-02 12:30:00 UTC", "Apple unveils new product line", "AAPL"),
    ]
    df = pd.DataFrame(rows)[USED_COLUMNS]
    # Pad with the unused trailing columns the real file carries.
    for c in ["Author", "Article", "Lsa_summary", "Luhn_summary", "Textrank_summary",
              "Lexrank_summary"]:
        df[c] = ""
    path = tmp_path / "fnspid_raw.csv"
    df.to_csv(path, index=False)
    return path


@pytest.fixture
def extracted(raw_csv, tmp_path) -> tuple[Path, dict]:
    out = tmp_path / "news_fnspid.parquet"
    stats = extract_universe(UNIVERSE, raw_path=raw_csv, out_path=out,
                             chunksize=4, progress=False)
    return out, stats


# --------------------------------------------------------------------------
# Schema and extraction
# --------------------------------------------------------------------------


def test_extract_produces_canonical_schema(extracted):
    out, _ = extracted
    df = pd.read_parquet(out)
    assert list(df.columns) == NEWS_SCHEMA


def test_chunked_extraction_matches_expected_rows(extracted):
    """chunksize=4 forces multiple chunks; rows must not be lost at boundaries."""
    out, stats = extracted
    df = pd.read_parquet(out)
    # 6 valid universe rows (the 7th is a duplicate, the 8th has a bad timestamp)
    assert stats["rows_written"] == 6
    assert len(df) == 6
    assert stats["rows_scanned"] == 10
    assert stats["rows_matched"] == 8  # includes the invalid-timestamp AAPL row + dupe


def test_invalid_timestamps_are_dropped_not_fabricated(extracted):
    out, stats = extracted
    assert stats["rows_invalid_timestamp"] == 1
    df = pd.read_parquet(out)
    assert df["published_at"].notna().all()
    assert "Broken timestamp article" not in set(df["headline"])


def test_duplicates_are_removed(extracted):
    _, stats = extracted
    assert stats["duplicates_removed"] == 1


def test_timestamps_are_timezone_aware_utc(extracted):
    out, _ = extracted
    df = pd.read_parquet(out)
    ts = pd.to_datetime(df["published_at"], utc=True)
    assert str(ts.dt.tz) == "UTC"
    # Value preserved exactly as published, not shifted.
    apple = df[df["headline"] == "Apple unveils new product line"]
    assert pd.Timestamp(apple["published_at"].iloc[0]) == pd.Timestamp(
        "2015-03-02 12:30:00", tz="UTC"
    )


def test_date_format_constant_matches_real_data():
    """Guards the strict parse against a silent format drift."""
    assert (
        pd.to_datetime("2020-06-05 06:30:54 UTC", format=FNSPID_DATE_FORMAT, utc=True)
        == pd.Timestamp("2020-06-05 06:30:54", tz="UTC")
    )


# --------------------------------------------------------------------------
# Step 5: ticker attribution comes ONLY from Stock_symbol
# --------------------------------------------------------------------------


def test_only_universe_tickers_survive(extracted):
    out, _ = extracted
    df = pd.read_parquet(out)
    assert set(df["ticker"]) <= set(UNIVERSE)
    assert "TSLA" not in set(df["ticker"])
    assert "A" not in set(df["ticker"]), "substring/partial symbol leaked in"


def test_apple_rows_map_only_to_aapl(extracted):
    out, _ = extracted
    src = FnspidNewsSource(out)
    aapl = src.fetch("AAPL")
    assert len(aapl) == 3
    assert aapl["ticker"].unique().tolist() == ["AAPL"]
    for other in ["MSFT", "SPY", "NVDA", "GOOGL"]:
        assert "Apple unveils new product line" not in set(src.fetch(other)["headline"])


def test_fetch_one_ticker_returns_only_that_ticker(extracted):
    out, _ = extracted
    src = FnspidNewsSource(out)
    for ticker in ["AAPL", "MSFT", "NVDA", "XOM"]:
        got = src.fetch(ticker)
        assert set(got["ticker"]) == {ticker}, f"{ticker} fetch leaked other tickers"


def test_attribution_ignores_title_content(extracted):
    """'Microsoft beats estimates' is attributed to MSFT by Stock_symbol.
    A title mentioning a company must never pull it into that ticker."""
    out, _ = extracted
    src = FnspidNewsSource(out)
    # "Apple closes higher after rally" mentions no other ticker, but the
    # Nvidia headline must not appear under AAPL despite both being tech.
    assert "Nvidia surges on earnings" not in set(src.fetch("AAPL")["headline"])
    assert "Nvidia surges on earnings" in set(src.fetch("NVDA")["headline"])


def test_ticker_lookup_is_case_insensitive(extracted):
    out, _ = extracted
    src = FnspidNewsSource(out)
    assert len(src.fetch("aapl")) == len(src.fetch("AAPL")) == 3


# --------------------------------------------------------------------------
# Date-range filtering
# --------------------------------------------------------------------------


def test_date_range_filtering(extracted):
    out, _ = extracted
    src = FnspidNewsSource(out)
    assert len(src.fetch("AAPL", start="2015-01-01", end="2015-12-31")) == 2
    assert len(src.fetch("AAPL", start="2018-01-01", end="2018-12-31")) == 1
    assert len(src.fetch("AAPL", start="2019-01-01", end="2019-12-31")) == 0
    assert len(src.fetch("AAPL", start="2005-01-01", end="2023-12-31")) == 3


def test_research_window_bounds_are_inclusive(extracted):
    out, _ = extracted
    src = FnspidNewsSource(out)
    exact = src.fetch("AAPL", start="2015-03-02", end="2015-03-03")
    assert len(exact) == 2


def test_missing_artifact_raises_clearly(tmp_path):
    src = FnspidNewsSource(tmp_path / "nope.parquet")
    assert not src.available()
    with pytest.raises(FileNotFoundError, match="extract_universe"):
        src.fetch("AAPL")


# --------------------------------------------------------------------------
# Step 4: FNSPID timestamps through the real-session bucketing
# --------------------------------------------------------------------------


@pytest.fixture
def sessions() -> pd.DatetimeIndex:
    px = pd.read_parquet("data/curated/prices/ticker=AAPL.parquet")
    return pd.DatetimeIndex(
        pd.to_datetime(px["date"]).dt.normalize().unique()
    ).sort_values()


def test_fnspid_timestamps_bucket_to_real_sessions(extracted, sessions):
    """Every FNSPID publication time must land on an actual trading session."""
    out, _ = extracted
    df = pd.read_parquet(out)
    ts = pd.to_datetime(df["published_at"], utc=True)
    bucketed = pd.DatetimeIndex(
        [first_tradable_session(t, sessions) for t in ts]
    ).dropna()
    assert len(bucketed) == len(df)
    assert set(bucketed) <= set(sessions)


def test_fnspid_premarket_maps_to_same_session(sessions):
    """2020-06-05 06:30:54 UTC = 02:30 ET Friday, before the 09:30 open."""
    ts = pd.Timestamp("2020-06-05 06:30:54", tz="UTC")
    assert first_tradable_session(ts, sessions) == pd.Timestamp("2020-06-05")


def test_fnspid_intraday_maps_to_next_session(sessions):
    """2015-03-02 12:30 UTC = 07:30 ET... still pre-open. Use 15:00 UTC =
    10:00 ET, inside the session, which must roll to the next day."""
    assert first_tradable_session(
        pd.Timestamp("2015-03-02 15:00:00", tz="UTC"), sessions
    ) == pd.Timestamp("2015-03-03")


def test_fnspid_after_close_maps_to_next_session(sessions):
    """2015-03-02 21:05 UTC = 16:05 ET, after the close."""
    assert first_tradable_session(
        pd.Timestamp("2015-03-02 21:05:00", tz="UTC"), sessions
    ) == pd.Timestamp("2015-03-03")


def test_fnspid_weekend_maps_to_monday(sessions):
    """Saturday 2021-11-13 -> Monday 2021-11-15."""
    assert first_tradable_session(
        pd.Timestamp("2021-11-13 14:00:00", tz="UTC"), sessions
    ) == pd.Timestamp("2021-11-15")


def test_fnspid_holiday_rolls_past_july_fourth(sessions):
    """2018-07-03 21:00 UTC = 17:00 ET, after close. Next calendar day is
    July 4th, which is not a session."""
    got = first_tradable_session(pd.Timestamp("2018-07-03 21:00:00", tz="UTC"), sessions)
    assert got == pd.Timestamp("2018-07-05")
    assert got in set(sessions)


def test_fnspid_bucketing_never_precedes_publication(extracted, sessions):
    """The bucketed session must never fall before the article's own date --
    that would make future news available in the past."""
    out, _ = extracted
    df = pd.read_parquet(out)
    for ts in pd.to_datetime(df["published_at"], utc=True):
        session = first_tradable_session(ts, sessions)
        if pd.isna(session):
            continue
        assert session.date() >= ts.tz_convert("America/New_York").date()


# --------------------------------------------------------------------------
# Real extracted artifact (skips when absent)
# --------------------------------------------------------------------------


@pytest.mark.skipif(not ARTIFACT.exists(), reason="FNSPID artifact not extracted")
def test_real_artifact_covers_universe_only():
    df = pd.read_parquet(ARTIFACT)
    assert set(df["ticker"]) <= set(UNIVERSE)
    assert list(df.columns) == NEWS_SCHEMA
    assert df["published_at"].notna().all()


@pytest.mark.skipif(not ARTIFACT.exists(), reason="FNSPID artifact not extracted")
def test_real_artifact_has_no_duplicate_article_ids():
    df = pd.read_parquet(ARTIFACT)
    assert df["article_id"].is_unique
