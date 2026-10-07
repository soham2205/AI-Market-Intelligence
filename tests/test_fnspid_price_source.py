"""FNSPID price source: schema handling, adjustment, and PriceSource contract.

Tests build a real zip archive on disk with FNSPID-shaped CSVs, so the actual
code path (zip -> CSV -> adjust -> frame) runs without network access.
"""

from __future__ import annotations

import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from app.data.sources.base import REQUIRED_OHLCV
from app.data.sources.fnspid_price_source import (
    ADJ_CLOSE,
    DEFAULT_PRICE_ZIP,
    RAW_COLUMNS,
    FnspidPriceSource,
    apply_adjustment,
    member_name,
    parse_price_csv,
)
from app.data.validate import validate_prices

# FNSPID stores newest-first and carries both close and adj close.
# AAPL rows here mimic a 2:1 split: adj close is half of close before the event.
AAPL_CSV = """date,open,high,low,close,adj close,volume
2023-12-28,194.14,194.66,193.17,193.58,193.58,34014500
2023-12-27,192.49,193.50,191.09,193.15,193.15,48087700
2023-12-26,193.61,193.89,192.83,193.05,193.05,28919300
2020-08-28,504.05,505.77,498.31,499.23,249.615,46907200
2020-08-27,508.00,509.94,495.33,500.04,250.02,39528000
"""

MSFT_CSV = """date,open,high,low,close,adj close,volume
2023-12-28,375.37,376.46,374.16,375.28,375.28,14327000
2023-12-27,374.66,375.10,372.81,374.07,374.07,14905400
"""

# A CSV with a zero close, to prove the factor guard holds.
EDGE_CSV = """date,open,high,low,close,adj close,volume
2022-01-04,10.0,11.0,9.0,10.0,5.0,1000
2022-01-03,0.0,0.0,0.0,0.0,0.0,0
"""


@pytest.fixture
def archive(tmp_path) -> Path:
    """A zip shaped like the real one, including macOS resource-fork junk."""
    path = tmp_path / "full_history.zip"
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("full_history/AAPL.csv", AAPL_CSV)
        zf.writestr("full_history/MSFT.csv", MSFT_CSV)
        zf.writestr("full_history/EDGE.csv", EDGE_CSV)
        # The real archive carries these alongside every member.
        zf.writestr("__MACOSX/._full_history", b"\x00\x05\x16\x07")
        zf.writestr("__MACOSX/full_history/._AAPL.csv", b"\x00\x05\x16\x07")
    return path


@pytest.fixture
def source(archive) -> FnspidPriceSource:
    return FnspidPriceSource(zip_path=archive)


# --------------------------------------------------------------------------
# Schema
# --------------------------------------------------------------------------


def test_raw_schema_constant_matches_the_real_file():
    assert RAW_COLUMNS == [
        "date",
        "open",
        "high",
        "low",
        "close",
        "adj close",
        "volume",
    ]


def test_parse_returns_price_source_shape():
    df = parse_price_csv(AAPL_CSV)
    assert list(df.columns) == REQUIRED_OHLCV
    assert df.index.name == "date"
    assert isinstance(df.index, pd.DatetimeIndex)
    assert df.index.tz is None, "index must be tz-naive like YFinanceSource"
    assert (df.index == df.index.normalize()).all()


def test_rows_are_reordered_ascending():
    """The archive stores newest-first; downstream code assumes ascending."""
    df = parse_price_csv(AAPL_CSV)
    assert df.index.is_monotonic_increasing
    assert df.index[0] == pd.Timestamp("2020-08-27")
    assert df.index[-1] == pd.Timestamp("2023-12-28")


def test_missing_columns_raise():
    bad = "date,open,high,low,close,volume\n2023-01-03,1,1,1,1,100\n"
    with pytest.raises(ValueError, match="missing columns"):
        parse_price_csv(bad)


def test_unparseable_dates_are_dropped():
    csv = AAPL_CSV + "not-a-date,1,1,1,1,1,1\n"
    df = parse_price_csv(csv)
    assert len(df) == 5
    assert df.index.notna().all()


# --------------------------------------------------------------------------
# The adjusted-close decision
# --------------------------------------------------------------------------


def test_close_becomes_adjusted_close():
    """`close` must carry the adjusted value, matching yfinance auto_adjust."""
    df = parse_price_csv(AAPL_CSV)
    assert df.loc[pd.Timestamp("2020-08-27"), "close"] == pytest.approx(250.02)
    assert df.loc[pd.Timestamp("2023-12-28"), "close"] == pytest.approx(193.58)


def test_ohlc_all_scaled_by_the_same_factor():
    """Adjusting close alone would break high >= max(open, close) and inject
    fake intraday ranges. Every price column takes the same factor."""
    df = parse_price_csv(AAPL_CSV)
    row = df.loc[pd.Timestamp("2020-08-27")]
    factor = 250.02 / 500.04  # adj close / close
    assert row["open"] == pytest.approx(508.00 * factor)
    assert row["high"] == pytest.approx(509.94 * factor)
    assert row["low"] == pytest.approx(495.33 * factor)


def test_adjustment_preserves_bar_structure():
    """A uniform positive factor cannot invalidate OHLC ordering."""
    df = parse_price_csv(AAPL_CSV)
    assert (df["high"] >= df[["open", "close"]].max(axis=1) - 1e-9).all()
    assert (df["low"] <= df[["open", "close"]].min(axis=1) + 1e-9).all()
    assert (df["high"] >= df["low"]).all()


def test_unadjusted_rows_are_left_alone():
    """Where adj close == close the factor is 1 and prices are untouched."""
    df = parse_price_csv(AAPL_CSV)
    row = df.loc[pd.Timestamp("2023-12-27")]
    assert row["open"] == pytest.approx(192.49)
    assert row["close"] == pytest.approx(193.15)


def test_volume_is_not_rescaled():
    """yfinance auto_adjust leaves volume alone; so do we."""
    df = parse_price_csv(AAPL_CSV)
    assert df.loc[pd.Timestamp("2020-08-27"), "volume"] == 39528000
    assert df.loc[pd.Timestamp("2023-12-28"), "volume"] == 34014500


def test_zero_close_does_not_produce_inf_or_nan_prices():
    df = parse_price_csv(EDGE_CSV)
    prices = df[["open", "high", "low", "close"]].to_numpy(dtype=float)
    assert not np.isinf(prices).any()
    assert not np.isnan(prices).any()


def test_apply_adjustment_is_pure():
    """The rule is testable without any zip or network."""
    raw = pd.DataFrame(
        {
            "date": pd.to_datetime(["2022-01-03"]),
            "open": [100.0],
            "high": [110.0],
            "low": [90.0],
            "close": [100.0],
            ADJ_CLOSE: [50.0],
            "volume": [1000.0],
        }
    )
    before = raw.copy()
    out = apply_adjustment(raw)
    pd.testing.assert_frame_equal(raw, before, check_dtype=False)
    assert out.loc[0, "open"] == pytest.approx(50.0)
    assert out.loc[0, "high"] == pytest.approx(55.0)
    assert out.loc[0, "low"] == pytest.approx(45.0)
    assert out.loc[0, "close"] == pytest.approx(50.0)
    assert out.loc[0, "volume"] == 1000.0


# --------------------------------------------------------------------------
# PriceSource contract
# --------------------------------------------------------------------------


def test_fetch_matches_yfinance_source_shape(source):
    df = source.fetch("AAPL")
    assert list(df.columns) == REQUIRED_OHLCV
    assert df.index.name == "date"
    assert df.index.tz is None
    assert df.index.is_monotonic_increasing


def test_fetch_is_case_insensitive(source):
    assert len(source.fetch("aapl")) == len(source.fetch("AAPL")) == 5


def test_fetch_date_filtering(source):
    assert len(source.fetch("AAPL", start="2023-01-01")) == 3
    assert len(source.fetch("AAPL", end="2021-01-01")) == 2
    assert len(source.fetch("AAPL", start="2023-12-27", end="2023-12-28")) == 2
    assert len(source.fetch("AAPL", start="2019-01-01", end="2019-12-31")) == 0


def test_absent_ticker_returns_empty_frame_not_error(source):
    """NVDA really is missing from the archive; callers must not crash."""
    df = source.fetch("NVDA")
    assert df.empty
    assert list(df.columns) == REQUIRED_OHLCV
    assert df.index.name == "date"


def test_has_ticker(source):
    assert source.has_ticker("AAPL")
    assert source.has_ticker("msft")
    assert not source.has_ticker("NVDA")


def test_tickers_excludes_macos_resource_forks(source):
    assert source.tickers() == ["AAPL", "EDGE", "MSFT"]


def test_member_name_normalises():
    assert member_name(" aapl ") == "full_history/AAPL.csv"


def test_available_reflects_local_cache(tmp_path, archive):
    assert FnspidPriceSource(zip_path=archive).available()
    assert not FnspidPriceSource(zip_path=tmp_path / "absent.zip").available()


def test_default_cache_path_is_inside_gitignored_data_dir():
    assert DEFAULT_PRICE_ZIP.parts[0] == "data"


# --------------------------------------------------------------------------
# Interoperability with the existing curation layer
# --------------------------------------------------------------------------


def test_output_passes_existing_price_validation(source):
    """The whole point of the protocol: FNSPID output must flow through the
    unmodified validator exactly like yfinance output does."""
    raw = source.fetch("AAPL")
    clean, report = validate_prices(raw, ticker="AAPL")
    assert report.n_rows_in == 5
    assert len(clean) == 5
    assert report.n_invalid_rows_dropped == 0
    assert list(clean.columns) == ["date", *REQUIRED_OHLCV, "ticker"]
    assert set(clean["ticker"]) == {"AAPL"}


def test_validation_screens_the_degenerate_row(source):
    """EDGE has a zero-price, zero-volume bar; the existing validator drops it
    without any change to the validator."""
    raw = source.fetch("EDGE")
    clean, report = validate_prices(raw, ticker="EDGE")
    assert report.n_invalid_rows_dropped == 1
    assert len(clean) == 1


def test_source_satisfies_price_source_protocol(source):
    """Structural conformance: PriceSource is a plain (non-runtime-checkable)
    Protocol, so compare the fetch signature against the existing yfinance
    implementation rather than using isinstance."""
    import inspect

    from app.data.sources.yfinance_source import YFinanceSource

    ours = inspect.signature(type(source).fetch)
    theirs = inspect.signature(YFinanceSource.fetch)
    assert list(ours.parameters) == list(theirs.parameters) == [
        "self",
        "ticker",
        "start",
        "end",
    ]
    assert callable(source.fetch)


def test_two_tickers_are_independent(source):
    aapl, msft = source.fetch("AAPL"), source.fetch("MSFT")
    assert len(aapl) == 5
    assert len(msft) == 2
    assert aapl.loc[pd.Timestamp("2023-12-28"), "close"] != pytest.approx(
        msft.loc[pd.Timestamp("2023-12-28"), "close"]
    )
