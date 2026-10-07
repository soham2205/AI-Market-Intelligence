from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.data.validate import validate_prices


def test_clean_data_passes_through(make_ohlcv):
    df = make_ohlcv(n=20)
    out, report = validate_prices(df, "AAPL")
    assert len(out) == 20
    assert report.n_rows_out == 20
    assert report.n_invalid_rows_dropped == 0
    assert report.n_null_rows_dropped == 0
    assert report.n_duplicates_removed == 0
    assert list(out.columns) == ["date", "open", "high", "low", "close", "volume", "ticker"]
    assert (out["ticker"] == "AAPL").all()


def test_missing_values_dropped(make_ohlcv):
    df = make_ohlcv(n=10)
    df.iloc[3, df.columns.get_loc("close")] = np.nan
    out, report = validate_prices(df, "MSFT")
    assert len(out) == 9
    assert report.n_null_rows_dropped == 1


def test_structurally_invalid_rows_dropped(make_ohlcv):
    df = make_ohlcv(n=10)

    df.iloc[2, df.columns.get_loc("high")] = 1.0
    df.iloc[2, df.columns.get_loc("low")] = 5.0

    df.iloc[4, df.columns.get_loc("close")] = -10.0

    df.iloc[6, df.columns.get_loc("volume")] = 0.0

    out, report = validate_prices(df, "NVDA")
    assert len(out) == 7
    assert report.n_invalid_rows_dropped == 3


def test_zero_volume_allowed_when_configured(make_ohlcv):
    df = make_ohlcv(n=5)
    df.iloc[1, df.columns.get_loc("volume")] = 0.0
    out, _ = validate_prices(df, "AAPL", allow_zero_volume=True)
    assert len(out) == 5


def test_duplicates_removed_keep_last(make_ohlcv):
    df = make_ohlcv(n=8)
    duplicated = pd.concat([df, df.tail(2)])
    out, report = validate_prices(duplicated, "TSLA")
    assert len(out) == 8
    assert report.n_duplicates_removed == 2


def test_empty_input_returns_empty_with_warning(make_ohlcv):
    empty = pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
    out, report = validate_prices(empty, "XYZ")
    assert out.empty
    assert any("no rows" in i for i in report.issues)


def test_missing_required_column_raises(make_ohlcv):
    df = make_ohlcv(n=5).drop(columns=["volume"])
    with pytest.raises(ValueError, match="missing required columns"):
        validate_prices(df, "AAPL")


def test_timezone_aware_index_normalized(make_ohlcv):
    df = make_ohlcv(n=5)
    df.index = df.index.tz_localize("UTC")
    out, _ = validate_prices(df, "AAPL")
    assert pd.api.types.is_datetime64_any_dtype(out["date"])
    assert out["date"].dt.tz is None


def test_output_sorted_by_date(make_ohlcv):
    df = make_ohlcv(n=15)
    shuffled = df.sample(frac=1.0, random_state=0)
    out, _ = validate_prices(shuffled, "AAPL")
    assert out["date"].is_monotonic_increasing
