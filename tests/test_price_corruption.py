"""Local-context corruption detection in validate_prices.

The FNSPID archive contains bars that pass every structural rule -- high>=low,
positive price above the floor, positive volume -- while sitting ~1/1000 of the
surrounding price. They produce six-figure percentage returns. These tests pin
the detector, and just as importantly pin what it must NOT reject.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from app.data.validate import (
    MAX_LOCAL_DEVIATION,
    implausible_vs_local_context,
    validate_prices,
)

PANEL = Path("data/curated_sp500")
LEGACY = Path("data/curated")


def series(values, start="2015-01-02") -> pd.Series:
    idx = pd.bdate_range(start, periods=len(values))
    return pd.Series(values, index=idx, dtype=float)


def frame(close, volume=None, flat=False) -> pd.DataFrame:
    close = pd.Series(close, dtype=float)
    idx = pd.bdate_range("2015-01-02", periods=len(close))
    close.index = idx
    if flat:
        o = h = low = close
    else:
        o, h, low = close * 0.999, close * 1.004, close * 0.996
    df = pd.DataFrame(
        {
            "open": o,
            "high": h,
            "low": low,
            "close": close,
            "volume": volume if volume is not None else 1e6,
        },
        index=idx,
    )
    df.index.name = "date"
    return df


# --------------------------------------------------------------------------
# Detection
# --------------------------------------------------------------------------


def test_detects_a_thousandfold_collapse():
    """The TT 2016-03-02 shape: one bar at ~1/1000 of its neighbours."""
    close = [38.0] * 40
    close[20] = 0.0343
    flagged = implausible_vs_local_context(series(close))
    assert flagged.iloc[20]
    assert flagged.sum() == 1, "only the corrupt bar may be flagged"


def test_detects_an_upward_spike():
    close = [38.0] * 40
    close[20] = 38_000.0
    flagged = implausible_vs_local_context(series(close))
    assert flagged.iloc[20]
    assert flagged.sum() == 1


def test_validator_drops_and_counts_the_corrupt_bar():
    close = [38.0] * 40
    close[20] = 0.0343
    df = frame(close)
    clean, report = validate_prices(df, ticker="TT")
    assert report.n_implausible_rows_dropped == 1
    assert len(clean) == 39
    assert any("implausibly far from local" in i for i in report.issues)
    assert report.n_rows_out == 39, "row accounting must include the new drop reason"


def test_known_corrupt_tickers_are_caught_in_the_real_archive():
    """Against the FNSPID archive itself, not a synthetic fixture."""
    from app.data.sources.fnspid_price_source import FnspidPriceSource

    src = FnspidPriceSource()
    if not src.available():
        pytest.skip("FNSPID price archive not cached")
    for ticker, minimum in (("TT", 50), ("VST", 10), ("VRT", 1)):
        raw = src.fetch(ticker, "2008-01-01", "2024-01-09")
        if raw.empty:
            pytest.skip(f"{ticker} not in archive")
        _, report = validate_prices(raw, ticker=ticker)
        assert report.n_implausible_rows_dropped >= minimum, (
            f"{ticker}: expected >={minimum} implausible bars, "
            f"got {report.n_implausible_rows_dropped}"
        )


# --------------------------------------------------------------------------
# What must NOT be rejected
# --------------------------------------------------------------------------


def test_flat_ohlc_alone_is_not_corruption():
    """A thinly traded session legitimately prints open=high=low=close.
    Flatness must never be the trigger."""
    df = frame([38.0 + 0.01 * i for i in range(40)], flat=True)
    clean, report = validate_prices(df, ticker="THIN")
    assert report.n_implausible_rows_dropped == 0
    assert len(clean) == 40


def test_flat_ohlc_near_local_context_survives():
    close = [38.0] * 40
    close[20] = 38.05  # flat but normal level
    df = frame(close, flat=True)
    _, report = validate_prices(df, ticker="THIN")
    assert report.n_implausible_rows_dropped == 0


def test_normal_volatility_is_untouched():
    rng = np.random.default_rng(0)
    close = 100 * (1 + np.cumsum(rng.normal(0, 0.015, 300)))
    _, report = validate_prices(frame(close), ticker="NORM")
    assert report.n_implausible_rows_dropped == 0


def test_a_legitimate_one_day_double_is_not_flagged():
    """HIG 2008-12-05 really did roughly double on 8x volume. A 2x move is
    nowhere near the threshold."""
    close = [5.8] * 20 + [11.8] * 20
    flagged = implausible_vs_local_context(series(close))
    assert not flagged.any()


def test_a_sustained_crash_is_not_flagged():
    """A genuine step down that persists: the centred median straddles the
    step, so the ratio lands near 1/2 and stays well inside the threshold."""
    close = [100.0] * 30 + [20.0] * 30
    flagged = implausible_vs_local_context(series(close))
    assert not flagged.any()


def test_a_sustained_tenfold_decline_over_time_is_not_flagged():
    """Gradual decline, no single-session excursion."""
    close = list(np.linspace(100, 8, 200))
    flagged = implausible_vs_local_context(series(close))
    assert not flagged.any()


def test_threshold_boundary_behaviour():
    """Just inside the threshold survives; well outside is caught."""
    safe = [10.0] * 40
    safe[20] = 10.0 * (MAX_LOCAL_DEVIATION * 0.8)
    assert not implausible_vs_local_context(series(safe)).any()

    caught = [10.0] * 40
    caught[20] = 10.0 * (MAX_LOCAL_DEVIATION * 1.5)
    assert implausible_vs_local_context(series(caught)).iloc[20]


def test_short_series_has_no_context_to_judge():
    """Too few rows for a reference: flag nothing rather than guess."""
    flagged = implausible_vs_local_context(series([10.0, 0.01, 10.0]))
    assert not flagged.any()


def test_threshold_is_configurable_and_deterministic():
    close = [38.0] * 40
    close[20] = 38.0 / 6
    assert not implausible_vs_local_context(series(close), max_deviation=10).any()
    assert implausible_vs_local_context(series(close), max_deviation=5).iloc[20]
    a = implausible_vs_local_context(series(close), max_deviation=5)
    b = implausible_vs_local_context(series(close), max_deviation=5)
    pd.testing.assert_series_equal(a, b)


def test_unrelated_validation_rules_unchanged():
    """The new check must not loosen anything else."""
    df = frame([38.0] * 30)
    df.iloc[10, df.columns.get_loc("volume")] = 0
    _, report = validate_prices(df, ticker="X")
    assert report.n_invalid_rows_dropped == 1, "zero-volume rule still applies"

    df2 = frame([38.0] * 30)
    df2.iloc[10, df2.columns.get_loc("high")] = 1.0  # high below open/close
    _, report2 = validate_prices(df2, ticker="X")
    assert report2.n_invalid_rows_dropped >= 1, "structural rule still applies"


# --------------------------------------------------------------------------
# Legacy pipeline and seam handling untouched
# --------------------------------------------------------------------------


@pytest.mark.skipif(not (LEGACY / "prices").exists(), reason="legacy panel absent")
def test_legacy_nine_ticker_panel_is_unaffected():
    """The new rule must drop nothing from the original nine tickers."""
    from app.data.curate import load_curated

    old = load_curated(LEGACY)
    assert old["ticker"].nunique() == 9
    total = 0
    for ticker, g in old.groupby("ticker"):
        px = g.set_index("date")[["open", "high", "low", "close", "volume"]]
        _, report = validate_prices(px, ticker=ticker)
        total += report.n_implausible_rows_dropped
    assert total == 0


def test_seam_handling_still_works():
    """The corruption check is independent of seam masking."""
    from app.data.seams import FNSPID_SEAM_DATES, SOURCE_COLUMN
    from app.features.build import build_features

    seam = FNSPID_SEAM_DATES[0]
    before = pd.bdate_range(end=seam - pd.Timedelta(days=1), periods=120)
    after = pd.bdate_range(start=seam, periods=120)
    idx = before.append(after)
    close = 100 + np.linspace(0, 5, len(idx))
    close[: len(before)] *= 4.0
    rng = np.random.default_rng(0)
    panel = pd.DataFrame(
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
    feats = build_features(panel, horizons=(5,)).set_index("date")
    assert pd.isna(feats["ret_1d"].loc[seam]), "seam masking must still apply"


# --------------------------------------------------------------------------
# The rebuilt panel
# --------------------------------------------------------------------------


@pytest.mark.skipif(not (PANEL / "prices").exists(), reason="expanded panel absent")
def test_recovered_tickers_are_clean_and_present():
    from app.data.curate import load_curated

    panel = load_curated(PANEL)
    for ticker in ("TT", "VRT", "VST"):
        g = panel[panel["ticker"] == ticker]
        assert len(g) > 1000, f"{ticker} missing from the panel"
        assert set(g["source"]) == {"yfinance"}, f"{ticker} provenance not updated"
        ret = g.sort_values("date")["close"].pct_change().abs()
        assert ret.max() < 1.0, f"{ticker} still has an impossible return"
