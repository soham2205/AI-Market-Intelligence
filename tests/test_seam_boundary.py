"""Seam-boundary invalidity handling.

The 2020-07-06 FNSPID seam is an artificial transition: the prices on either
side are fine, the step between them is not. These tests pin that the row and
its prices survive untouched, that any statistic whose window crosses the
boundary is invalidated, and that everything recovers once the window has
moved past it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.data.seams import (
    FNSPID_SEAM_DATES,
    NO_SEAMS,
    backward_window_invalid,
    forward_window_invalid,
    sessions_since_seam,
)
from app.features.build import build_features
from app.features.technical import (
    FEATURE_LOOKBACK,
    compute_indicators,
    mask_seam_contaminated,
)

SEAM = FNSPID_SEAM_DATES[0]
PRICE_COLS = ["open", "high", "low", "close", "volume"]


@pytest.fixture
def panel() -> pd.DataFrame:
    """One ticker spanning the seam, with a hard 4x level step at it.

    Enough history on both sides to let the longest lookback (82) recover.
    """
    before = pd.bdate_range(end=SEAM - pd.Timedelta(days=1), periods=200)
    after = pd.bdate_range(start=SEAM, periods=200)
    idx = before.append(after)

    base = 100 + np.linspace(0, 20, len(idx))
    close = base.copy()
    close[: len(before)] *= 4.0  # the artefact
    # Volume needs genuine variation or its rolling std is zero and the
    # z-score is undefined for reasons unrelated to the seam.
    rng = np.random.default_rng(0)
    vol = 1_000_000.0 * (1 + rng.normal(0, 0.05, len(idx)))
    vol[: len(before)] /= 4.0  # volume is on the other basis too

    return pd.DataFrame(
        {
            "date": idx,
            "ticker": "SEAMY",
            "open": close * 0.995,
            "high": close * 1.01,
            "low": close * 0.99,
            "close": close,
            "volume": vol,
        }
    )


@pytest.fixture
def ohlcv(panel) -> pd.DataFrame:
    return panel.set_index("date")[["open", "high", "low", "close", "volume"]]


# --------------------------------------------------------------------------
# Seam arithmetic
# --------------------------------------------------------------------------


def test_sessions_since_seam_counts_from_the_seam_row(ohlcv):
    since = sessions_since_seam(ohlcv.index, FNSPID_SEAM_DATES)
    pos = ohlcv.index.searchsorted(SEAM)
    assert since[pos] == 0
    assert since[pos + 1] == 1
    assert since[pos + 25] == 25
    assert since[pos - 1] > 10**6, "rows before the seam are not contaminated"


def test_no_seams_means_no_invalidation(ohlcv):
    since = sessions_since_seam(ohlcv.index, NO_SEAMS)
    assert (since > 10**6).all()
    assert not backward_window_invalid(ohlcv.index, 50, NO_SEAMS).any()
    assert not forward_window_invalid(ohlcv.index, 5, NO_SEAMS).any()


def test_backward_window_invalid_spans_lookback_minus_one(ohlcv):
    pos = ohlcv.index.searchsorted(SEAM)
    bad = backward_window_invalid(ohlcv.index, 20, FNSPID_SEAM_DATES)
    assert bad[pos : pos + 19].all()
    assert not bad[pos + 19 :].any()
    assert not bad[:pos].any()


def test_single_row_features_are_never_invalidated(ohlcv):
    assert not backward_window_invalid(ohlcv.index, 1, FNSPID_SEAM_DATES).any()


def test_forward_window_invalidates_rows_before_the_seam(ohlcv):
    pos = ohlcv.index.searchsorted(SEAM)
    bad = forward_window_invalid(ohlcv.index, 5, FNSPID_SEAM_DATES)
    assert bad[pos - 5 : pos].all()
    assert not bad[pos:].any()
    assert not bad[: pos - 5].any()


def test_seam_outside_series_is_a_noop():
    idx = pd.DatetimeIndex(pd.bdate_range("2021-01-04", periods=30))
    assert not backward_window_invalid(idx, 50, FNSPID_SEAM_DATES).any()
    idx2 = pd.DatetimeIndex(pd.bdate_range("2019-01-02", periods=30))
    assert not backward_window_invalid(idx2, 50, FNSPID_SEAM_DATES).any()


# --------------------------------------------------------------------------
# The row and its prices survive
# --------------------------------------------------------------------------


def test_seam_row_is_still_present(panel):
    feats = build_features(panel, horizons=(1,), seam_dates=FNSPID_SEAM_DATES)
    assert SEAM in set(feats["date"])
    assert len(feats) == len(panel), "no row may be dropped"


def test_prices_are_not_altered(panel):
    """Boundary handling must not touch a single price or volume."""
    before = panel.copy()
    build_features(panel, horizons=(1,), seam_dates=FNSPID_SEAM_DATES)
    pd.testing.assert_frame_equal(panel, before)


def test_indicator_masking_does_not_change_prices(ohlcv):
    before = ohlcv.copy()
    compute_indicators(ohlcv, seam_dates=FNSPID_SEAM_DATES)
    pd.testing.assert_frame_equal(ohlcv, before)


# --------------------------------------------------------------------------
# The cross-seam return is invalidated
# --------------------------------------------------------------------------


def test_cross_seam_return_is_invalidated(ohlcv):
    raw = compute_indicators(ohlcv)
    assert raw["ret_1d"].loc[SEAM] < -0.70, "fixture must contain the artefact"

    masked = compute_indicators(ohlcv, seam_dates=FNSPID_SEAM_DATES)
    assert pd.isna(masked["ret_1d"].loc[SEAM])


def test_no_extreme_return_survives_masking(ohlcv):
    masked = compute_indicators(ohlcv, seam_dates=FNSPID_SEAM_DATES)
    assert masked["ret_1d"].abs().max() < 0.05
    assert masked["ret_5d"].abs().max() < 0.10


def test_forward_label_spanning_seam_is_invalidated(panel):
    feats = build_features(
        panel, horizons=(1, 5), seam_dates=FNSPID_SEAM_DATES
    ).set_index("date")
    idx = feats.index
    pos = idx.searchsorted(SEAM)

    assert pd.isna(feats["fwd_ret_1d"].iloc[pos - 1])
    assert pd.isna(feats["label_up_1d"].iloc[pos - 1])
    assert feats["fwd_ret_5d"].iloc[pos - 5 : pos].isna().all()
    # Two sessions clear of the window, labels are usable again.
    assert feats["fwd_ret_5d"].iloc[pos - 7]  is not None
    assert not pd.isna(feats["fwd_ret_5d"].iloc[pos - 7])


# --------------------------------------------------------------------------
# Rolling features do not consume the jump, and recover afterwards
# --------------------------------------------------------------------------


@pytest.mark.parametrize("column,lookback", sorted(FEATURE_LOOKBACK.items()))
def test_every_feature_masked_for_exactly_its_lookback(ohlcv, column, lookback):
    feats = compute_indicators(ohlcv)
    feats["day_of_week"] = ohlcv.index.dayofweek
    masked = mask_seam_contaminated(feats, FNSPID_SEAM_DATES)

    pos = ohlcv.index.searchsorted(SEAM)
    if lookback < 2:
        pd.testing.assert_series_equal(masked[column], feats[column])
        return

    window = masked[column].iloc[pos : pos + lookback - 1]
    assert window.isna().all(), f"{column} still consumes the seam"

    # The first row clear of the seam is valid again.
    recovered = masked[column].iloc[pos + lookback - 1]
    assert not pd.isna(recovered), f"{column} stays invalid too long"


def test_features_before_the_seam_are_untouched(ohlcv):
    raw = compute_indicators(ohlcv)
    masked = compute_indicators(ohlcv, seam_dates=FNSPID_SEAM_DATES)
    pos = ohlcv.index.searchsorted(SEAM)
    pd.testing.assert_frame_equal(raw.iloc[:pos], masked.iloc[:pos])


def test_long_lookback_features_stay_invalid_longer_than_short_ones(ohlcv):
    masked = compute_indicators(ohlcv, seam_dates=FNSPID_SEAM_DATES)
    pos = ohlcv.index.searchsorted(SEAM)
    invalid = {c: int(masked[c].iloc[pos : pos + 120].isna().sum()) for c in masked.columns}
    assert invalid["ret_1d"] < invalid["ret_5d"] < invalid["ret_21d"]
    assert invalid["sma_20"] < invalid["sma_50"]
    assert invalid["vol_21d"] < invalid["vol_63d"]


def test_volume_feature_is_invalidated(ohlcv):
    """Volume is also on a different basis pre-seam, so volume-derived
    indicators are contaminated too -- not only price ones."""
    masked = compute_indicators(ohlcv, seam_dates=FNSPID_SEAM_DATES)
    pos = ohlcv.index.searchsorted(SEAM)
    assert masked["volume_zscore_21"].iloc[pos : pos + 20].isna().all()
    assert masked["obv_delta_5d"].iloc[pos : pos + 5].isna().all()


def test_day_of_week_is_never_invalidated(panel):
    feats = build_features(panel, horizons=(1,), seam_dates=FNSPID_SEAM_DATES)
    assert feats["day_of_week"].notna().all()


# --------------------------------------------------------------------------
# Unaffected data is untouched
# --------------------------------------------------------------------------


def test_default_is_no_masking(ohlcv):
    """Series with no known seam -- a live yfinance pull -- are unaffected."""
    a = compute_indicators(ohlcv)
    b = compute_indicators(ohlcv, seam_dates=NO_SEAMS)
    pd.testing.assert_frame_equal(a, b)


def test_build_features_default_unchanged(panel):
    a = build_features(panel, horizons=(1, 5))
    b = build_features(panel, horizons=(1, 5), seam_dates=NO_SEAMS)
    pd.testing.assert_frame_equal(a, b)


def test_ticker_without_the_seam_in_range_is_unchanged():
    """A ticker whose history starts after the seam has nothing to mask."""
    idx = pd.bdate_range("2021-01-04", periods=150)
    panel = pd.DataFrame(
        {
            "date": idx,
            "ticker": "LATE",
            "open": 100.0,
            "high": 101.0,
            "low": 99.0,
            "close": np.linspace(100, 120, len(idx)),
            "volume": 1e6,
        }
    )
    a = build_features(panel, horizons=(1,))
    b = build_features(panel, horizons=(1,), seam_dates=FNSPID_SEAM_DATES)
    pd.testing.assert_frame_equal(a, b)


def test_multi_ticker_masking_is_per_ticker(panel):
    late = pd.DataFrame(
        {
            "date": pd.bdate_range("2021-01-04", periods=150),
            "ticker": "LATE",
            "open": 100.0,
            "high": 101.0,
            "low": 99.0,
            "close": np.linspace(100, 120, 150),
            "volume": 1e6,
        }
    )
    both = pd.concat([panel, late], ignore_index=True)
    feats = build_features(both, horizons=(1,), seam_dates=FNSPID_SEAM_DATES)

    seamy = feats[feats["ticker"] == "SEAMY"].set_index("date")
    other = feats[feats["ticker"] == "LATE"].set_index("date")
    assert pd.isna(seamy["ret_1d"].loc[SEAM])
    assert other["ret_1d"].iloc[1:].notna().all(), "LATE must be untouched"


def test_masking_is_deterministic(ohlcv):
    a = compute_indicators(ohlcv, seam_dates=FNSPID_SEAM_DATES)
    b = compute_indicators(ohlcv, seam_dates=FNSPID_SEAM_DATES)
    pd.testing.assert_frame_equal(a, b)
