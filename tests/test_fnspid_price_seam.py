"""Repair of the FNSPID archive's adjustment seam.

The archive splices two differently-adjusted vintages at SEAM_DATE. For any
ticker that split after that date the pre-seam segment sits on the wrong
basis, producing a synthetic one-day crash. These tests pin the detection
rule, the repair, and -- just as importantly -- that unaffected tickers and
volume are left alone.

Fixtures are synthetic so the rules are exercised without the 0.59 GB archive.
"""

from __future__ import annotations

import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from app.data.sources.base import REQUIRED_OHLCV
from app.data.sources.fnspid_price_source import (
    MIN_SEAM_GAP,
    SEAM_DATE,
    FnspidPriceSource,
    detect_seam,
    parse_price_csv,
    repair_seam,
)

SPLIT = 4.0  # pre-seam prices are 4x too high, as AAPL's are


def _series(n_before: int = 30, n_after: int = 30, factor: float = SPLIT) -> pd.DataFrame:
    """Build a smooth series, then push the pre-seam half up by `factor`.

    Sessions are laid out around SEAM_DATE on real business days.
    """
    before = pd.bdate_range(end=SEAM_DATE - pd.Timedelta(days=1), periods=n_before)
    after = pd.bdate_range(start=SEAM_DATE, periods=n_after)
    idx = before.append(after)

    base = 100 + np.linspace(0, 4, len(idx))
    close = base.copy()
    close[: len(before)] *= factor  # the artefact

    df = pd.DataFrame(
        {
            "open": close * 0.995,
            "high": close * 1.01,
            "low": close * 0.99,
            "close": close,
            "volume": np.full(len(idx), 1_000_000.0),
        },
        index=idx,
    )
    df.index.name = "date"
    return df


@pytest.fixture
def seamed() -> pd.DataFrame:
    return _series()


@pytest.fixture
def clean() -> pd.DataFrame:
    return _series(factor=1.0)


# --------------------------------------------------------------------------
# Detection
# --------------------------------------------------------------------------


def test_detects_the_seam(seamed):
    factor = detect_seam(seamed)
    assert factor is not None
    assert factor == pytest.approx(SPLIT / 0.995, rel=0.02)


def test_clean_series_reports_no_seam(clean):
    assert detect_seam(clean) is None


def test_small_gaps_are_ignored():
    """A 10% move is ordinary volatility, not an adjustment artefact."""
    assert detect_seam(_series(factor=1.10)) is None


def test_threshold_is_respected():
    assert detect_seam(_series(factor=MIN_SEAM_GAP * 1.05)) is not None
    assert detect_seam(_series(factor=MIN_SEAM_GAP * 0.95)) is None


def test_one_day_spike_is_not_treated_as_a_seam():
    """A spike that snaps back is a glitch, not a rescale. The persistence
    check must reject it or we would corrupt real history."""
    df = _series(factor=1.0)
    pos = df.index.searchsorted(SEAM_DATE)
    df.iloc[pos, df.columns.get_loc("open")] /= 5.0
    df.iloc[pos, df.columns.get_loc("close")] /= 5.0
    assert detect_seam(df) is None


def test_no_seam_when_file_ends_before_seam_date():
    """GOOGL's real data stops in 2020-04; there is no seam inside the file."""
    df = _series()
    df = df[df.index < SEAM_DATE - pd.Timedelta(days=30)]
    assert detect_seam(df) is None


def test_no_seam_when_file_starts_after_seam_date():
    df = _series()
    df = df[df.index > SEAM_DATE + pd.Timedelta(days=5)]
    assert detect_seam(df) is None


def test_too_short_series_is_safe():
    assert detect_seam(pd.DataFrame(columns=REQUIRED_OHLCV)) is None


def test_reverse_direction_seam_is_detected():
    """A reverse split leaves the pre-segment too LOW; both directions count."""
    assert detect_seam(_series(factor=1 / 5.0)) is not None


# --------------------------------------------------------------------------
# Repair
# --------------------------------------------------------------------------


def test_repair_removes_the_artificial_return(seamed):
    raw_ret = seamed["close"].pct_change().loc[SEAM_DATE]
    assert raw_ret < -0.70, "fixture should contain the artefact"

    fixed, factor = repair_seam(seamed)
    assert factor is not None
    assert abs(fixed["close"].pct_change().loc[SEAM_DATE]) < 0.05


def test_no_extreme_return_remains_anywhere(seamed):
    fixed, _ = repair_seam(seamed)
    assert fixed["close"].pct_change().abs().max() < 0.05


def test_repair_preserves_ohlc_relationships(seamed):
    fixed, _ = repair_seam(seamed)
    assert (fixed["high"] >= fixed[["open", "close"]].max(axis=1) - 1e-9).all()
    assert (fixed["low"] <= fixed[["open", "close"]].min(axis=1) + 1e-9).all()
    assert (fixed["high"] >= fixed["low"]).all()
    assert (fixed[["open", "high", "low", "close"]] > 0).all().all()


def test_all_four_price_columns_scaled_by_the_same_factor(seamed):
    """Scaling close alone would invent intraday ranges."""
    fixed, factor = repair_seam(seamed)
    pre = seamed.index < SEAM_DATE
    for col in ("open", "high", "low", "close"):
        np.testing.assert_allclose(
            fixed.loc[pre, col].to_numpy(),
            seamed.loc[pre, col].to_numpy() / factor,
            rtol=1e-12,
        )


def test_volume_is_never_rescaled(seamed):
    fixed, _ = repair_seam(seamed)
    pd.testing.assert_series_equal(fixed["volume"], seamed["volume"])


def test_post_seam_rows_are_untouched(seamed):
    fixed, _ = repair_seam(seamed)
    post = seamed.index >= SEAM_DATE
    pd.testing.assert_frame_equal(fixed.loc[post], seamed.loc[post])


def test_pre_seam_history_is_kept_not_discarded(seamed):
    """The repair rescales history; it must never truncate it."""
    fixed, _ = repair_seam(seamed)
    assert len(fixed) == len(seamed)
    assert (fixed.index == seamed.index).all()
    assert (fixed.index < SEAM_DATE).sum() > 0


def test_clean_series_is_returned_unchanged(clean):
    fixed, factor = repair_seam(clean)
    assert factor is None
    pd.testing.assert_frame_equal(fixed, clean)


def test_repair_is_deterministic(seamed):
    a, fa = repair_seam(seamed)
    b, fb = repair_seam(seamed)
    assert fa == fb
    pd.testing.assert_frame_equal(a, b)


def test_repair_does_not_mutate_input(seamed):
    before = seamed.copy()
    repair_seam(seamed)
    pd.testing.assert_frame_equal(seamed, before)


def test_repaired_frame_has_no_remaining_seam(seamed):
    """Re-running finds nothing: the repair is a fixed point."""
    fixed, _ = repair_seam(seamed)
    assert detect_seam(fixed) is None
    again, factor = repair_seam(fixed)
    assert factor is None
    pd.testing.assert_frame_equal(again, fixed)


# --------------------------------------------------------------------------
# Wiring through the source
# --------------------------------------------------------------------------

SEAMED_CSV = """date,open,high,low,close,adj close,volume
2020-07-09,102.0,103.0,101.0,102.5,102.5,1000000
2020-07-08,101.0,102.0,100.0,101.5,101.5,1000000
2020-07-07,100.5,101.5,99.5,101.0,101.0,1000000
2020-07-06,100.0,101.0,99.0,100.5,100.5,1000000
2020-07-02,400.0,404.0,396.0,400.0,400.0,1000000
2020-07-01,398.0,402.0,394.0,399.0,399.0,1000000
2020-06-30,396.0,400.0,392.0,398.0,398.0,1000000
"""

CLEAN_CSV = """date,open,high,low,close,adj close,volume
2020-07-09,102.0,103.0,101.0,102.5,102.5,1000000
2020-07-08,101.0,102.0,100.0,101.5,101.5,1000000
2020-07-07,100.5,101.5,99.5,101.0,101.0,1000000
2020-07-06,100.0,101.0,99.0,100.5,100.5,1000000
2020-07-02,99.0,100.0,98.0,99.5,99.5,1000000
2020-07-01,98.0,99.0,97.0,98.5,98.5,1000000
2020-06-30,97.0,98.0,96.0,97.5,97.5,1000000
"""


@pytest.fixture
def archive(tmp_path) -> Path:
    path = tmp_path / "full_history.zip"
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("full_history/SEAM.csv", SEAMED_CSV)
        zf.writestr("full_history/CLEAN.csv", CLEAN_CSV)
    return path


def test_source_repairs_by_default(archive):
    df = FnspidPriceSource(zip_path=archive).fetch("SEAM")
    assert df["close"].pct_change().abs().max() < 0.10


def test_source_can_read_verbatim(archive):
    df = FnspidPriceSource(zip_path=archive, repair=False).fetch("SEAM")
    assert df["close"].pct_change().min() < -0.70


def test_repair_flag_does_not_change_clean_tickers(archive):
    fixed = FnspidPriceSource(zip_path=archive, repair=True).fetch("CLEAN")
    raw = FnspidPriceSource(zip_path=archive, repair=False).fetch("CLEAN")
    pd.testing.assert_frame_equal(fixed, raw)


def test_repair_applied_before_date_filtering(archive):
    """Filtering to a post-seam window must still yield repaired prices --
    the detection needs rows on both sides, so order matters."""
    src = FnspidPriceSource(zip_path=archive)
    windowed = src.fetch("SEAM", start="2020-07-06")
    full = src.fetch("SEAM")
    pd.testing.assert_frame_equal(windowed, full.loc[full.index >= "2020-07-06"])


def test_parse_price_csv_repair_flag():
    fixed = parse_price_csv(SEAMED_CSV, repair=True)
    raw = parse_price_csv(SEAMED_CSV, repair=False)
    assert fixed["close"].pct_change().abs().max() < 0.10
    assert raw["close"].pct_change().min() < -0.70
    assert len(fixed) == len(raw)


def test_repaired_output_still_passes_validation(archive):
    from app.data.validate import validate_prices

    raw = FnspidPriceSource(zip_path=archive).fetch("SEAM")
    clean, report = validate_prices(raw, ticker="SEAM")
    assert report.n_invalid_rows_dropped == 0
    assert len(clean) == len(raw)


def test_volume_unchanged_through_the_source(archive):
    fixed = FnspidPriceSource(zip_path=archive, repair=True).fetch("SEAM")
    raw = FnspidPriceSource(zip_path=archive, repair=False).fetch("SEAM")
    pd.testing.assert_series_equal(fixed["volume"], raw["volume"])
