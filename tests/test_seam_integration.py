"""End-to-end: FNSPID seam metadata reaching feature and label construction.

Proves the wiring, not just the arithmetic: provenance recorded at curation
time is read back automatically during feature building, so no caller has to
remember to pass seam dates -- and a clean yfinance series stays untouched.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from app.data.curate import load_curated, save_curated
from app.data.seams import (
    FNSPID_SEAM_DATES,
    NO_SEAMS,
    SEAMS_BY_SOURCE,
    SOURCE_COLUMN,
    seams_for_frame,
    seams_for_source,
)
from app.data.sources.fnspid_price_source import FnspidPriceSource
from app.data.validate import validate_prices
from app.features.build import build_features

SEAM = FNSPID_SEAM_DATES[0]
ARCHIVE = Path("data/raw/prices/fnspid_full_history.zip")


# --------------------------------------------------------------------------
# The seam date lives in exactly one place
# --------------------------------------------------------------------------


def test_seam_date_is_defined_once(tmp_path):
    """Requirement: no duplicated 2020-07-06 literal across modules."""
    src = Path("src/app")
    offenders = [
        str(p)
        for p in src.rglob("*.py")
        if "2020-07-06" in p.read_text(encoding="utf-8") and p.name != "seams.py"
    ]
    # fnspid_price_source documents the seam in prose (SEAM_DATE for repair);
    # nothing else may hard-code it.
    assert offenders in ([], ["src\\app\\data\\sources\\fnspid_price_source.py"]) or all(
        "fnspid_price_source" in o for o in offenders
    ), f"seam date duplicated in: {offenders}"


def test_source_registry_maps_providers():
    assert seams_for_source("fnspid") == FNSPID_SEAM_DATES
    assert seams_for_source("yfinance") == NO_SEAMS
    assert seams_for_source("FNSPID") == FNSPID_SEAM_DATES, "case-insensitive"
    assert seams_for_source(None) == NO_SEAMS
    assert seams_for_source("something-new") == NO_SEAMS, "unknown -> no masking"
    assert set(SEAMS_BY_SOURCE) >= {"fnspid", "yfinance"}


# --------------------------------------------------------------------------
# Provenance survives curation
# --------------------------------------------------------------------------


def _panel(source: str | None, n_before: int = 120, n_after: int = 120) -> pd.DataFrame:
    before = pd.bdate_range(end=SEAM - pd.Timedelta(days=1), periods=n_before)
    after = pd.bdate_range(start=SEAM, periods=n_after)
    idx = before.append(after)
    close = 100 + np.linspace(0, 10, len(idx))
    close[:n_before] *= 4.0  # the archive artefact
    rng = np.random.default_rng(0)
    df = pd.DataFrame(
        {
            "date": idx,
            "ticker": "SEAMY",
            "open": close * 0.995,
            "high": close * 1.01,
            "low": close * 0.99,
            "close": close,
            "volume": 1e6 * (1 + rng.normal(0, 0.05, len(idx))),
        }
    )
    if source is not None:
        df[SOURCE_COLUMN] = source
    return df


def test_provenance_round_trips_through_curation(tmp_path):
    df = _panel("fnspid").drop(columns=[SOURCE_COLUMN])
    save_curated(df, tmp_path, "SEAMY", source="fnspid")
    back = load_curated(tmp_path)
    assert SOURCE_COLUMN in back.columns
    assert set(back[SOURCE_COLUMN]) == {"fnspid"}
    assert seams_for_frame(back) == FNSPID_SEAM_DATES


def test_curation_without_source_records_none(tmp_path):
    df = _panel(None)
    save_curated(df, tmp_path, "SEAMY")
    back = load_curated(tmp_path)
    assert SOURCE_COLUMN not in back.columns
    assert seams_for_frame(back) == NO_SEAMS


# --------------------------------------------------------------------------
# The wiring: features pick the seam up on their own
# --------------------------------------------------------------------------


def test_fnspid_provenance_invalidates_cross_seam_return_automatically():
    """No seam_dates argument anywhere -- it must be inferred."""
    feats = build_features(_panel("fnspid"), horizons=(1,)).set_index("date")
    assert pd.isna(feats["ret_1d"].loc[SEAM])


def test_yfinance_provenance_leaves_everything_valid():
    feats = build_features(_panel("yfinance"), horizons=(1,)).set_index("date")
    assert not pd.isna(feats["ret_1d"].loc[SEAM])


def test_yfinance_path_is_byte_identical_to_no_seam_handling():
    """Requirement 3: a clean yfinance series must be completely unchanged."""
    yf_panel = _panel("yfinance")
    plain = _panel(None)
    a = build_features(yf_panel, horizons=(1, 5))
    b = build_features(plain, horizons=(1, 5))
    pd.testing.assert_frame_equal(a, b)


def test_rolling_features_invalidated_via_provenance():
    feats = build_features(_panel("fnspid"), horizons=(1,)).set_index("date")
    pos = feats.index.searchsorted(SEAM)
    assert feats["sma_20"].iloc[pos : pos + 19].isna().all()
    assert not pd.isna(feats["sma_20"].iloc[pos + 19])
    assert feats["ret_5d"].iloc[pos : pos + 5].isna().all()


def test_forward_labels_invalidated_via_provenance():
    feats = build_features(_panel("fnspid"), horizons=(1, 5)).set_index("date")
    pos = feats.index.searchsorted(SEAM)
    assert pd.isna(feats["fwd_ret_1d"].iloc[pos - 1])
    assert pd.isna(feats["label_up_1d"].iloc[pos - 1])
    assert feats["fwd_ret_5d"].iloc[pos - 5 : pos].isna().all()
    assert feats["label_up_5d"].iloc[pos - 5 : pos].isna().all()


def test_prices_and_rows_untouched_by_the_wiring():
    panel = _panel("fnspid")
    before = panel.copy()
    feats = build_features(panel, horizons=(1,))
    pd.testing.assert_frame_equal(panel, before)
    assert len(feats) == len(panel)
    assert SEAM in set(feats["date"])


def test_explicit_argument_overrides_inference():
    """An explicit NO_SEAMS must be able to switch masking off."""
    feats = build_features(
        _panel("fnspid"), horizons=(1,), seam_dates=NO_SEAMS
    ).set_index("date")
    assert not pd.isna(feats["ret_1d"].loc[SEAM])


def test_mixed_provenance_is_resolved_per_ticker():
    """A yfinance ticker must not inherit FNSPID's seam from a shared panel."""
    fn = _panel("fnspid")
    yf = _panel("yfinance").assign(ticker="CLEAN")
    both = pd.concat([fn, yf], ignore_index=True)
    feats = build_features(both, horizons=(1,))

    seamy = feats[feats["ticker"] == "SEAMY"].set_index("date")
    clean = feats[feats["ticker"] == "CLEAN"].set_index("date")
    assert pd.isna(seamy["ret_1d"].loc[SEAM])
    assert not pd.isna(clean["ret_1d"].loc[SEAM]), "yfinance ticker was masked"


# --------------------------------------------------------------------------
# Against the real archive (skips when it is not cached)
# --------------------------------------------------------------------------


@pytest.mark.skipif(not ARCHIVE.exists(), reason="FNSPID price archive not cached")
def test_real_fnspid_ingest_to_features_path(tmp_path):
    """The genuine path: FnspidPriceSource -> validate -> curate -> features."""
    src = FnspidPriceSource()
    raw = src.fetch("XOM", start="2020-01-01", end="2020-12-31")
    clean, _ = validate_prices(raw, ticker="XOM")

    save_curated(clean, tmp_path, "XOM", source=src.name)
    curated = load_curated(tmp_path)
    assert seams_for_frame(curated) == FNSPID_SEAM_DATES

    feats = build_features(curated, horizons=(1, 5)).set_index("date")

    # The 2020-07-06 row and its prices survive.
    assert SEAM in feats.index
    assert len(feats) == len(curated)

    # XOM's real cross-seam return is -15.5%; it must not reach the features.
    assert pd.isna(feats["ret_1d"].loc[SEAM])
    assert feats["ret_1d"].abs().max() < 0.30

    # Backward and forward windows both handled.
    pos = feats.index.searchsorted(SEAM)
    assert feats["sma_20"].iloc[pos : pos + 19].isna().all()
    assert pd.isna(feats["fwd_ret_1d"].iloc[pos - 1])


@pytest.mark.skipif(not ARCHIVE.exists(), reason="FNSPID price archive not cached")
def test_real_unaffected_ticker_keeps_its_returns(tmp_path):
    """MSFT has no large seam; masking must still apply (its seam is small)
    but the series must otherwise be intact."""
    src = FnspidPriceSource()
    raw = src.fetch("MSFT", start="2020-01-01", end="2020-12-31")
    clean, _ = validate_prices(raw, ticker="MSFT")
    save_curated(clean, tmp_path, "MSFT", source=src.name)
    feats = build_features(load_curated(tmp_path), horizons=(1,)).set_index("date")

    assert pd.isna(feats["ret_1d"].loc[SEAM])
    valid = feats["ret_1d"].dropna()
    assert len(valid) > 200, "only the seam window should be removed"
    assert valid.abs().max() < 0.20
