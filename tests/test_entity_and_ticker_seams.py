"""Per-ticker adjustment seams and ticker-to-entity attribution.

Two defect classes that the archive-wide seam and the structural validators
both miss:

* A single FNSPID series stitched at its OWN date (NEE/NDAQ/EW 2020-04-02,
  ETN 2019-08-15, MO 2008-03-31). A persistent level step, so the local-context
  corruption check correctly ignores it, and not archive-wide, so the
  2020-07-06 seam does not cover it.
* A ticker that changed hands, so old rows describe a different company
  (IR, COR). No price rule can detect this; it needs an identity ruling.

These tests pin both, and pin what must NOT be touched: DHR's real spin-off
move and every ticker with no registered defect.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from app.data.entities import (
    NEWS_ENTITY_START,
    news_entity_reason,
    news_entity_start,
    news_is_current_entity,
)
from app.data.seams import (
    FNSPID_SEAM_DATES,
    NO_SEAMS,
    SEAMS_BY_SOURCE_TICKER,
    SOURCE_COLUMN,
    backward_window_invalid,
    forward_window_invalid,
    seams_for_frame,
    seams_for_source,
)

PANEL = Path("data/curated_sp500")
ARCHIVE_SEAM = FNSPID_SEAM_DATES[0]

# Verified individually against yfinance; see SEAMS_BY_SOURCE_TICKER.
REGISTERED = {
    "NEE": pd.Timestamp("2020-04-02"),
    "NDAQ": pd.Timestamp("2020-04-02"),
    "EW": pd.Timestamp("2020-04-02"),
    "ETN": pd.Timestamp("2019-08-15"),
    "MO": pd.Timestamp("2008-03-31"),
}


# --------------------------------------------------------------------------
# The registry
# --------------------------------------------------------------------------


def test_every_registered_ticker_resolves_to_its_own_date_plus_the_archive_seam():
    for ticker, date in REGISTERED.items():
        seams = seams_for_source("fnspid", ticker)
        assert date in seams, f"{ticker}: {date.date()} not registered"
        assert ARCHIVE_SEAM in seams, f"{ticker}: lost the archive-wide seam"
        assert seams == tuple(sorted(seams)), "seams must come back sorted"


def test_unregistered_fnspid_ticker_gets_only_the_archive_seam():
    for ticker in ("AAPL", "MSFT", "XOM", "JNJ"):
        assert seams_for_source("fnspid", ticker) == FNSPID_SEAM_DATES


def test_dhr_is_not_registered_its_move_is_a_real_spinoff():
    """DHR 2016-07-05 shows ~1.6x in BOTH sources (ratio step 0.9501x, i.e.
    none). Masking it would delete the genuine Fortive spin-off."""
    assert ("fnspid", "DHR") not in SEAMS_BY_SOURCE_TICKER
    assert seams_for_source("fnspid", "DHR") == FNSPID_SEAM_DATES


def test_gen_is_not_registered_it_failed_the_seam_test():
    """GEN's series is pervasively mismatched (1,584 sessions disagree >2%),
    not stitched at one date, so a seam is the wrong instrument."""
    assert ("fnspid", "GEN") not in SEAMS_BY_SOURCE_TICKER


def test_yfinance_tickers_never_inherit_a_per_ticker_seam():
    for ticker in REGISTERED:
        assert seams_for_source("yfinance", ticker) == NO_SEAMS


def test_omitting_the_ticker_yields_only_source_wide_seams():
    """A caller that does not know the ticker must not pick up another's."""
    assert seams_for_source("fnspid") == FNSPID_SEAM_DATES
    assert seams_for_source("fnspid", None) == FNSPID_SEAM_DATES


def test_unknown_source_still_resolves_to_no_seams():
    assert seams_for_source("some_new_vendor", "NEE") == NO_SEAMS
    assert seams_for_source(None, "NEE") == NO_SEAMS


def test_registry_keys_are_normalised_lowercase_source_uppercase_ticker():
    for source, ticker in SEAMS_BY_SOURCE_TICKER:
        assert source == source.lower()
        assert ticker == ticker.upper()


def test_lookup_is_case_insensitive():
    assert seams_for_source("FNSPID", "nee") == seams_for_source("fnspid", "NEE")


# --------------------------------------------------------------------------
# Resolution from a frame's provenance
# --------------------------------------------------------------------------


def frame(ticker: str, source: str, n: int = 5) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": pd.bdate_range("2020-01-01", periods=n),
            "ticker": ticker,
            SOURCE_COLUMN: source,
        }
    )


def test_frame_lookup_picks_up_the_per_ticker_seam():
    assert seams_for_frame(frame("NEE", "fnspid")) == (
        pd.Timestamp("2020-04-02"),
        ARCHIVE_SEAM,
    )


def test_frame_lookup_leaves_an_unregistered_ticker_alone():
    assert seams_for_frame(frame("AAPL", "fnspid")) == FNSPID_SEAM_DATES


def test_frame_lookup_respects_provenance_over_ticker():
    """A re-sourced NEE would carry no seam at all."""
    assert seams_for_frame(frame("NEE", "yfinance")) == NO_SEAMS


def test_frame_without_a_ticker_column_falls_back_to_source_wide():
    df = frame("NEE", "fnspid").drop(columns=["ticker"])
    assert seams_for_frame(df) == FNSPID_SEAM_DATES


def test_frame_without_provenance_has_no_seams():
    df = frame("NEE", "fnspid").drop(columns=[SOURCE_COLUMN])
    assert seams_for_frame(df) == NO_SEAMS


# --------------------------------------------------------------------------
# The seam actually masks features and labels
# --------------------------------------------------------------------------


def synthetic(ticker: str, seam: pd.Timestamp, step: float, source: str = "fnspid"):
    """A clean series with one artificial level step at `seam`."""
    before = pd.bdate_range(end=seam - pd.Timedelta(days=1), periods=120)
    after = pd.bdate_range(start=seam, periods=120)
    idx = before.append(after)
    close = 100 + np.linspace(0, 5, len(idx))
    close[: len(before)] /= step
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        {
            "date": idx,
            "ticker": ticker,
            "open": close * 0.999,
            "high": close * 1.004,
            "low": close * 0.996,
            "close": close,
            "volume": 1e6 * (1 + rng.normal(0, 0.05, len(idx))),
            SOURCE_COLUMN: source,
        }
    )


def test_a_registered_seam_masks_the_return_across_it():
    from app.features.build import build_features

    seam = pd.Timestamp("2020-04-02")
    panel = synthetic("NEE", seam, 0.2335)
    feats = build_features(panel, horizons=(5,)).set_index("date")
    assert pd.isna(feats.loc[seam, "ret_1d"]), "the step must not become a return"


def test_an_unregistered_ticker_with_the_same_shape_is_not_masked():
    """Proof the masking is driven by the registry, not by the data shape."""
    from app.features.build import build_features

    seam = pd.Timestamp("2020-04-02")
    panel = synthetic("AAPL", seam, 0.2335)
    feats = build_features(panel, horizons=(5,)).set_index("date")
    assert not pd.isna(feats.loc[seam, "ret_1d"])


def test_a_registered_seam_invalidates_forward_labels_that_cross_it():
    from app.features.build import build_features

    seam = pd.Timestamp("2020-04-02")
    panel = synthetic("NEE", seam, 0.2335)
    feats = build_features(panel, horizons=(5,)).set_index("date")
    idx = feats.index
    pos = idx.searchsorted(seam)
    crossing = idx[max(0, pos - 5) : pos]
    assert feats.loc[crossing, "fwd_ret_5d"].isna().all(), (
        "a 5-day forward return spanning the seam is not a real return"
    )
    assert feats.loc[crossing, "label_up_5d"].isna().all()


def test_deep_lookback_is_invalidated_for_longer_than_a_shallow_one():
    seam = pd.Timestamp("2020-04-02")
    # Long enough after the seam that a 63-deep window can run its full course
    # (it invalidates lookback-1 rows), but stopping short of the archive-wide
    # 2020-07-06 seam so this measures ONE boundary.
    idx = pd.bdate_range(end=seam, periods=80).append(
        pd.bdate_range(start=seam, periods=66)[1:]
    )
    seams = seams_for_source("fnspid", "NEE")
    assert sum(s in idx for s in seams) == 1, "index must span exactly one seam"
    shallow = backward_window_invalid(idx, 2, seams).sum()
    deep = backward_window_invalid(idx, 63, seams).sum()
    assert shallow == 1, "a 2-session window spans one transition: the seam row only"
    assert deep == 62, "a 63-session window stays contaminated for lookback-1 rows"
    assert deep > shallow


def test_two_seams_on_one_ticker_are_both_honoured():
    """NEE carries its own 2020-04-02 plus the archive-wide 2020-07-06."""
    seams = seams_for_source("fnspid", "NEE")
    assert len(seams) == 2
    idx = pd.bdate_range("2020-01-02", "2020-12-31")
    for seam in seams:
        pos = idx.searchsorted(seam)
        assert backward_window_invalid(idx, 21, seams)[pos], f"{seam.date()} not masked"
        assert forward_window_invalid(idx, 5, seams)[pos - 1], f"{seam.date()} label leak"


# --------------------------------------------------------------------------
# Entity attribution
# --------------------------------------------------------------------------


def test_news_before_a_handover_is_not_the_current_entity():
    assert not news_is_current_entity("IR", "2017-06-01")
    assert not news_is_current_entity("COR", "2015-01-01")


def test_news_after_a_handover_is_the_current_entity():
    assert news_is_current_entity("IR", "2020-03-02")
    assert news_is_current_entity("IR", "2020-06-01")
    assert news_is_current_entity("COR", "2023-08-30")


def test_the_boundary_date_itself_belongs_to_the_current_entity():
    assert news_is_current_entity("IR", news_entity_start("IR"))


def test_an_unregistered_ticker_keeps_all_its_news():
    for ticker in ("AAPL", "MSFT", "JNJ"):
        assert news_entity_start(ticker) is None
        assert news_is_current_entity(ticker, "2008-01-02")


def test_every_registered_entity_bound_has_a_documented_reason():
    for ticker in NEWS_ENTITY_START:
        reason = news_entity_reason(ticker)
        assert reason and len(reason) > 40, f"{ticker}: reason too thin"


def test_ir_and_cor_bounds_match_the_verified_handover_dates():
    assert news_entity_start("IR") == pd.Timestamp("2020-03-02")
    assert news_entity_start("COR") == pd.Timestamp("2023-08-30")


# --------------------------------------------------------------------------
# The rebuilt panel and artifacts
# --------------------------------------------------------------------------


@pytest.mark.skipif(not (PANEL / "prices").exists(), reason="expanded panel absent")
def test_ir_panel_starts_at_the_current_entitys_first_session():
    from app.data.curate import load_curated

    g = load_curated(PANEL, tickers=["IR"])
    assert len(g), "IR missing from the panel"
    assert g["date"].min() == pd.Timestamp("2017-05-12")
    assert set(g["source"]) == {"yfinance"}


@pytest.mark.skipif(not (PANEL / "prices").exists(), reason="expanded panel absent")
def test_reentity_sourced_tickers_are_clean_and_yfinance():
    from app.data.curate import load_curated

    for ticker in ("IR", "COR", "GEN"):
        g = load_curated(PANEL, tickers=[ticker]).sort_values("date")
        assert len(g) > 1000, f"{ticker} missing"
        assert set(g["source"]) == {"yfinance"}, f"{ticker} provenance not updated"
        assert g["close"].pct_change().abs().max() < 0.5, f"{ticker} still discontinuous"


@pytest.mark.skipif(not (PANEL / "prices").exists(), reason="expanded panel absent")
def test_the_predecessor_is_still_present_under_its_own_ticker():
    """Excluding IR's pre-2017 rows must not lose Ingersoll-Rand plc: it is
    TT, which stands on its own."""
    from app.data.curate import load_curated

    tt = load_curated(PANEL, tickers=["TT"])
    assert len(tt) > 1000
    assert tt["date"].min() <= pd.Timestamp("2008-01-31")


AUDITED_V2 = Path("data/curated/news_sentiment_daily_sp500_audited_v2.parquet")


@pytest.mark.skipif(not AUDITED_V2.exists(), reason="audited v2 artifact absent")
def test_no_misattributed_news_survives_in_the_audited_artifact():
    news = pd.read_parquet(AUDITED_V2)
    for ticker, (start, _) in NEWS_ENTITY_START.items():
        rows = news[news["ticker"] == ticker]
        if not len(rows):
            continue
        assert rows["trade_date"].min() >= start, (
            f"{ticker}: news dated {rows['trade_date'].min().date()} predates the "
            f"handover on {start.date()}"
        )
