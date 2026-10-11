"""Phase 3 rolling-window tests.

Covers window membership and boundary dates, per-horizon outcome validity,
minimum-sample rules, the stock fast path's equivalence to the reused Phase 1
describe routine, reproducibility, and non-interference with Phase 0/1/2
artifacts.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from app.events.benchmark_returns import abnormal_ret_col, valid_col
from app.events.category_horizon_analysis import _describe_group, session_positions
from app.events.rolling_impact import (
    ROLL_MIN_EVENTS,
    ROLL_MIN_TICKERS,
    ROLL_STOCK_EXPLORATORY_MIN,
    ROLL_STOCK_MIN_EVENTS,
    TREND_MIN_WINDOWS,
    _direction_label,
    _label_change,
    _sparkline,
    _stock_describe_fast,
    _stride_label,
    candidate_concentration,
    classify_temporal_change,
    data_freshness,
    non_overlapping_subset,
    rolling_category_horizon,
    rolling_stock,
    run,
    window_grid,
)

HORIZONS_T = (1, 20)
SESSIONS = pd.bdate_range("2009-01-01", periods=4200)
SESSION_POS = pd.Series(np.arange(len(SESSIONS)), index=SESSIONS)

PHASE0_RETURNS = Path("data/curated/benchmark_adjusted_returns_v1.parquet")
PHASE1_EVENTS = Path("data/curated/phase1_events_v1.parquet")
PHASE1_SUMMARY = Path("data/curated/phase1_category_horizon_summary_v1.parquet")
PHASE2_SUMMARY = Path("data/curated/phase2_stock_category_horizon_summary_v1.parquet")
PHASE2_RANKING = Path("data/curated/phase2_stock_ranking_v1.parquet")
MATRIX_H5 = Path("data/curated/matrix_h5_v4.parquet")


def make_events(rows, horizons=HORIZONS_T) -> pd.DataFrame:
    """Build a phase1_events_v1-shaped frame. Each row dict needs ticker, date,
    category, abnormal_ret; `valid` defaults True and may be set per horizon as
    valid_{h}d to exercise per-horizon validity."""
    out = []
    for r in rows:
        rec = {
            "ticker": r["ticker"],
            "date": pd.Timestamp(r["date"]),
            "category": r["category"],
            "n_articles": r.get("n_articles", 1),
            "source": "test",
            "sector": r.get("sector", "Industrials"),
        }
        for h in horizons:
            rec[f"stock_fwd_ret_{h}d"] = r["abnormal_ret"]
            rec[f"benchmark_fwd_ret_{h}d"] = 0.0
            rec[abnormal_ret_col(h)] = r["abnormal_ret"]
            rec[valid_col(h)] = bool(r.get(f"valid_{h}d", r.get("valid", True)))
        out.append(rec)
    return pd.DataFrame(out)


def spread_events(ticker, category, dates, mean, seed=0, **kw):
    rng = np.random.default_rng(seed)
    return [
        {"ticker": ticker, "category": category, "date": d,
         "abnormal_ret": float(rng.normal(mean, 0.01)), **kw}
        for d in dates
    ]


# --------------------------------------------------------------------------
# Window membership and boundary dates
# --------------------------------------------------------------------------


def test_window_is_exactly_365_calendar_days_inclusive():
    g = window_grid("2015-01-01", "2020-01-01", window_days=365, stride_days=91)
    for start, end in g:
        assert (end - start).days + 1 == 365


def test_last_window_ends_exactly_on_last_data_date():
    g = window_grid("2015-01-01", "2020-06-30", window_days=365, stride_days=91)
    assert g[-1][1] == pd.Timestamp("2020-06-30")


def test_windows_are_chronological_and_fit_inside_the_data_range():
    first, last = pd.Timestamp("2015-01-01"), pd.Timestamp("2020-06-30")
    g = window_grid(first, last, window_days=365, stride_days=91)
    assert g == sorted(g)
    assert all(s >= first and e <= last for s, e in g)


def test_no_window_emitted_when_range_shorter_than_window():
    assert window_grid("2020-01-01", "2020-06-01", window_days=365, stride_days=91) == []


def test_event_exactly_on_either_window_boundary_is_included():
    """Both window edges are inclusive, so an event on the first or last day of
    the window must contribute -- an off-by-one here silently drops events.

    All dates are taken from the session calendar: real event dates always sit
    on a trading session, so a synthetic weekend date would be an unrealistic
    input (and would map to no session position at all).
    """
    i = 1800
    before, start = SESSIONS[i - 1], SESSIONS[i]
    end, after = SESSIONS[i + 250], SESSIONS[i + 251]
    rows = [
        {"ticker": "AAA", "category": "EARNINGS_RESULTS", "date": start, "abnormal_ret": 0.01},
        {"ticker": "BBB", "category": "EARNINGS_RESULTS", "date": end, "abnormal_ret": 0.02},
        {"ticker": "CCC", "category": "EARNINGS_RESULTS", "date": before, "abnormal_ret": 0.99},
        {"ticker": "DDD", "category": "EARNINGS_RESULTS", "date": after, "abnormal_ret": -0.99},
    ]
    events = make_events(rows)
    in_win = events[(events["date"] >= start) & (events["date"] <= end)]
    assert set(in_win["ticker"]) == {"AAA", "BBB"}
    desc = _describe_group(in_win, 1, SESSION_POS)
    assert desc["n_events_valid"] == 2
    assert desc["mean_abnormal_return"] == pytest.approx(0.015)


def test_non_overlapping_subset_windows_share_no_calendar_days():
    g = window_grid("2009-04-29", "2024-01-08", window_days=365, stride_days=91)
    ends = [e for _, e in g]
    keep = non_overlapping_subset(ends, 365)
    chosen = [ends[i] for i in keep]
    gaps = [(chosen[i + 1] - chosen[i]).days for i in range(len(chosen) - 1)]
    assert all(gap >= 365 for gap in gaps)


def test_non_overlapping_subset_always_includes_the_most_recent_window():
    """Recency is the point of this phase, so the newest window must never be
    the one dropped when thinning to a disjoint set."""
    g = window_grid("2009-04-29", "2024-01-08", window_days=365, stride_days=91)
    ends = [e for _, e in g]
    keep = non_overlapping_subset(ends, 365)
    assert keep[-1] == len(ends) - 1


def test_non_overlapping_subset_is_sorted_and_handles_empty():
    assert non_overlapping_subset([], 365) == []
    g = window_grid("2015-01-01", "2020-01-01", 365, 91)
    keep = non_overlapping_subset([e for _, e in g], 365)
    assert keep == sorted(keep)


# --------------------------------------------------------------------------
# Horizon validity
# --------------------------------------------------------------------------


def test_event_invalid_at_a_horizon_does_not_contribute_at_that_horizon():
    """Phase 0's per-horizon validity flags are the gate: the same event can
    count at 1d and be excluded at 20d when its 20d window is incomplete."""
    rows = spread_events(
        "AAA", "EARNINGS_RESULTS", SESSIONS[1600:1640], 0.01, seed=1
    )
    for r in rows[:15]:
        r["valid_20d"] = False
    events = make_events(rows)
    d1 = _describe_group(events, 1, SESSION_POS)
    d20 = _describe_group(events, 20, SESSION_POS)
    assert d1["n_events_valid"] == 40
    assert d20["n_events_valid"] == 25
    assert d20["n_events_invalid"] == 15


def test_freshness_reports_a_later_frontier_for_short_horizons():
    rows = spread_events(
        "AAA", "EARNINGS_RESULTS", SESSIONS[1600:1630], 0.0, seed=2
    )
    for r in rows[-10:]:
        r["valid_20d"] = False
    f = data_freshness(make_events(rows), HORIZONS_T, as_of="2020-01-01")
    assert f["per_horizon"]["1d"]["last_valid_event_date"] > (
        f["per_horizon"]["20d"]["last_valid_event_date"]
    )
    assert f["per_horizon"]["20d"]["n_events_invalid"] == 10
    assert f["real_time"] is False


def test_freshness_staleness_is_measured_against_as_of():
    dates = SESSIONS[1600:1605]
    last = dates[-1]
    as_of = last + pd.Timedelta(days=10)
    f = data_freshness(make_events(spread_events("AAA", "OTHER", dates, 0.0)), HORIZONS_T,
                       as_of=as_of)
    assert f["last_news_date"] == str(last.date())
    assert f["news_staleness_days_vs_as_of"] == 10


# --------------------------------------------------------------------------
# Minimum-sample rules
# --------------------------------------------------------------------------


def _category_plus_long_other(cat_rows):
    """A category's events plus an OTHER baseline long enough that the data
    span exceeds one window, so at least one window exists to report on."""
    other = spread_events("Z", "OTHER", SESSIONS[1500:2100], 0.0, seed=99)
    return make_events(cat_rows + other)


def test_window_below_min_events_is_not_sufficient():
    rows = []
    for i, d in enumerate(SESSIONS[1600:1600 + ROLL_MIN_EVENTS - 1]):
        rows += spread_events(f"T{i % ROLL_MIN_TICKERS}", "MANAGEMENT", [d], 0.02, seed=i)
    r = rolling_category_horizon(
        _category_plus_long_other(rows), SESSION_POS, horizons=(1,), stride_days=91
    )
    mg = r[(r["category"] == "MANAGEMENT") & (r["n_events_valid"] > 0)]
    assert len(mg)
    assert (mg["n_events_valid"] < ROLL_MIN_EVENTS).all()
    assert not mg["window_sufficient"].any()


def test_window_below_min_tickers_is_not_sufficient_even_with_many_events():
    """Breadth matters: many events from 3 tickers is not a category-wide claim."""
    rows = []
    for i, d in enumerate(SESSIONS[1600:1700]):
        rows += spread_events(f"T{i % 3}", "MANAGEMENT", [d], 0.02, seed=i)
    r = rolling_category_horizon(
        _category_plus_long_other(rows), SESSION_POS, horizons=(1,), stride_days=91
    )
    mg = r[(r["category"] == "MANAGEMENT") & (r["n_events_valid"] >= ROLL_MIN_EVENTS)]
    assert len(mg)
    assert (mg["n_tickers"] < ROLL_MIN_TICKERS).all()
    assert not mg["window_sufficient"].any()


def test_window_meeting_both_minimums_is_sufficient():
    rows = []
    for i, d in enumerate(SESSIONS[1600:1700]):
        rows += spread_events(f"T{i % 12}", "MANAGEMENT", [d], 0.02, seed=i)
    r = rolling_category_horizon(
        _category_plus_long_other(rows), SESSION_POS, horizons=(1,), stride_days=91
    )
    mg = r[
        (r["category"] == "MANAGEMENT")
        & (r["n_events_valid"] >= ROLL_MIN_EVENTS)
        & (r["n_tickers"] >= ROLL_MIN_TICKERS)
    ]
    assert len(mg)
    assert mg["window_sufficient"].all()


def test_insufficient_window_is_never_labelled_as_no_effect():
    """An under-sampled window must read as 'inconclusive', which means unknown
    -- it must never be presented as a zero/absent effect."""
    assert _direction_label(None, None) == "inconclusive"
    assert _direction_label(float("nan"), float("nan")) == "inconclusive"
    assert _direction_label(-0.01, 0.01) == "inconclusive"
    assert _direction_label(0.001, 0.01) == "positive"
    assert _direction_label(-0.01, -0.001) == "negative"


def test_stock_tier_ceiling_is_exploratory_and_boundaries_are_exact():
    from app.events.rolling_impact import _stock_evidence_tier

    assert _stock_evidence_tier(ROLL_STOCK_MIN_EVENTS) == "exploratory"
    assert _stock_evidence_tier(ROLL_STOCK_MIN_EVENTS - 1) == "sparse"
    assert _stock_evidence_tier(ROLL_STOCK_EXPLORATORY_MIN) == "sparse"
    assert _stock_evidence_tier(ROLL_STOCK_EXPLORATORY_MIN - 1) == "insufficient"
    # no stock window can ever be called adequately supported
    tiers = {_stock_evidence_tier(n) for n in range(0, 500)}
    assert "adequately_supported" not in tiers


# --------------------------------------------------------------------------
# Stock fast path equals the reused Phase 1 routine
# --------------------------------------------------------------------------


def test_stock_fast_path_matches_describe_group():
    """The stock path skips _describe_group's pandas date-aggregation for speed.
    That is only legitimate because one row per (ticker, date, category) makes
    aggregation an identity, so the two must agree to floating-point equality."""
    rows = spread_events(
        "AAA", "EARNINGS_RESULTS", SESSIONS[1600:1720], 0.01, seed=3
    )
    events = make_events(rows)
    for h in HORIZONS_T:
        v = events[events[valid_col(h)]]
        fast = _stock_describe_fast(
            v[abnormal_ret_col(h)].to_numpy(float),
            session_positions(v["date"], SESSION_POS).astype(float),
            h,
        )
        slow = _describe_group(events, h, SESSION_POS)
        for key in (
            "n_events_valid", "mean_abnormal_return", "median_abnormal_return",
            "p25_abnormal_return", "p75_abnormal_return", "std_abnormal_return",
            "pct_positive", "pct_negative", "nw_mean", "nw_se", "nw_ci95_lo", "nw_ci95_hi",
        ):
            assert fast[key] == pytest.approx(slow[key]), key


def test_event_table_has_one_row_per_ticker_date_category_fast_path_premise():
    rows = spread_events("AAA", "EARNINGS_RESULTS", pd.bdate_range("2016-01-04", periods=10), 0.0)
    rows += spread_events("AAA", "OTHER", pd.bdate_range("2016-01-04", periods=10), 0.0)
    events = make_events(rows)
    assert not events.duplicated(subset=["ticker", "date", "category"]).any()


@pytest.mark.skipif(not PHASE1_EVENTS.exists(), reason="Phase 1 events absent")
def test_real_event_table_upholds_the_fast_path_premise():
    """If the real event table ever gains duplicate (ticker, date, category)
    rows, the stock fast path's identity-aggregation assumption breaks."""
    events = pd.read_parquet(PHASE1_EVENTS, columns=["ticker", "date", "category"])
    assert not events.duplicated(subset=["ticker", "date", "category"]).any()


# --------------------------------------------------------------------------
# Stock-level rolling behaviour
# --------------------------------------------------------------------------


def test_phase2_candidate_rows_are_emitted_even_when_far_too_sparse():
    """Their sparsity is the finding; dropping them would hide it."""
    rows = spread_events(
        "BR", "MANAGEMENT", SESSIONS[[1600, 1860, 2120]], -0.04
    )
    rows += spread_events("ZZ", "OTHER", SESSIONS[1600:2000], 0.0, seed=4)
    df, scope = rolling_stock(make_events(rows), SESSION_POS, horizons=(1,), stride_days=365)
    cand = df[(df["ticker"] == "BR") & (df["category"] == "MANAGEMENT")]
    assert len(cand) > 0
    assert cand["phase2_candidate"].all()
    assert not cand["window_sufficient"].any()
    assert set(cand["evidence_tier"]) <= {"insufficient", "sparse"}


def test_suppressed_stock_cells_are_counted_not_hidden():
    rows = spread_events("QQ", "MANAGEMENT", pd.to_datetime(["2016-02-01", "2017-03-01"]), 0.01)
    rows += spread_events("QQ", "EARNINGS_RESULTS", SESSIONS[1600:1660],
                          0.01, seed=8)
    df, scope = rolling_stock(make_events(rows), SESSION_POS, horizons=(1,), stride_days=365)
    assert scope["n_window_cells_suppressed_below_min"] >= 0
    assert scope["n_ticker_category_pairs_analysed"] <= (
        scope["n_ticker_category_pairs_screened"]
    )
    # the 2-event MANAGEMENT pair cannot reach the minimum and is screened out
    assert df[(df["ticker"] == "QQ") & (df["category"] == "MANAGEMENT")].empty


def test_stock_window_sufficiency_follows_the_documented_threshold():
    rows = spread_events("AAA", "EARNINGS_RESULTS", SESSIONS[1600:1900], 0.01, seed=9)
    df, _ = rolling_stock(make_events(rows), SESSION_POS, horizons=(1,), stride_days=91)
    ok = df[df["n_events_valid"] >= ROLL_STOCK_MIN_EVENTS]
    assert len(ok)
    assert ok["window_sufficient"].all()
    assert (df[df["n_events_valid"] < ROLL_STOCK_MIN_EVENTS]["window_sufficient"] == False).all()  # noqa: E712


# --------------------------------------------------------------------------
# Temporal change
# --------------------------------------------------------------------------


def _two_regime_rolling(early_mean, late_mean, n_tickers=12, seed=11):
    """Category whose effect is `early_mean` in the first years and `late_mean`
    in the last, with a dense OTHER baseline throughout."""
    rows = []
    early = SESSIONS[260:520]
    late = SESSIONS[2800:3060]
    for i, d in enumerate(early):
        rows += spread_events(f"T{i % n_tickers}", "MANAGEMENT", [d], early_mean, seed=seed + i)
    for i, d in enumerate(late):
        rows += spread_events(f"T{i % n_tickers}", "MANAGEMENT", [d], late_mean,
                              seed=seed + 5000 + i)
    for i, d in enumerate(SESSIONS[260:3100]):
        rows += spread_events(f"O{i % n_tickers}", "OTHER", [d], 0.0, seed=seed + 9000 + i)
    return make_events(rows)


def test_too_few_sufficient_windows_reports_insufficient_not_a_trend():
    rows = spread_events("AAA", "MANAGEMENT", SESSIONS[[1600, 1860]], 0.05)
    rows += spread_events("ZZ", "OTHER", SESSIONS[1600:2100], 0.0, seed=12)
    r = rolling_category_horizon(make_events(rows), SESSION_POS, horizons=(1,), stride_days=365)
    ch = classify_temporal_change(r, horizons=(1,))
    mg = ch[(ch["category"] == "MANAGEMENT") & (ch["metric"] == "vs_other_diff")]
    assert len(mg) == 1
    assert mg.iloc[0]["change_label"] == "insufficient_windows"
    assert mg.iloc[0]["n_nonoverlapping_sufficient"] < TREND_MIN_WINDOWS
    assert pd.isna(mg.iloc[0]["diff_late_minus_early"])


def test_sign_flip_is_not_called_weakened():
    """A move from -0.09pp to +0.05pp is a direction change, not a weakening --
    applying a magnitude ratio across a sign boundary misdescribes it."""
    early = pd.Series({"m": -0.0009, "lo": -0.004, "hi": 0.002})
    late = pd.Series({"m": 0.0005, "lo": -0.002, "hi": 0.003})
    assert _label_change(early, late, "m", "lo", "hi") == "direction_unstable"


def test_sign_flip_with_both_intervals_excluding_zero_is_reversed():
    early = pd.Series({"m": -0.02, "lo": -0.03, "hi": -0.01})
    late = pd.Series({"m": 0.02, "lo": 0.01, "hi": 0.03})
    assert _label_change(early, late, "m", "lo", "hi") == "reversed"


def test_same_sign_magnitude_changes_are_labelled_by_ratio():
    base = {"lo": -1.0, "hi": 1.0}  # intervals deliberately wide: label is size-based
    early = pd.Series({"m": 0.01, **base})
    assert _label_change(early, pd.Series({"m": 0.02, **base}), "m", "lo", "hi") == "strengthened"
    assert _label_change(early, pd.Series({"m": 0.004, **base}), "m", "lo", "hi") == "weakened"
    assert _label_change(
        early, pd.Series({"m": 0.0105, **base}), "m", "lo", "hi"
    ) == "broadly_consistent"


def test_change_established_requires_both_fdr_and_disjoint_intervals():
    events = _two_regime_rolling(0.0, 0.0)
    r = rolling_category_horizon(events, SESSION_POS, horizons=(1,), stride_days=182)
    ch = classify_temporal_change(r, horizons=(1,))
    assert not ch.empty
    for _, row in ch.iterrows():
        if row["change_established"]:
            assert row["diff_significant_fdr"]
            assert row["intervals_disjoint"]


def test_recent_window_insufficiency_is_flagged_explicitly():
    """A category that stops generating news must come back as
    insufficient_recent_data, not as a weakened or absent effect."""
    rows = []
    for i, d in enumerate(SESSIONS[260:660]):
        rows += spread_events(f"T{i % 12}", "MANAGEMENT", [d], 0.02, seed=20 + i)
    for i, d in enumerate(SESSIONS[260:3060]):
        rows += spread_events(f"O{i % 12}", "OTHER", [d], 0.0, seed=30000 + i)
    r = rolling_category_horizon(make_events(rows), SESSION_POS, horizons=(1,), stride_days=182)
    ch = classify_temporal_change(r, horizons=(1,))
    mg = ch[(ch["category"] == "MANAGEMENT") & (ch["metric"] == "vs_other_diff")].iloc[0]
    assert mg["recent_status"] == "insufficient_recent_data"
    assert not mg["recent_window_sufficient"]


def test_temporal_change_compares_only_disjoint_windows():
    """n_nonoverlapping_sufficient can never exceed the count of disjoint
    windows available, which is what keeps the comparison honest."""
    events = _two_regime_rolling(0.01, 0.03)
    r = rolling_category_horizon(events, SESSION_POS, horizons=(1,), stride_days=91)
    ch = classify_temporal_change(r, horizons=(1,))
    for _, row in ch.iterrows():
        assert row["n_nonoverlapping_sufficient"] <= row["n_nonoverlapping_windows"]
        assert row["n_nonoverlapping_windows"] <= row["n_windows_total"]


# --------------------------------------------------------------------------
# Candidate concentration
# --------------------------------------------------------------------------


def test_candidate_with_too_few_events_reports_insufficient_evidence():
    rows = spread_events(
        "BR", "MANAGEMENT",
        SESSIONS[[1600, 1860, 2120, 2380]], -0.04, seed=40,
    )
    rows += spread_events("ZZ", "OTHER", SESSIONS[1600:1900], 0.0, seed=41)
    out = candidate_concentration(
        make_events(rows), SESSION_POS, candidates=(("BR", "MANAGEMENT", 1),)
    )
    rec = out[0]
    assert rec["n_events_valid_full_history"] == 4
    assert not rec["any_window_meets_rolling_min"]
    assert "INSUFFICIENT EVIDENCE" in rec["verdict"]
    assert rec["event_dominated"]


def test_candidate_leave_one_out_detects_single_event_domination():
    dates = SESSIONS[1600:1640]
    rows = [
        {"ticker": "XX", "category": "MANAGEMENT", "date": d, "abnormal_ret": 0.0}
        for d in dates
    ]
    rows[0]["abnormal_ret"] = 0.80  # one dominant outlier
    out = candidate_concentration(
        make_events(rows), SESSION_POS, candidates=(("XX", "MANAGEMENT", 1),)
    )
    rec = out[0]
    assert rec["mean_excl_most_extreme_event"] == pytest.approx(0.0)
    assert rec["pct_change_in_mean_excl_most_extreme"] == pytest.approx(100.0)
    assert rec["event_dominated"]
    assert rec["largest_single_event_share_of_sum_pct"] == pytest.approx(100.0)


def test_candidate_not_dominated_when_effect_is_broad_based():
    dates = SESSIONS[1600:1800]
    rows = spread_events("YY", "MANAGEMENT", dates, -0.02, seed=42)
    out = candidate_concentration(
        make_events(rows), SESSION_POS, candidates=(("YY", "MANAGEMENT", 1),)
    )
    rec = out[0]
    assert rec["pct_change_in_mean_excl_most_extreme"] < 25.0
    assert not rec["event_dominated"]
    assert rec["any_window_meets_rolling_min"]


# --------------------------------------------------------------------------
# Reporting helpers
# --------------------------------------------------------------------------


def test_sparkline_leaves_a_gap_for_missing_windows():
    s = _sparkline([0.0, None, 1.0])
    assert len(s) == 3
    assert s[1] == " "


def test_sparkline_handles_all_missing_and_flat_series():
    assert _sparkline([None, None]) == ""
    flat = _sparkline([0.5, 0.5, 0.5])
    assert len(flat) == 3 and len(set(flat)) == 1


def test_stride_label_is_derived_not_hardcoded():
    assert _stride_label(91) == "~quarterly"
    assert _stride_label(182) == "~semi-annual"
    assert _stride_label(365) == "~annual"
    assert _stride_label(900) == "every 900 days"


# --------------------------------------------------------------------------
# Reproducibility
# --------------------------------------------------------------------------


def test_rolling_category_is_deterministic():
    events = _two_regime_rolling(0.01, 0.02)
    a = rolling_category_horizon(events, SESSION_POS, horizons=(1,), stride_days=182)
    b = rolling_category_horizon(events, SESSION_POS, horizons=(1,), stride_days=182)
    pd.testing.assert_frame_equal(a, b)


def test_rolling_stock_is_deterministic():
    events = _two_regime_rolling(0.01, 0.02)
    a, sa = rolling_stock(events, SESSION_POS, horizons=(1,), stride_days=365)
    b, sb = rolling_stock(events, SESSION_POS, horizons=(1,), stride_days=365)
    pd.testing.assert_frame_equal(a, b)
    assert sa == sb


def test_temporal_change_is_deterministic():
    events = _two_regime_rolling(0.01, 0.02)
    r = rolling_category_horizon(events, SESSION_POS, horizons=(1,), stride_days=182)
    pd.testing.assert_frame_equal(
        classify_temporal_change(r, horizons=(1,)), classify_temporal_change(r, horizons=(1,))
    )


def test_freshness_is_reproducible_for_a_fixed_as_of():
    events = _two_regime_rolling(0.01, 0.02)
    assert data_freshness(events, (1,), as_of="2025-01-01") == data_freshness(
        events, (1,), as_of="2025-01-01"
    )


# --------------------------------------------------------------------------
# Non-interference with earlier phases
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "path",
    [PHASE0_RETURNS, PHASE1_EVENTS, PHASE1_SUMMARY, PHASE2_SUMMARY, PHASE2_RANKING, MATRIX_H5],
)
def test_importing_and_running_phase3_does_not_touch_earlier_outputs(path):
    if not path.exists():
        pytest.skip(f"{path} absent")
    before = path.stat().st_mtime, path.stat().st_size
    events = _two_regime_rolling(0.01, 0.02)
    r = rolling_category_horizon(events, SESSION_POS, horizons=(1,), stride_days=365)
    classify_temporal_change(r, horizons=(1,))
    rolling_stock(events, SESSION_POS, horizons=(1,), stride_days=365)
    after = path.stat().st_mtime, path.stat().st_size
    assert before == after


def test_phase3_writes_only_phase3_prefixed_files(tmp_path):
    events = _two_regime_rolling(0.01, 0.02)
    events_path = tmp_path / "events.parquet"
    events.to_parquet(events_path, index=False)

    from app.data.curate import save_curated

    panel_dir = tmp_path / "panel"
    for t in sorted(events["ticker"].unique())[:3]:
        save_curated(
            pd.DataFrame({
                "date": SESSIONS, "ticker": t, "open": 100.0, "high": 101.0,
                "low": 99.0, "close": 100.0, "volume": 1e6,
            }),
            panel_dir, t, source="yfinance",
        )
    out_dir = tmp_path / "out"
    run(
        events_path=events_path, panel_dir=panel_dir, out_dir=out_dir, version="t",
        horizons=(1,), category_stride_days=365, stock_stride_days=365,
        as_of="2026-01-01", figures=False,
    )
    written = sorted(p.name for p in out_dir.iterdir())
    assert written, "run() wrote nothing"
    assert all(n.startswith("phase3_") for n in written), written


def test_run_end_to_end_writes_all_outputs_and_states_staleness(tmp_path):
    events = _two_regime_rolling(0.01, 0.03)
    events_path = tmp_path / "events.parquet"
    events.to_parquet(events_path, index=False)

    from app.data.curate import save_curated

    panel_dir = tmp_path / "panel"
    for t in sorted(events["ticker"].unique())[:3]:
        save_curated(
            pd.DataFrame({
                "date": SESSIONS, "ticker": t, "open": 100.0, "high": 101.0,
                "low": 99.0, "close": 100.0, "volume": 1e6,
            }),
            panel_dir, t, source="yfinance",
        )

    out_dir = tmp_path / "out"
    summary = run(
        events_path=events_path, panel_dir=panel_dir, out_dir=out_dir, version="t",
        horizons=(1,), category_stride_days=182, stock_stride_days=365,
        as_of="2026-01-01", figures=False,
    )
    for key in ("rolling_category_horizon", "rolling_stock", "temporal_change"):
        assert Path(summary["outputs"][key]).exists()
    assert Path(summary["freshness_path"]).exists()
    report = Path(summary["report_path"]).read_text(encoding="utf-8")
    assert "NOT real-time" in report
    assert "causation" in report.lower()
    assert summary["freshness"]["real_time"] is False
    assert summary["freshness"]["news_staleness_days_vs_as_of"] > 0


def test_run_is_reproducible_across_two_invocations(tmp_path):
    events = _two_regime_rolling(0.01, 0.02)
    events_path = tmp_path / "events.parquet"
    events.to_parquet(events_path, index=False)

    from app.data.curate import save_curated

    panel_dir = tmp_path / "panel"
    for t in sorted(events["ticker"].unique())[:3]:
        save_curated(
            pd.DataFrame({
                "date": SESSIONS, "ticker": t, "open": 100.0, "high": 101.0,
                "low": 99.0, "close": 100.0, "volume": 1e6,
            }),
            panel_dir, t, source="yfinance",
        )
    kw = dict(
        events_path=events_path, panel_dir=panel_dir, horizons=(1,),
        category_stride_days=365, stock_stride_days=365, as_of="2026-01-01", figures=False,
    )
    s1 = run(out_dir=tmp_path / "o1", version="t", **kw)
    s2 = run(out_dir=tmp_path / "o2", version="t", **kw)
    for key in ("rolling_category_horizon", "temporal_change", "rolling_stock"):
        pd.testing.assert_frame_equal(
            pd.read_parquet(s1["outputs"][key]), pd.read_parquet(s2["outputs"][key])
        )
    r1 = Path(s1["report_path"]).read_text(encoding="utf-8")
    r2 = Path(s2["report_path"]).read_text(encoding="utf-8")
    assert r1 == r2
