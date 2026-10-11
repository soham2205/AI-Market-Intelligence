"""Phase 1: news category x investment horizon (and x sector) returns.

Synthetic fixtures keep these fast; tests against the real built artifacts
skip when they have not been generated.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from app.events.benchmark_returns import abnormal_ret_col, bench_ret_col, stock_ret_col, valid_col
from app.events.category_horizon_analysis import (
    MIN_EVENTS,
    MIN_OVERLAP_DATES,
    MIN_TICKERS,
    benjamini_hochberg,
    build_event_table,
    category_horizon_summary,
    category_vs_other,
    newey_west_mean_ci,
    newey_west_mean_ci_irregular,
    run,
    sector_category_horizon_summary,
    session_positions,
)

REAL_EVENTS = Path("data/curated/phase1_events_v1.parquet")
REAL_CAT_SUMMARY = Path("data/curated/phase1_category_horizon_summary_v1.parquet")
REAL_CAT_VS_OTHER = Path("data/curated/phase1_category_vs_other_v1.parquet")
REAL_COVERAGE = Path("data/curated/phase1_coverage_v1.json")
MATRIX_H5 = Path("data/curated/matrix_h5_v4.parquet")
PHASE0_OUTPUT = Path("data/curated/benchmark_adjusted_returns_v1.parquet")


# --------------------------------------------------------------------------
# Synthetic fixtures
# --------------------------------------------------------------------------


def make_classified_articles(rows: list[dict]) -> pd.DataFrame:
    """Each row: dict with ticker, published_at (str, ET-ish), category,
    label, score. article_id/headline/source/url filled in automatically."""
    out = []
    for i, r in enumerate(rows):
        out.append(
            {
                "article_id": f"a{i}",
                "headline": f"headline {i}",
                "source": "unknown",
                "url": f"https://example.com/{i}",
                "published_at": pd.Timestamp(r["published_at"], tz="UTC"),
                "ticker": r["ticker"],
                "label": r.get("label", "neutral"),
                "score": r.get("score", 0.0),
                "category": r["category"],
            }
        )
    return pd.DataFrame(out)


def make_returns(tickers, sessions, horizons=(1, 5), seed=0) -> pd.DataFrame:
    """A Phase-0-shaped returns table: valid everywhere except a few
    deliberately-invalid rows the caller can inspect."""
    rng = np.random.default_rng(seed)
    rows = []
    for t in tickers:
        for d in sessions:
            row = {"date": d, "ticker": t, "source": "yfinance"}
            for h in horizons:
                row[stock_ret_col(h)] = rng.normal(0, 0.02)
                row[bench_ret_col(h)] = rng.normal(0, 0.01)
                row[abnormal_ret_col(h)] = row[stock_ret_col(h)] - row[bench_ret_col(h)]
                row[valid_col(h)] = True
            rows.append(row)
    return pd.DataFrame(rows)


def write_returns(tmp_path: Path, returns: pd.DataFrame) -> Path:
    p = tmp_path / "returns.parquet"
    returns.to_parquet(p, index=False)
    return p


def write_classified(tmp_path: Path, articles: pd.DataFrame) -> Path:
    p = tmp_path / "classified.parquet"
    articles.to_parquet(p, index=False)
    return p


def write_panel(tmp_path: Path, tickers, sessions, source="yfinance") -> Path:
    from app.data.curate import save_curated

    panel_dir = tmp_path / "panel"
    for t in tickers:
        df = pd.DataFrame(
            {
                "date": sessions,
                "ticker": t,
                "open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0,
                "volume": 1e6,
            }
        )
        save_curated(df, panel_dir, t, source=source)
    return panel_dir


def write_sector_map(tmp_path: Path, ticker_sector: dict) -> Path:
    p = tmp_path / "sectors.json"
    p.write_text(json.dumps({
        "source": "test", "snapshot_date": "2024-01-01",
        "classification_basis": "test fixture",
        "count": len(ticker_sector), "sectors": sorted(set(ticker_sector.values())),
        "ticker_sector": ticker_sector,
    }))
    return p


SESSIONS = pd.bdate_range("2021-01-04", periods=40)
SESSION_POS = pd.Series(np.arange(len(SESSIONS)), index=SESSIONS)


# --------------------------------------------------------------------------
# 1. Trading-session / publication-date alignment
# --------------------------------------------------------------------------


def test_article_before_open_buckets_to_same_session():
    """09:00 ET is before the 09:30 open -> same session (if a session)."""
    arts = make_classified_articles([
        {"ticker": "AAA", "published_at": "2021-01-05T14:00:00Z", "category": "OTHER"},  # 09:00 ET
    ])
    from app.nlp.daily_sentiment import assign_trade_date

    assigned, _ = assign_trade_date(arts, SESSIONS)
    assert assigned["trade_date"].iloc[0] == pd.Timestamp("2021-01-05")


def test_article_after_open_buckets_to_next_session():
    """20:00 ET (the dominant placeholder time in this archive) is well
    after the open -> next session, not the same day."""
    arts = make_classified_articles([
        # prev-day midnight UTC = 19:00 ET prior day
        {"ticker": "AAA", "published_at": "2021-01-05T00:00:00Z", "category": "OTHER"},
    ])
    from app.nlp.daily_sentiment import assign_trade_date

    assigned, _ = assign_trade_date(arts, SESSIONS)
    # 2021-01-05T00:00Z -> 2021-01-04 19:00 ET -> after open -> next session
    assert assigned["trade_date"].iloc[0] == pd.Timestamp("2021-01-05")


# --------------------------------------------------------------------------
# 2. Join to the correct ticker-date return record
# --------------------------------------------------------------------------


def test_event_joins_to_the_correct_return_row(tmp_path):
    arts = make_classified_articles([
        {"ticker": "AAA", "published_at": "2021-01-05T14:00:00Z", "category": "EARNINGS_RESULTS"},
    ])
    returns = make_returns(["AAA", "BBB"], SESSIONS, horizons=(1,))
    cp = write_classified(tmp_path, arts)
    rp = write_returns(tmp_path, returns)
    pd_ = write_panel(tmp_path, ["AAA", "BBB"], SESSIONS)
    sp = write_sector_map(tmp_path, {"AAA": "Technology", "BBB": "Technology"})

    events, _ = build_event_table(cp, rp, pd_, sp)
    assert len(events) == 1
    r = events.iloc[0]
    match = (returns.ticker == "AAA") & (returns.date == pd.Timestamp("2021-01-05"))
    expected = returns[match].iloc[0]
    assert r[abnormal_ret_col(1)] == pytest.approx(expected[abnormal_ret_col(1)])


def test_event_with_no_price_bar_is_excluded_and_counted(tmp_path):
    """An event whose trade_date has NO corresponding Phase 0 row must be
    excluded (never silently kept with a NaN return) and counted."""
    arts = make_classified_articles([
        {"ticker": "ZZZ", "published_at": "2021-01-05T14:00:00Z", "category": "OTHER"},
    ])
    returns = make_returns(["AAA"], SESSIONS, horizons=(1,))  # ZZZ absent entirely
    cp = write_classified(tmp_path, arts)
    rp = write_returns(tmp_path, returns)
    pd_ = write_panel(tmp_path, ["AAA", "ZZZ"], SESSIONS)
    sp = write_sector_map(tmp_path, {"AAA": "Technology", "ZZZ": "Technology"})

    events, stats = build_event_table(cp, rp, pd_, sp)
    assert len(events) == 0
    assert stats["events_excluded_no_price_bar_at_event_date"] == 1


def test_valid_columns_are_proper_bool_dtype_after_a_partial_left_join(tmp_path):
    """Regression pin: a LEFT merge against Phase 0 upcasts valid_{h}d to
    object dtype whenever ANY event row fails to match (to hold NaN there).
    Even after those unmatched rows are dropped, the surviving column can
    stay object-dtype holding native Python bool -- which silently breaks
    `~` (bitwise invert on Python bool gives -2/-1 via int semantics, not
    logical negation) wherever a caller inverts the mask, e.g.
    invalidity_reasons(). build_event_table must restore real bool dtype."""
    arts = make_classified_articles([
        {"ticker": "AAA", "published_at": "2021-01-05T14:00:00Z", "category": "EARNINGS_RESULTS"},
        {"ticker": "ZZZ", "published_at": "2021-01-05T14:00:00Z", "category": "EARNINGS_RESULTS"},
    ])
    # ZZZ has NO return row at all -> forces a partially-unmatched left join,
    # the exact condition that upcasts valid_1d to object dtype.
    returns = make_returns(["AAA"], SESSIONS, horizons=(1,))
    cp = write_classified(tmp_path, arts)
    rp = write_returns(tmp_path, returns)
    pd_ = write_panel(tmp_path, ["AAA", "ZZZ"], SESSIONS)
    sp = write_sector_map(tmp_path, {"AAA": "Technology", "ZZZ": "Technology"})

    events, _ = build_event_table(cp, rp, pd_, sp)
    assert len(events) == 1  # only AAA survives; ZZZ excluded as no-price-bar
    assert events[valid_col(1)].dtype == bool
    # the actual failure mode: inverting the mask must behave logically,
    # not as a bitwise invert of a Python bool-as-int.
    inverted = ~events[valid_col(1)]
    assert inverted.dtype == bool
    assert set(inverted.unique()) <= {True, False}


# --------------------------------------------------------------------------
# 3. Duplicate article handling
# --------------------------------------------------------------------------


def test_multiple_articles_same_ticker_date_category_collapse_to_one_event(tmp_path):
    # All three times are before 09:30 ET (12:00/12:15/12:29 UTC = 07:00/
    # 07:15/07:29 ET in January, EST) so all three bucket to the SAME
    # session -- unlike 14:00/15:00/16:00Z, which straddle the market open
    # and would roll the later two to the next session.
    arts = make_classified_articles([
        {"ticker": "AAA", "published_at": "2021-01-05T12:00:00Z", "category": "EARNINGS_RESULTS"},
        {"ticker": "AAA", "published_at": "2021-01-05T12:15:00Z", "category": "EARNINGS_RESULTS"},
        {"ticker": "AAA", "published_at": "2021-01-05T12:29:00Z", "category": "EARNINGS_RESULTS"},
    ])
    returns = make_returns(["AAA"], SESSIONS, horizons=(1,))
    cp = write_classified(tmp_path, arts)
    rp = write_returns(tmp_path, returns)
    pd_ = write_panel(tmp_path, ["AAA"], SESSIONS)
    sp = write_sector_map(tmp_path, {"AAA": "Technology"})

    events, _ = build_event_table(cp, rp, pd_, sp)
    assert len(events) == 1, "three same-category articles on the same ticker-day -> one event"
    assert events.iloc[0]["n_articles"] == 3


def test_different_categories_same_ticker_date_stay_separate(tmp_path):
    # Both times are before 09:30 ET (09:00 and 09:15 ET) so both land on
    # the SAME trade_date -- the point of the test is isolating "same date,
    # different category", not an accidental date difference.
    arts = make_classified_articles([
        {"ticker": "AAA", "published_at": "2021-01-05T14:00:00Z", "category": "EARNINGS_RESULTS"},
        {"ticker": "AAA", "published_at": "2021-01-05T14:15:00Z", "category": "MANAGEMENT"},
    ])
    returns = make_returns(["AAA"], SESSIONS, horizons=(1,))
    cp = write_classified(tmp_path, arts)
    rp = write_returns(tmp_path, returns)
    pd_ = write_panel(tmp_path, ["AAA"], SESSIONS)
    sp = write_sector_map(tmp_path, {"AAA": "Technology"})

    events, _ = build_event_table(cp, rp, pd_, sp)
    assert len(events) == 2
    assert set(events["category"]) == {"EARNINGS_RESULTS", "MANAGEMENT"}
    assert (events["n_articles"] == 1).all()


# --------------------------------------------------------------------------
# 4. Category and horizon grouping, plus correct mean/median/counts
# --------------------------------------------------------------------------


def test_category_horizon_summary_mean_median_and_counts_are_exact(tmp_path):
    tickers = ["AAA", "BBB", "CCC"]
    arts = make_classified_articles([
        {"ticker": t, "published_at": "2021-01-05T14:00:00Z", "category": "EARNINGS_RESULTS"}
        for t in tickers
    ])
    returns = make_returns(tickers, SESSIONS, horizons=(1,))
    cp = write_classified(tmp_path, arts)
    rp = write_returns(tmp_path, returns)
    pd_ = write_panel(tmp_path, tickers, SESSIONS)
    sp = write_sector_map(tmp_path, {t: "Technology" for t in tickers})

    events, _ = build_event_table(cp, rp, pd_, sp)
    summary = category_horizon_summary(events, SESSION_POS, horizons=(1,))
    row = summary[(summary.category == "EARNINGS_RESULTS") & (summary.horizon == 1)].iloc[0]

    expected_vals = []
    for t in tickers:
        er = returns[(returns.ticker == t) & (returns.date == pd.Timestamp("2021-01-05"))]
        expected_vals.append(er[abnormal_ret_col(1)].iloc[0])
    expected_vals = np.array(expected_vals)

    assert row["n_events_valid"] == 3
    assert row["n_tickers"] == 3
    assert row["mean_abnormal_return"] == pytest.approx(expected_vals.mean())
    assert row["median_abnormal_return"] == pytest.approx(np.median(expected_vals))


# --------------------------------------------------------------------------
# 5. Missing or invalid return windows
# --------------------------------------------------------------------------


def test_invalid_horizon_is_excluded_from_that_horizons_stats_not_substituted(tmp_path):
    tickers = ["AAA", "BBB"]
    arts = make_classified_articles([
        {"ticker": t, "published_at": "2021-01-05T14:00:00Z", "category": "EARNINGS_RESULTS"}
        for t in tickers
    ])
    returns = make_returns(tickers, SESSIONS, horizons=(1,))
    # mark AAA's 1d window invalid on this date -- must drop out of that
    # horizon's stats, and its abnormal return must NOT leak in as a valid
    # number (the "never silently substitute" requirement).
    mask = (returns.ticker == "AAA") & (returns.date == pd.Timestamp("2021-01-05"))
    returns.loc[mask, valid_col(1)] = False
    returns.loc[mask, abnormal_ret_col(1)] = np.nan
    returns.loc[mask, stock_ret_col(1)] = np.nan
    returns.loc[mask, bench_ret_col(1)] = np.nan

    cp = write_classified(tmp_path, arts)
    rp = write_returns(tmp_path, returns)
    pd_ = write_panel(tmp_path, tickers, SESSIONS)
    sp = write_sector_map(tmp_path, {t: "Technology" for t in tickers})

    events, _ = build_event_table(cp, rp, pd_, sp)
    summary = category_horizon_summary(events, SESSION_POS, horizons=(1,))
    row = summary[(summary.category == "EARNINGS_RESULTS") & (summary.horizon == 1)].iloc[0]
    assert row["n_events_total"] == 2
    assert row["n_events_valid"] == 1, "the invalid AAA row must not count as valid"
    assert row["n_events_invalid"] == 1


# --------------------------------------------------------------------------
# Newey-West / BH already covered in a focused smoke test; here, integration
# --------------------------------------------------------------------------


def test_nw_ci_widens_with_lag_on_correlated_series():
    rng = np.random.default_rng(1)
    ar = np.zeros(300)
    for i in range(1, 300):
        ar[i] = 0.7 * ar[i - 1] + rng.normal()
    lag0 = newey_west_mean_ci(ar, lag=0)
    lag10 = newey_west_mean_ci(ar, lag=10)
    assert lag10["se"] > lag0["se"]


# --------------------------------------------------------------------------
# Irregular-spacing HAC (audit fix): lag must track real session distance,
# not array position, for sparse event-date series.
# --------------------------------------------------------------------------


def test_irregular_matches_regular_on_densely_consecutive_positions():
    """When every observation sits at consecutive integer positions (the
    dense case, e.g. EARNINGS_RESULTS overall), array-position lag and
    real-session-position lag coincide -- the irregular estimator must
    agree with the plain one exactly."""
    rng = np.random.default_rng(2)
    ar = np.zeros(200)
    for i in range(1, 200):
        ar[i] = 0.6 * ar[i - 1] + rng.normal()
    positions = np.arange(200)  # fully dense: no gaps
    plain = newey_west_mean_ci(ar, lag=10)
    irregular = newey_west_mean_ci_irregular(ar, positions, lag=10)
    assert irregular["mean"] == pytest.approx(plain["mean"])
    assert irregular["se"] == pytest.approx(plain["se"], rel=1e-9)


def test_irregular_finds_no_dependence_when_gaps_exceed_the_lag_window():
    """This is the actual bug being fixed: observations array-adjacent but
    far apart in REAL session terms (gap > lag) must be treated as having
    NO captured serial dependence between them -- se should reduce to the
    plain (gamma0-only) estimate, not be inflated by phantom "adjacency"."""
    rng = np.random.default_rng(3)
    n = 100
    values = rng.normal(0, 1, n)
    # observations 20 real sessions apart -- far beyond a lag-5 window
    positions = np.arange(n) * 20
    iid_only = newey_west_mean_ci_irregular(values, positions, lag=0)
    with_lag = newey_west_mean_ci_irregular(values, positions, lag=5)
    assert with_lag["se"] == pytest.approx(iid_only["se"], rel=1e-9)


def test_irregular_detects_exact_known_lag_dependence():
    """Construct a series with dependence EXACTLY at real-session-gap 7
    (not array position 7) and confirm the irregular estimator's variance
    picks it up while a naive array-position lag of the same NUMBER would
    not, when the array itself is sparse."""
    rng = np.random.default_rng(4)
    n = 150
    base = rng.normal(0, 1, n)
    # positions irregular: random increasing gaps of 1-15 sessions
    gaps = rng.integers(1, 16, n - 1)
    positions = np.concatenate([[0], np.cumsum(gaps)])
    # inject real dependence: value at position p correlates with value at
    # position p-7, when such a position exists in our sparse sample
    pos_to_idx = {int(p): i for i, p in enumerate(positions)}
    values = base.copy()
    for i, p in enumerate(positions):
        j = pos_to_idx.get(int(p) - 7)
        if j is not None:
            values[i] += 0.8 * base[j]
    no_lag = newey_west_mean_ci_irregular(values, positions, lag=0)
    with_lag7 = newey_west_mean_ci_irregular(values, positions, lag=7)
    assert with_lag7["se"] != pytest.approx(no_lag["se"], rel=1e-6)


def test_session_positions_maps_dates_via_the_lookup():
    dates = pd.DatetimeIndex([SESSIONS[0], SESSIONS[5], SESSIONS[10]])
    pos = session_positions(dates, SESSION_POS)
    assert list(pos) == [0, 5, 10]


def test_bh_correction_is_less_permissive_than_uncorrected():
    pvals = [0.04, 0.03, 0.02, 0.5, 0.6, 0.7, 0.8, 0.9]
    flags = benjamini_hochberg(pvals, q=0.10)
    raw_sig = [p < 0.05 for p in pvals]
    assert sum(flags) <= sum(raw_sig)


def test_bh_flags_a_p_value_that_underflows_to_exact_zero():
    """Regression pin (found via a Phase 2 audit): when the single smallest
    p-value in a family underflows to exactly 0.0 -- routine with large
    sample sizes and a strong effect -- it must still be flagged
    significant. An earlier version used a float `thresh=0.0` default that
    was indistinguishable from "no threshold found", silently excluding
    the most significant result in the family."""
    pvals = [0.0, 0.5, 0.6, 0.7, 0.8]
    flags = benjamini_hochberg(pvals, q=0.10)
    assert flags[0], "a p-value of exactly 0.0 must be flagged significant"
    assert not any(flags[1:])


def test_bh_returns_all_false_when_truly_nothing_survives():
    """The no-threshold-found case (every p-value fails its step-up
    criterion) must still yield no flags -- distinct from the p=0.0 case
    above, which is why `thresh` is a sentinel rather than a float."""
    pvals = [0.5, 0.6, 0.7, 0.8, 0.9]
    flags = benjamini_hochberg(pvals, q=0.10)
    assert not any(flags)


def test_insufficient_sample_is_flagged_not_forced(tmp_path):
    """A single event is nowhere near MIN_EVENTS/MIN_TICKERS -- must be
    flagged insufficient, not reported as if reliable."""
    arts = make_classified_articles([
        {"ticker": "AAA", "published_at": "2021-01-05T14:00:00Z", "category": "M_AND_A"},
    ])
    returns = make_returns(["AAA", "BBB"], SESSIONS, horizons=(1,))
    cp = write_classified(tmp_path, arts)
    rp = write_returns(tmp_path, returns)
    pd_ = write_panel(tmp_path, ["AAA", "BBB"], SESSIONS)
    sp = write_sector_map(tmp_path, {"AAA": "Technology", "BBB": "Technology"})

    events, _ = build_event_table(cp, rp, pd_, sp)
    summary = category_horizon_summary(events, SESSION_POS, horizons=(1,))
    row = summary[summary.category == "M_AND_A"].iloc[0]
    assert not row["sufficient_sample"]
    assert row["n_events_valid"] == 1 < MIN_EVENTS


# --------------------------------------------------------------------------
# Sector grouping
# --------------------------------------------------------------------------


def test_sector_grouping_only_includes_known_sector_tickers(tmp_path):
    arts = make_classified_articles([
        {"ticker": "AAA", "published_at": "2021-01-05T14:00:00Z", "category": "EARNINGS_RESULTS"},
        {"ticker": "ZZZ", "published_at": "2021-01-05T14:00:00Z", "category": "EARNINGS_RESULTS"},
    ])
    returns = make_returns(["AAA", "ZZZ"], SESSIONS, horizons=(1,))
    cp = write_classified(tmp_path, arts)
    rp = write_returns(tmp_path, returns)
    pd_ = write_panel(tmp_path, ["AAA", "ZZZ"], SESSIONS)
    sp = write_sector_map(tmp_path, {"AAA": "Technology"})  # ZZZ deliberately absent

    events, stats = build_event_table(cp, rp, pd_, sp)
    assert stats["events_with_unknown_sector"] == 1
    # ZZZ's event stays in the event table (sector is a lookup, not a
    # filter) but carries no sector, and is excluded from the sector table.
    zzz_row = events[events["ticker"] == "ZZZ"].iloc[0]
    assert pd.isna(zzz_row["sector"])
    sec_summary = sector_category_horizon_summary(events, SESSION_POS, horizons=(1,))
    assert set(sec_summary["sector"]) == {"Technology"}


# --------------------------------------------------------------------------
# Comparison vs OTHER requires minimum overlapping dates
# --------------------------------------------------------------------------


def test_vs_other_requires_minimum_overlap(tmp_path):
    tickers = [f"T{i}" for i in range(MIN_TICKERS)]
    arts = [
        {"ticker": t, "published_at": "2021-01-05T14:00:00Z", "category": "EARNINGS_RESULTS"}
        for t in tickers
    ] * (MIN_EVENTS // MIN_TICKERS + 1)
    # not enough DISTINCT overlapping dates even if event count is high
    arts = make_classified_articles(arts[: MIN_EVENTS + 5])
    returns = make_returns(tickers, SESSIONS, horizons=(1,))
    cp = write_classified(tmp_path, arts)
    rp = write_returns(tmp_path, returns)
    pd_ = write_panel(tmp_path, tickers, SESSIONS)
    sp = write_sector_map(tmp_path, {t: "Technology" for t in tickers})

    events, _ = build_event_table(cp, rp, pd_, sp)
    cmp = category_vs_other(events, SESSION_POS, horizons=(1,))
    row = cmp[(cmp.category == "EARNINGS_RESULTS") & (cmp.horizon == 1)].iloc[0]
    assert row["n_overlapping_dates"] < MIN_OVERLAP_DATES
    assert not row["sufficient_sample"]


# --------------------------------------------------------------------------
# Reproducibility / non-interference
# --------------------------------------------------------------------------


@pytest.mark.skipif(not MATRIX_H5.exists(), reason="matrix_h5_v4 absent")
def test_does_not_modify_matrix_h5():
    before = MATRIX_H5.stat().st_mtime, MATRIX_H5.stat().st_size
    import app.events.category_horizon_analysis  # noqa: F401

    after = MATRIX_H5.stat().st_mtime, MATRIX_H5.stat().st_size
    assert before == after


@pytest.mark.skipif(not PHASE0_OUTPUT.exists(), reason="Phase 0 output absent")
def test_does_not_modify_phase0_output():
    before = PHASE0_OUTPUT.stat().st_mtime, PHASE0_OUTPUT.stat().st_size
    import app.events.category_horizon_analysis  # noqa: F401

    after = PHASE0_OUTPUT.stat().st_mtime, PHASE0_OUTPUT.stat().st_size
    assert before == after


def test_build_event_table_is_deterministic(tmp_path):
    arts = make_classified_articles([
        {"ticker": "AAA", "published_at": "2021-01-05T14:00:00Z", "category": "EARNINGS_RESULTS"},
        {"ticker": "AAA", "published_at": "2021-01-06T14:00:00Z", "category": "OTHER"},
    ])
    returns = make_returns(["AAA"], SESSIONS, horizons=(1,))
    cp = write_classified(tmp_path, arts)
    rp = write_returns(tmp_path, returns)
    pd_ = write_panel(tmp_path, ["AAA"], SESSIONS)
    sp = write_sector_map(tmp_path, {"AAA": "Technology"})

    e1, _ = build_event_table(cp, rp, pd_, sp)
    e2, _ = build_event_table(cp, rp, pd_, sp)
    pd.testing.assert_frame_equal(e1.reset_index(drop=True), e2.reset_index(drop=True))


# --------------------------------------------------------------------------
# Full pipeline, synthetic end-to-end
# --------------------------------------------------------------------------


def test_run_end_to_end_writes_all_outputs(tmp_path):
    tickers = ["AAA", "BBB", "CCC", "DDD", "EEE", "FFF"]
    rows = []
    for t in tickers:
        for d in SESSIONS[:20]:
            rows.append({
                "ticker": t, "published_at": d.strftime("%Y-%m-%dT14:00:00Z"),
                "category": "OTHER",
            })
    for t in tickers[:5]:
        for d in SESSIONS[:20:2]:
            rows.append({
                "ticker": t, "published_at": d.strftime("%Y-%m-%dT14:00:00Z"),
                "category": "EARNINGS_RESULTS",
            })
    arts = make_classified_articles(rows)
    returns = make_returns(tickers, SESSIONS, horizons=(1, 5))
    cp = write_classified(tmp_path, arts)
    rp = write_returns(tmp_path, returns)
    pd_ = write_panel(tmp_path, tickers, SESSIONS)
    sp = write_sector_map(tmp_path, {t: "Technology" for t in tickers})
    out_dir = tmp_path / "out"

    summary = run(cp, rp, pd_, sp, out_dir, version="test", horizons=(1, 5))
    for key in (
        "events", "category_horizon_summary", "category_vs_other",
        "sector_category_horizon_summary", "sector_category_vs_other",
    ):
        assert Path(summary["outputs"][key]).exists()
    assert Path(summary["coverage_path"]).exists()
    assert Path(summary["report_path"]).exists()
    report_raw = Path(summary["report_path"]).read_text()
    report_text = report_raw.lower()
    assert "EARNINGS_RESULTS" in report_raw
    assert "not evidence" in report_text and "caused" in report_text, (
        "report must carry the association-not-causation caveat"
    )


# --------------------------------------------------------------------------
# Against the real built artifacts (skip if not yet generated)
# --------------------------------------------------------------------------

REAL = pytest.mark.skipif(not REAL_EVENTS.exists(), reason="Phase 1 output not built")


@REAL
def test_real_events_have_no_duplicate_ticker_date_category():
    events = pd.read_parquet(REAL_EVENTS)
    dupes = events.duplicated(subset=["ticker", "date", "category"]).sum()
    assert dupes == 0


@REAL
def test_real_category_vs_other_excludes_other_itself():
    cmp = pd.read_parquet(REAL_CAT_VS_OTHER)
    assert "OTHER" not in set(cmp["category"])


@REAL
def test_real_coverage_json_is_well_formed():
    cov = json.loads(REAL_COVERAGE.read_text())
    assert cov["build_stats"]["events_final"] > 0
    assert "inference_method" in cov
