"""Phase 2: stock-specific news-category sensitivity.

Synthetic fixtures keep these fast; tests against the real built artifacts
skip when they have not been generated.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from app.events.benchmark_returns import abnormal_ret_col, valid_col
from app.events.stock_sensitivity import (
    SHRINKAGE_MIN_TICKERS,
    STOCK_EXPLORATORY_MIN_EVENTS,
    STOCK_MIN_EVENTS,
    STOCK_MIN_OTHER_DATES,
    _der_simonian_laird_tau2,
    _evidence_tier,
    _stock_compare_vs_other,
    attach_sector_baseline,
    rank_stock_sensitivities,
    run,
    shrink_category_horizon,
    stock_category_horizon_summary,
    stock_vs_other,
)

REAL_SUMMARY = Path("data/curated/phase2_stock_category_horizon_summary_v1.parquet")
REAL_VS_OTHER = Path("data/curated/phase2_stock_vs_other_v1.parquet")
REAL_RANKING = Path("data/curated/phase2_stock_ranking_v1.parquet")
PHASE1_EVENTS = Path("data/curated/phase1_events_v1.parquet")
MATRIX_H5 = Path("data/curated/matrix_h5_v4.parquet")
PHASE0_OUTPUT = Path("data/curated/benchmark_adjusted_returns_v1.parquet")

SESSIONS = pd.bdate_range("2015-01-02", periods=600)
SESSION_POS = pd.Series(np.arange(len(SESSIONS)), index=SESSIONS)


# --------------------------------------------------------------------------
# Synthetic event-table fixtures
# --------------------------------------------------------------------------


def make_events(rows: list[dict], horizon: int = 1) -> pd.DataFrame:
    """Each row: ticker, date, category, abnormal_ret (float), valid (bool,
    default True). Produces a phase1_events_v1-shaped frame for one horizon."""
    ret_c, valid_c = abnormal_ret_col(horizon), valid_col(horizon)
    out = []
    for r in rows:
        out.append({
            "ticker": r["ticker"], "date": pd.Timestamp(r["date"]), "category": r["category"],
            "n_articles": r.get("n_articles", 1), "source": "yfinance", "sector": r.get("sector"),
            ret_c: r["abnormal_ret"], valid_c: r.get("valid", True),
        })
    return pd.DataFrame(out)


def dense_category_events(ticker: str, category: str, n: int, mean: float, seed: int, horizon=1):
    """n events of one category for one ticker, on consecutive sessions
    starting at index 0, with returns drawn from N(mean, 0.01)."""
    rng = np.random.default_rng(seed)
    dates = SESSIONS[:n]
    vals = rng.normal(mean, 0.01, n)
    rows = [
        {"ticker": ticker, "date": d, "category": category, "abnormal_ret": v}
        for d, v in zip(dates, vals, strict=True)
    ]
    return make_events(rows, horizon=horizon)


# --------------------------------------------------------------------------
# 1. Calculations: mean/median/percentiles via stock_category_horizon_summary
# --------------------------------------------------------------------------


def test_stock_summary_mean_matches_hand_computed_mean():
    vals = [0.01, 0.02, -0.01, 0.03, 0.0]
    events = make_events([
        {"ticker": "AAA", "date": SESSIONS[i], "category": "EARNINGS_RESULTS", "abnormal_ret": v}
        for i, v in enumerate(vals)
    ])
    summary = stock_category_horizon_summary(events, SESSION_POS, horizons=(1,))
    row = summary.iloc[0]
    assert row["n_events_valid"] == 5
    assert row["mean_abnormal_return"] == pytest.approx(np.mean(vals))
    assert row["median_abnormal_return"] == pytest.approx(np.median(vals))
    assert row["p25_abnormal_return"] == pytest.approx(np.quantile(vals, 0.25))
    assert row["p75_abnormal_return"] == pytest.approx(np.quantile(vals, 0.75))
    assert row["pct_positive"] == pytest.approx(3 / 5)


def test_evidence_tier_boundaries_are_exact():
    assert _evidence_tier(STOCK_MIN_EVENTS) == "adequately_supported"
    assert _evidence_tier(STOCK_MIN_EVENTS - 1) == "exploratory"
    assert _evidence_tier(STOCK_EXPLORATORY_MIN_EVENTS) == "exploratory"
    assert _evidence_tier(STOCK_EXPLORATORY_MIN_EVENTS - 1) == "insufficient"
    assert _evidence_tier(0) == "insufficient"


# --------------------------------------------------------------------------
# 2. Comparison baseline: the actual bug this module was built to avoid
# --------------------------------------------------------------------------


def test_stock_comparison_is_not_the_sector_style_paired_date_difference():
    """Regression pin for the real bug found building this module: a
    same-date paired difference (Phase 1's sector-level device) is
    mathematically guaranteed to be exactly zero for a single ticker
    whenever a date carries both a category tag and an OTHER tag, because
    both rows share the SAME underlying (ticker, date) return. The
    two-sample comparison must NOT reduce to that degenerate zero."""
    # AAA has EARNINGS_RESULTS on some dates and OTHER on the SAME dates
    # (deliberately overlapping) plus some OTHER-only dates with a clearly
    # different typical return.
    shared_dates = SESSIONS[:30]
    other_only_dates = SESSIONS[30:80]
    rng = np.random.default_rng(0)
    rows = []
    for d in shared_dates:
        r = rng.normal(0.05, 0.001)  # big, consistent positive return
        rows.append({"ticker": "AAA", "date": d, "category": "EARNINGS_RESULTS", "abnormal_ret": r})
        rows.append({"ticker": "AAA", "date": d, "category": "OTHER", "abnormal_ret": r})
    for d in other_only_dates:
        rows.append({
            "ticker": "AAA", "date": d, "category": "OTHER",
            "abnormal_ret": rng.normal(0.0, 0.001),
        })
    events = make_events(rows)
    cat_events = events[events.category == "EARNINGS_RESULTS"]
    other_events = events[events.category == "OTHER"]
    result = _stock_compare_vs_other(cat_events, other_events, 1, SESSION_POS)
    assert result["diff_mean"] is not None
    assert result["diff_mean"] != 0.0
    # EARNINGS_RESULTS dates average ~0.05, OTHER-only dates average ~0.0
    assert result["diff_mean"] == pytest.approx(0.05, abs=0.01)


def test_stock_comparison_excludes_same_day_overlap_from_other_side():
    shared = SESSIONS[:10]
    other_only = SESSIONS[10:40]
    rows = [
        {"ticker": "AAA", "date": d, "category": "EARNINGS_RESULTS", "abnormal_ret": 0.1}
        for d in shared
    ] + [
        {"ticker": "AAA", "date": d, "category": "OTHER", "abnormal_ret": 0.1} for d in shared
    ] + [
        {"ticker": "AAA", "date": d, "category": "OTHER", "abnormal_ret": 0.0} for d in other_only
    ]
    events = make_events(rows)
    cat_events = events[events.category == "EARNINGS_RESULTS"]
    other_events = events[events.category == "OTHER"]
    result = _stock_compare_vs_other(cat_events, other_events, 1, SESSION_POS)
    assert result["n_other_dates"] == 30  # the 10 shared dates excluded from OTHER
    assert result["n_same_day_overlap_excluded"] == 10


def test_sufficient_sample_requires_both_event_count_and_other_dates():
    few_events = dense_category_events("AAA", "M_AND_A", STOCK_MIN_EVENTS, 0.01, seed=1)
    plenty_other = dense_category_events("AAA", "OTHER", 200, 0.0, seed=2)
    events = pd.concat([few_events, plenty_other], ignore_index=True)
    result = _stock_compare_vs_other(
        events[events.category == "M_AND_A"], events[events.category == "OTHER"], 1, SESSION_POS
    )
    assert result["sufficient_sample"]

    thin_other = dense_category_events("AAA", "OTHER", STOCK_MIN_OTHER_DATES - 1, 0.0, seed=3)
    events2 = pd.concat([few_events, thin_other], ignore_index=True)
    result2 = _stock_compare_vs_other(
        events2[events2.category == "M_AND_A"], events2[events2.category == "OTHER"], 1, SESSION_POS
    )
    assert not result2["sufficient_sample"]


# --------------------------------------------------------------------------
# 3. Event counts (dedup convention preserved from Phase 1)
# --------------------------------------------------------------------------


def test_event_counts_do_not_double_count_same_day_multiple_articles():
    """phase1_events_v1 already collapses same-day same-category articles
    into one row with n_articles as a count; this module must not re-split
    or re-inflate that -- n_events_valid should equal the number of EVENT
    ROWS, not n_articles summed."""
    events = make_events([
        {"ticker": "AAA", "date": SESSIONS[0], "category": "EARNINGS_RESULTS",
         "abnormal_ret": 0.02, "n_articles": 5},
        {"ticker": "AAA", "date": SESSIONS[1], "category": "EARNINGS_RESULTS",
         "abnormal_ret": 0.01, "n_articles": 3},
    ])
    summary = stock_category_horizon_summary(events, SESSION_POS, horizons=(1,))
    assert summary.iloc[0]["n_events_valid"] == 2  # not 8 (5+3 articles)


# --------------------------------------------------------------------------
# 4. Sparse samples: tiers and reporting
# --------------------------------------------------------------------------


def test_sparse_category_is_labelled_exploratory_not_reliable():
    events = dense_category_events("AAA", "M_AND_A", STOCK_EXPLORATORY_MIN_EVENTS, 0.05, seed=4)
    summary = stock_category_horizon_summary(events, SESSION_POS, horizons=(1,))
    row = summary.iloc[0]
    assert row["evidence_tier"] == "exploratory"
    assert row["mean_abnormal_return"] is not None  # still shown, just labelled


def test_near_empty_category_is_insufficient():
    events = dense_category_events("AAA", "M_AND_A", 2, 0.05, seed=5)
    summary = stock_category_horizon_summary(events, SESSION_POS, horizons=(1,))
    assert summary.iloc[0]["evidence_tier"] == "insufficient"


def test_a_few_extreme_events_do_not_reach_adequately_supported():
    """A handful of huge outcomes must not look reliable just because the
    raw mean is large -- the evidence tier is driven by event count, not
    effect size."""
    events = make_events([
        {"ticker": "AAA", "date": SESSIONS[0], "category": "M_AND_A", "abnormal_ret": 0.50},
        {"ticker": "AAA", "date": SESSIONS[1], "category": "M_AND_A", "abnormal_ret": 0.45},
        {"ticker": "AAA", "date": SESSIONS[2], "category": "M_AND_A", "abnormal_ret": 0.60},
    ])
    summary = stock_category_horizon_summary(events, SESSION_POS, horizons=(1,))
    row = summary.iloc[0]
    assert row["mean_abnormal_return"] > 0.4  # a big raw number
    assert row["evidence_tier"] != "adequately_supported"  # but not trusted


# --------------------------------------------------------------------------
# Shrinkage
# --------------------------------------------------------------------------


def test_der_simonian_laird_tau2_is_nonnegative():
    thetas = np.array([0.01, 0.02, -0.01, 0.015, 0.005])
    ses = np.array([0.005, 0.004, 0.006, 0.003, 0.004])
    tau2 = _der_simonian_laird_tau2(thetas, ses)
    assert tau2 >= 0.0


def test_der_simonian_laird_tau2_is_zero_for_identical_inputs():
    """If all tickers report the exact same effect, there is no between-
    ticker variance to explain -- tau2 must be (clamped to) zero."""
    thetas = np.full(10, 0.02)
    ses = np.full(10, 0.01)
    tau2 = _der_simonian_laird_tau2(thetas, ses)
    assert tau2 == pytest.approx(0.0, abs=1e-12)


def test_shrinkage_pulls_noisy_estimate_toward_prior():
    """A ticker with few (but above-floor) events and a wildly different
    raw mean from the rest should shrink noticeably toward the category
    prior; the point of the method is exactly this."""
    rng = np.random.default_rng(6)
    rows = []
    tickers = [f"T{i}" for i in range(SHRINKAGE_MIN_TICKERS + 5)]
    for t in tickers[:-1]:
        n = 100
        vals = rng.normal(0.01, 0.01, n)
        rows.append(make_events([
            {"ticker": t, "date": SESSIONS[i], "category": "EARNINGS_RESULTS", "abnormal_ret": v}
            for i, v in enumerate(vals)
        ]))
    # one outlier ticker: few events, huge mean
    outlier_vals = rng.normal(0.5, 0.01, STOCK_MIN_EVENTS)
    outlier_rows = [
        {"ticker": tickers[-1], "date": SESSIONS[i], "category": "EARNINGS_RESULTS",
         "abnormal_ret": v}
        for i, v in enumerate(outlier_vals)
    ]
    rows.append(make_events(outlier_rows))
    events = pd.concat(rows, ignore_index=True)
    summary = stock_category_horizon_summary(events, SESSION_POS, horizons=(1,))

    category_summary = pd.DataFrame([
        {"category": "EARNINGS_RESULTS", "horizon": 1, "mean_abnormal_return": 0.01}
    ])
    shrunk = shrink_category_horizon(summary, category_summary)
    outlier_row = shrunk[shrunk.ticker == tickers[-1]].iloc[0]
    assert outlier_row["shrunk_mean"] < outlier_row["mean_abnormal_return"]
    assert outlier_row["shrunk_mean"] > 0.01  # pulled toward, not all the way to, the prior


def test_shrinkage_skipped_when_too_few_tickers_sufficiently_sampled():
    events = dense_category_events("AAA", "M_AND_A", STOCK_MIN_EVENTS, 0.05, seed=7)
    summary = stock_category_horizon_summary(events, SESSION_POS, horizons=(1,))
    category_summary = pd.DataFrame([
        {"category": "M_AND_A", "horizon": 1, "mean_abnormal_return": 0.0}
    ])
    shrunk = shrink_category_horizon(summary, category_summary)
    row = shrunk.iloc[0]
    assert pd.isna(row["tau2"])
    assert row["shrunk_mean"] == pytest.approx(row["mean_abnormal_return"])


# --------------------------------------------------------------------------
# Multiple-testing correction
# --------------------------------------------------------------------------


def test_fdr_correction_is_applied_within_category_horizon_family_only():
    """Two different (category, horizon) families must be corrected
    SEPARATELY -- a significant result in one family must not change
    whether a borderline result in a different family survives."""
    rng = np.random.default_rng(8)
    rows = []
    for i in range(30):
        t = f"T{i}"
        mean = 0.05 if i < 3 else 0.0  # 3 genuinely strong tickers
        vals = rng.normal(mean, 0.01, 100)
        rows.append(make_events([
            {"ticker": t, "date": SESSIONS[j], "category": "M_AND_A", "abnormal_ret": v}
            for j, v in enumerate(vals)
        ]))
        other_vals = rng.normal(0.0, 0.01, 200)
        rows.append(make_events([
            {"ticker": t, "date": SESSIONS[100 + j], "category": "OTHER", "abnormal_ret": v}
            for j, v in enumerate(other_vals)
        ]))
    events = pd.concat(rows, ignore_index=True)
    vs_other = stock_vs_other(events, SESSION_POS, horizons=(1,))
    family = vs_other[(vs_other.category == "M_AND_A") & (vs_other.horizon == 1)]
    assert family["n_tests_in_family"].iloc[0] <= 30
    # at least the strongest tickers should be flagged
    assert family[family.significant_fdr]["ticker"].isin(["T0", "T1", "T2"]).any()


def test_insufficient_samples_never_get_a_pvalue_fed_to_fdr():
    events = dense_category_events("AAA", "M_AND_A", 3, 0.5, seed=9)
    other = dense_category_events("AAA", "OTHER", 50, 0.0, seed=10)
    all_events = pd.concat([events, other], ignore_index=True)
    vs_other = stock_vs_other(all_events, SESSION_POS, horizons=(1,))
    row = vs_other.iloc[0]
    assert not row["sufficient_sample"]
    assert not row["significant_fdr"]


# --------------------------------------------------------------------------
# Ranking: never by raw mean alone
# --------------------------------------------------------------------------


def test_ranking_excludes_insufficient_and_nonsignificant_cells():
    events = dense_category_events("AAA", "M_AND_A", 3, 0.9, seed=11)  # huge mean, tiny n
    other = dense_category_events("AAA", "OTHER", 300, 0.0, seed=12)
    cat_summary = pd.DataFrame([{"category": "M_AND_A", "horizon": 1, "mean_abnormal_return": 0.0}])
    all_events = pd.concat([events, other], ignore_index=True)
    summary = stock_category_horizon_summary(all_events, SESSION_POS, horizons=(1,))
    shrunk = shrink_category_horizon(summary, cat_summary)
    vs_other = stock_vs_other(all_events, SESSION_POS, horizons=(1,))
    ranking = rank_stock_sensitivities(shrunk, vs_other)
    assert len(ranking) == 0  # 3 events is far below STOCK_MIN_EVENTS; must not rank


# --------------------------------------------------------------------------
# Reproducibility / non-interference
# --------------------------------------------------------------------------


def test_stock_category_horizon_summary_is_deterministic():
    events = dense_category_events("AAA", "EARNINGS_RESULTS", 50, 0.01, seed=13)
    s1 = stock_category_horizon_summary(events, SESSION_POS, horizons=(1,))
    s2 = stock_category_horizon_summary(events, SESSION_POS, horizons=(1,))
    pd.testing.assert_frame_equal(s1.reset_index(drop=True), s2.reset_index(drop=True))


def test_sector_baseline_attached_only_when_sector_summary_sufficient():
    stock_summary = pd.DataFrame([
        {"ticker": "AAA", "category": "EARNINGS_RESULTS", "horizon": 1},
        {"ticker": "BBB", "category": "EARNINGS_RESULTS", "horizon": 1},
    ])
    sector_summary = pd.DataFrame([
        {
            "sector": "Tech", "category": "EARNINGS_RESULTS", "horizon": 1,
            "mean_abnormal_return": 0.02, "nw_mean": 0.02, "nw_se": 0.001,
            "n_tickers": 40, "sufficient_sample": True,
        },
        {
            "sector": "Energy", "category": "EARNINGS_RESULTS", "horizon": 1,
            "mean_abnormal_return": 0.03, "nw_mean": 0.03, "nw_se": 0.01,
            "n_tickers": 2, "sufficient_sample": False,
        },
    ])
    sector_by_ticker = pd.Series({"AAA": "Tech", "BBB": "Energy"})
    out = attach_sector_baseline(stock_summary, sector_summary, sector_by_ticker)
    aaa = out[out["ticker"] == "AAA"].iloc[0]
    bbb = out[out["ticker"] == "BBB"].iloc[0]
    assert aaa["sector"] == "Tech"
    assert aaa["sector_nw_mean"] == pytest.approx(0.02)
    assert bbb["sector"] == "Energy"
    assert pd.isna(bbb["sector_nw_mean"])


def test_sector_baseline_is_nan_for_unknown_sector():
    stock_summary = pd.DataFrame([{"ticker": "ZZZ", "category": "EARNINGS_RESULTS", "horizon": 1}])
    sector_summary = pd.DataFrame([
        {
            "sector": "Tech", "category": "EARNINGS_RESULTS", "horizon": 1,
            "mean_abnormal_return": 0.02, "nw_mean": 0.02, "nw_se": 0.001,
            "n_tickers": 40, "sufficient_sample": True,
        },
    ])
    # sector_series always has an entry for every known ticker, possibly None --
    # it never omits a key the way an unrelated mapping might.
    sector_by_ticker = pd.Series({"ZZZ": None}, dtype=object)
    out = attach_sector_baseline(stock_summary, sector_summary, sector_by_ticker)
    assert pd.isna(out.iloc[0]["sector"])
    assert pd.isna(out.iloc[0]["sector_nw_mean"])


@pytest.mark.skipif(not MATRIX_H5.exists(), reason="matrix_h5_v4 absent")
def test_does_not_modify_matrix_h5():
    before = MATRIX_H5.stat().st_mtime, MATRIX_H5.stat().st_size
    import app.events.stock_sensitivity  # noqa: F401

    after = MATRIX_H5.stat().st_mtime, MATRIX_H5.stat().st_size
    assert before == after


@pytest.mark.skipif(not PHASE0_OUTPUT.exists(), reason="Phase 0 output absent")
def test_does_not_modify_phase0_output():
    before = PHASE0_OUTPUT.stat().st_mtime, PHASE0_OUTPUT.stat().st_size
    import app.events.stock_sensitivity  # noqa: F401

    after = PHASE0_OUTPUT.stat().st_mtime, PHASE0_OUTPUT.stat().st_size
    assert before == after


@pytest.mark.skipif(not PHASE1_EVENTS.exists(), reason="Phase 1 events absent")
def test_does_not_modify_phase1_events():
    before = PHASE1_EVENTS.stat().st_mtime, PHASE1_EVENTS.stat().st_size
    import app.events.stock_sensitivity  # noqa: F401

    after = PHASE1_EVENTS.stat().st_mtime, PHASE1_EVENTS.stat().st_size
    assert before == after


# --------------------------------------------------------------------------
# Full synthetic end-to-end
# --------------------------------------------------------------------------


def test_run_end_to_end_writes_all_outputs(tmp_path):
    tickers = [f"T{i}" for i in range(25)]
    rng = np.random.default_rng(14)
    rows = []
    for t in tickers:
        for cat, mean, n in [
            ("EARNINGS_RESULTS", 0.01, 60), ("OTHER", 0.0, 300), ("M_AND_A", 0.0, 25),
        ]:
            vals = rng.normal(mean, 0.01, n)
            rows.append(make_events([
                {"ticker": t, "date": SESSIONS[i], "category": cat, "abnormal_ret": v}
                for i, v in enumerate(vals)
            ]))
    events = pd.concat(rows, ignore_index=True)
    events_path = tmp_path / "events.parquet"
    events.to_parquet(events_path, index=False)

    cat_summary = pd.DataFrame([
        {"category": "EARNINGS_RESULTS", "horizon": 1, "mean_abnormal_return": 0.01},
        {"category": "M_AND_A", "horizon": 1, "mean_abnormal_return": 0.0},
    ])
    cat_summary_path = tmp_path / "cat_summary.parquet"
    cat_summary.to_parquet(cat_summary_path, index=False)

    sector_summary = pd.DataFrame([
        {
            "sector": "Information Technology", "category": "EARNINGS_RESULTS", "horizon": 1,
            "mean_abnormal_return": 0.01, "nw_mean": 0.01, "nw_se": 0.001,
            "n_tickers": 30, "sufficient_sample": True,
        },
    ])
    sector_summary_path = tmp_path / "sector_summary.parquet"
    sector_summary.to_parquet(sector_summary_path, index=False)

    from app.data.curate import save_curated

    panel_dir = tmp_path / "panel"
    for t in tickers:
        df = pd.DataFrame({
            "date": SESSIONS[:300], "ticker": t,
            "open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0, "volume": 1e6,
        })
        save_curated(df, panel_dir, t, source="yfinance")

    out_dir = tmp_path / "out"
    summary = run(
        events_path=events_path,
        category_summary_path=cat_summary_path,
        sector_summary_path=sector_summary_path,
        panel_dir=panel_dir,
        out_dir=out_dir,
        version="test",
        horizons=(1,),
    )
    for key in ("stock_category_horizon_summary", "stock_vs_other", "stock_ranking"):
        assert Path(summary["outputs"][key]).exists()
    assert Path(summary["coverage_path"]).exists()
    assert Path(summary["report_path"]).exists()
    report = Path(summary["report_path"]).read_text()
    assert "causation" in report.lower()
