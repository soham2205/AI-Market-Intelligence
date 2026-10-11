"""Phase 1: news category x investment horizon (and x sector) returns.

    classified_articles_EXPERIMENTAL.parquet (7-category classified articles,
    already built -- NOT reclassified, NOT rescored, NOT a re-run of FinBERT)
            |
            |  assign_trade_date() + news_is_current_entity()   <- REUSED
            v
    article-level, entity-correct (ticker, trade_date, category)
            |
            |  collapse: one row per (ticker, trade_date, category),
            |  n_articles = how many articles collapsed into it
            v
    event table
            |
            |  left-join on (ticker, date=trade_date) to Phase 0's
            |  benchmark_adjusted_returns_v1.parquet (READ-ONLY)
            |
            |  left-join sector_map (current GICS sector, READ-ONLY lookup)
            v
    event-level abnormal returns + validity flags, 6 horizons, + sector
            |
            +--> overall category x horizon summary / vs-OTHER comparison
            +--> sector x category x horizon summary / vs-OTHER-within-sector
            +--> EARNINGS_RESULTS horizon profile (PEAD view)
            v
    phase1_events_v1.parquet
    phase1_category_horizon_summary_v1.parquet
    phase1_category_vs_other_v1.parquet
    phase1_sector_category_horizon_summary_v1.parquet
    phase1_sector_category_vs_other_v1.parquet
    phase1_coverage_v1.json
    phase1_report_v1.md

Timestamp precision (inspected before writing any of this): 99.68% of
article `published_at` values collapse to exactly one of two fixed
end-of-day placeholder times (19:00 or 20:00 ET, i.e. 00:00 UTC under two
different DST offsets), not genuine scraped publication times -- this
archive is effectively DATE-ONLY, not intraday-precise. Both placeholder
times fall after the 09:30 ET market open, so the EXISTING
app.nlp.bucketing.first_tradable_session rule (before-open -> same session,
at/after-open -> next session) already assigns virtually every article to
the NEXT trading session after its date stamp -- which is exactly the
conservative, documented convention this task asks for when timing is
unreliable. That rule is reused UNCHANGED here, via
app.nlp.daily_sentiment.assign_trade_date, rather than re-derived.

Counting unit (documented once, referenced everywhere): one observation is
one (ticker, trade_date, category) triple. Multiple articles of the SAME
category on the same ticker-day collapse into ONE observation (n_articles
records how many). Multiple DIFFERENT categories on the same ticker-day
remain as separate observations, one per category -- a ticker-day with both
an EARNINGS_RESULTS and a MANAGEMENT article contributes one event to each.

Inference: same-day cross-sectional dependence (multiple tickers/events on
one date share a market-wide return factor) is handled by aggregating to one
point per CALENDAR DATE before any test -- the same device the standalone
sentiment diagnostics used. Overlapping forward-return windows (adjacent
daily observations at a 252-session horizon share up to 251 days of
underlying price history) are handled with a Newey-West HAC standard error,
lag = horizon - 1 sessions (the maximum possible overlap), on that
date-aggregated series. No `statsmodels` dependency is introduced; the HAC
formula is the standard one and is implemented directly (see
newey_west_mean_ci below).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from app.data.curate import load_curated
from app.data.entities import news_is_current_entity
from app.data.seams import forward_window_invalid, seams_for_frame
from app.events.benchmark_returns import (
    HORIZONS,
    abnormal_ret_col,
    valid_col,
)
from app.events.sector_map import DEFAULT_PATH as SECTOR_MAP_PATH
from app.events.sector_map import sector_series
from app.nlp.daily_sentiment import assign_trade_date, session_calendar

CATEGORIES: tuple[str, ...] = (
    "EARNINGS_RESULTS", "ANALYST_ACTION", "M_AND_A", "REGULATORY_LEGAL",
    "MANAGEMENT", "MAJOR_COMPANY_NEWS", "OTHER",
)
NON_OTHER_CATEGORIES = tuple(c for c in CATEGORIES if c != "OTHER")
EXPLORATORY_HORIZONS = (60, 126, 252)  # long enough that overlap dominates

# Minimum-sample gates. Below these, a cell is reported as "insufficient
# data" rather than with a number that looks precise but isn't.
MIN_EVENTS = 30
MIN_TICKERS = 5
MIN_OVERLAP_DATES = 15  # for a vs-OTHER comparison specifically
FDR_Q = 0.10  # Benjamini-Hochberg false-discovery rate

DEFAULT_CLASSIFIED = Path(
    "data/diagnostics_EXPERIMENTAL/news_category_vs_return/classified_articles_EXPERIMENTAL.parquet"
)
DEFAULT_RETURNS = Path("data/curated/benchmark_adjusted_returns_v1.parquet")
DEFAULT_PANEL_DIR = "data/curated_sp500"
DEFAULT_OUT_DIR = Path("data/curated")
OUTPUT_VERSION = "v1"


# ===========================================================================
# 1. Event table: article -> (ticker, trade_date, category) -> dedup -> join
# ===========================================================================


def build_event_table(
    classified_path: str | Path = DEFAULT_CLASSIFIED,
    returns_path: str | Path = DEFAULT_RETURNS,
    panel_dir: str | Path = DEFAULT_PANEL_DIR,
    sector_path: str | Path = SECTOR_MAP_PATH,
) -> tuple[pd.DataFrame, dict]:
    """Article -> entity-correct trade_date -> dedup to one row per
    (ticker, trade_date, category) -> join Phase 0 abnormal returns + sector.

    Returns (event_table, stats). Every exclusion is counted, never silent.
    """
    articles = pd.read_parquet(classified_path)
    sessions = session_calendar(panel_dir)
    assigned, assign_stats = assign_trade_date(articles, sessions)

    current_entity = [
        news_is_current_entity(t, d)
        for t, d in zip(assigned["ticker"], assigned["trade_date"], strict=True)
    ]
    n_misattributed = int(len(current_entity) - sum(current_entity))
    assigned = assigned[current_entity].copy()

    # ---- dedup: one row per (ticker, trade_date, category) -------------
    event = (
        assigned.groupby(["ticker", "trade_date", "category"], sort=True)
        .agg(n_articles=("article_id", "size"))
        .reset_index()
        .rename(columns={"trade_date": "date"})
    )

    # ---- join Phase 0 (LEFT, so a miss is counted, not silently dropped).
    # `source` is NaN exclusively when the left-join found no Phase 0 row at
    # all for (ticker, date) -- distinct from a matched row whose specific
    # horizon is merely invalid (that row's `source` is a real string).
    returns = pd.read_parquet(returns_path)
    merged = event.merge(returns, on=["ticker", "date"], how="left")
    no_price_bar = merged["source"].isna()
    n_no_price_bar = int(no_price_bar.sum())
    merged = merged[~no_price_bar].copy()
    # The LEFT merge upcasts the bool valid_{h}d columns to object dtype
    # (to hold NaN on unmatched rows); even after dropping those rows the
    # dtype stays object, holding native Python bool. That silently breaks
    # `~` (bitwise invert dispatches to int semantics on Python bool,
    # giving -2/-1, not logical negation) wherever a caller inverts the
    # mask -- restore proper bool dtype now, once, at the source.
    for h in HORIZONS:
        col = valid_col(h)
        if col in merged.columns:
            merged[col] = merged[col].astype(bool)

    # ---- sector (current GICS membership; documented limitation) -------
    sec = sector_series(merged["ticker"].unique(), path=sector_path)
    merged["sector"] = merged["ticker"].map(sec)
    n_no_sector = int(merged["sector"].isna().sum())

    stats_out = {
        "articles_in": int(assign_stats["articles_in"]),
        "dropped_before_calendar": int(assign_stats["dropped_before_calendar"]),
        "dropped_after_calendar": int(assign_stats["dropped_after_calendar"]),
        "dropped_as_misattributed_entity": n_misattributed,
        "events_after_dedup": int(len(event)),
        "events_excluded_no_price_bar_at_event_date": n_no_price_bar,
        "events_final": int(len(merged)),
        "events_with_unknown_sector": n_no_sector,
        "timestamp_precision_note": (
            "99.68% of published_at values are one of two fixed end-of-day "
            "placeholder times (19:00/20:00 ET); the archive is effectively "
            "date-only. Both placeholders fall after market open, so the "
            "existing next-session bucketing rule is already conservative "
            "for this data and is reused unchanged."
        ),
        "counting_unit": "one observation = one (ticker, trade_date, category)",
    }
    return merged.reset_index(drop=True), stats_out


# ===========================================================================
# 2. Newey-West HAC inference
# ===========================================================================


def newey_west_mean_ci(x: np.ndarray, lag: int) -> dict:
    """HAC (Newey-West, Bartlett kernel) standard error of the sample mean
    of a (possibly serially correlated) series `x`.

    `lag` should be the horizon length minus 1 sessions -- the maximum
    number of sessions two overlapping return windows can share. At lag 0
    this reduces to the ordinary (iid) standard error.
    """
    x = np.asarray(x, dtype=float)
    n = len(x)
    if n < 2:
        return {"n": n, "mean": float(x.mean()) if n else None, "se": None,
                "t_stat": None, "p_value": None, "ci95": [None, None]}
    xbar = float(x.mean())
    resid = x - xbar
    gamma0 = float(resid @ resid) / n
    var = gamma0
    lag = max(0, min(lag, n - 1))
    for k in range(1, lag + 1):
        w = 1 - k / (lag + 1)  # Bartlett kernel
        cov = float(resid[k:] @ resid[:-k]) / n
        var += 2 * w * cov
    var = max(var, 0.0)
    se = float(np.sqrt(var / n))
    if se > 0:
        t_stat = xbar / se
        p_value = float(2 * (1 - stats.t.cdf(abs(t_stat), df=max(n - 1, 1))))
        ci = [xbar - 1.96 * se, xbar + 1.96 * se]
    else:
        t_stat, p_value, ci = None, None, [xbar, xbar]
    return {"n": n, "mean": xbar, "se": se, "t_stat": t_stat, "p_value": p_value, "ci95": ci}


def newey_west_mean_ci_irregular(values: np.ndarray, positions: np.ndarray, lag: int) -> dict:
    """HAC mean/SE for an IRREGULARLY-SPACED series, lag measured in the
    series' own natural units (here: trading sessions on the master
    calendar), not array position.

    `newey_west_mean_ci` above implicitly assumes consecutive array entries
    are one time-step apart -- correct for a densely/regularly sampled
    series, but WRONG whenever observations skip sessions: event-dates for a
    sparse category (e.g. one sector's MANAGEMENT news) can sit many actual
    trading sessions apart even though they are array-adjacent. Applying the
    Bartlett kernel to array-position lag then smears the "correction for
    overlapping windows" across a window far wider, in real time, than the
    overlap it is meant to capture -- found during a Phase 1 audit: for
    Materials/MANAGEMENT the median gap between consecutive observed dates
    was 5 sessions (mean ~9.85, max 123), so a lag of 251 ARRAY positions
    corresponded to roughly 2,500 actual sessions, an order of magnitude
    beyond the intended horizon-1 window.

    `positions` gives each observation's true location on a regular clock
    (its index in the master session calendar). For each lag k = 1..lag,
    this sums the Bartlett-weighted product of residuals for every PAIR of
    observations whose real positions differ by EXACTLY k sessions. Pairs
    with no match at a given k contribute nothing at that k, which is
    correct: if no two observations are exactly that many sessions apart,
    there is no evidence about dependence at that lag.

    Implementation: scattering the (mean-centred) values onto a dense array
    indexed by real session position -- NaN where unobserved -- turns "every
    pair exactly k apart" into a single vectorized `dense[k:] * dense[:-k]`
    per lag (nan-masked), rather than a Python-level loop with a dict
    lookup per observation per lag. The two are mathematically identical;
    the vectorized form is what makes this usable across several hundred
    category/sector/horizon cells -- an earlier dict-based version of this
    function was computationally correct but did not finish within 30
    minutes on the real Phase 1 run.
    """
    values = np.asarray(values, dtype=float)
    positions = np.asarray(positions, dtype=np.int64)
    n = len(values)
    if n < 2:
        return {"n": n, "mean": float(values.mean()) if n else None, "se": None,
                "t_stat": None, "p_value": None, "ci95": [None, None]}
    order = np.argsort(positions)
    values, positions = values[order], positions[order]
    xbar = float(values.mean())
    resid = values - xbar
    gamma0 = float(resid @ resid) / n
    var = gamma0
    lag = max(0, lag)

    lo = int(positions[0])
    dense = np.full(int(positions[-1]) - lo + 1, np.nan)
    dense[positions - lo] = resid
    for k in range(1, min(lag, len(dense) - 1) + 1):
        w = 1 - k / (lag + 1)  # Bartlett kernel, by REAL session distance
        prod = dense[k:] * dense[: len(dense) - k]
        if np.any(~np.isnan(prod)):
            # Same /n convention as gamma0 (population, over all n
            # observations, not over the count of matched pairs), so this
            # agrees exactly with newey_west_mean_ci on dense/regular input
            # -- see test_irregular_matches_regular_on_densely_consecutive_positions.
            cov = float(np.nansum(prod)) / n
            var += 2 * w * cov
    var = max(var, 0.0)
    se = float(np.sqrt(var / n))
    if se > 0:
        t_stat = xbar / se
        p_value = float(2 * (1 - stats.t.cdf(abs(t_stat), df=max(n - 1, 1))))
        ci = [xbar - 1.96 * se, xbar + 1.96 * se]
    else:
        t_stat, p_value, ci = None, None, [xbar, xbar]
    return {"n": n, "mean": xbar, "se": se, "t_stat": t_stat, "p_value": p_value, "ci95": ci}


def session_positions(dates, session_pos: pd.Series) -> np.ndarray:
    """Map an array/Index of dates to their integer position on the master
    trading-session calendar, via a precomputed date->position lookup."""
    return session_pos.reindex(dates).to_numpy()


def benjamini_hochberg(p_values: list[float], q: float = FDR_Q) -> list[bool]:
    """Benjamini-Hochberg step-up FDR: returns a significance flag per
    p-value, controlling the expected false-discovery rate at `q` across
    the whole family of tests passed in -- not a per-test significance
    level. NaN p-values are never flagged significant.

    `thresh` tracks the largest p-value clearing its step-up criterion as a
    SENTINEL (None = no threshold found yet), not as a float defaulting to
    0.0 -- found via a Phase 2 audit: when the single smallest p-value in a
    family underflows to exactly 0.0 (routine with large stock-level sample
    sizes and a strong effect), a float default of 0.0 is indistinguishable
    from "no threshold found", and the old `thresh > 0` guard wrongly
    excluded that most-significant result. No stored Phase 1 p-value ever
    hit this edge case (smallest observed was ~9.3e-05), so this fix does
    not change any already-reported Phase 1 number -- confirmed before
    changing this function, not merely assumed.
    """
    idx = [i for i, p in enumerate(p_values) if p is not None and not np.isnan(p)]
    m = len(idx)
    flags = [False] * len(p_values)
    if m == 0:
        return flags
    ordered = sorted(idx, key=lambda i: p_values[i])
    thresh: float | None = None
    for rank, i in enumerate(ordered, start=1):
        crit = (rank / m) * q
        if p_values[i] <= crit:
            thresh = p_values[i]
    if thresh is None:
        return flags
    for i in idx:
        flags[i] = p_values[i] <= thresh
    return flags


def _date_aggregate(group: pd.DataFrame, ret_col: str, valid_c: str) -> pd.Series:
    """Mean abnormal return per DATE for a group of events, valid rows only.
    One point per date removes same-day cross-sectional dependence before
    any time-series inference is applied."""
    v = group[group[valid_c]]
    if v.empty:
        return pd.Series(dtype=float)
    return v.groupby("date")[ret_col].mean()


# ===========================================================================
# 3. Per-group (category[, sector]) x horizon descriptive stats
# ===========================================================================


def _describe_group(events: pd.DataFrame, horizon: int, session_pos: pd.Series) -> dict:
    ret_c, valid_c = abnormal_ret_col(horizon), valid_col(horizon)
    total = len(events)
    valid = events[events[valid_c]]
    n_valid = len(valid)
    r = valid[ret_c]
    n_tickers = int(valid["ticker"].nunique())
    sufficient = n_valid >= MIN_EVENTS and n_tickers >= MIN_TICKERS

    series = _date_aggregate(events, ret_c, valid_c)
    nw = (
        newey_west_mean_ci_irregular(
            series.to_numpy(), session_positions(series.index, session_pos), lag=max(horizon - 1, 0)
        )
        if len(series)
        else None
    )

    return {
        "horizon": horizon,
        "n_events_total": total,
        "n_events_valid": n_valid,
        "n_events_invalid": total - n_valid,
        "n_tickers": n_tickers,
        "n_dates": int(series.shape[0]) if len(series) else 0,
        "sufficient_sample": sufficient,
        "exploratory_horizon": horizon in EXPLORATORY_HORIZONS,
        "mean_abnormal_return": float(r.mean()) if n_valid else None,
        "median_abnormal_return": float(r.median()) if n_valid else None,
        "p25_abnormal_return": float(r.quantile(0.25)) if n_valid else None,
        "p75_abnormal_return": float(r.quantile(0.75)) if n_valid else None,
        "std_abnormal_return": float(r.std()) if n_valid > 1 else None,
        "pct_positive": float((r > 0).mean()) if n_valid else None,
        "pct_negative": float((r < 0).mean()) if n_valid else None,
        "nw_mean": nw["mean"] if nw else None,
        "nw_se": nw["se"] if nw else None,
        "nw_ci95_lo": nw["ci95"][0] if nw else None,
        "nw_ci95_hi": nw["ci95"][1] if nw else None,
        "nw_lag_sessions": max(horizon - 1, 0),
    }


def category_horizon_summary(
    events: pd.DataFrame, session_pos: pd.Series, horizons=HORIZONS
) -> pd.DataFrame:
    """Overall (all-sector) category x horizon descriptive table."""
    rows = []
    for cat, g in events.groupby("category", sort=True):
        for h in horizons:
            rows.append({"category": cat, **_describe_group(g, h, session_pos)})
    return pd.DataFrame(rows)


def sector_category_horizon_summary(
    events: pd.DataFrame, session_pos: pd.Series, horizons=HORIZONS
) -> pd.DataFrame:
    """sector x category x horizon descriptive table (known sectors only)."""
    known = events[events["sector"].notna()]
    rows = []
    for (sec, cat), g in known.groupby(["sector", "category"], sort=True):
        for h in horizons:
            rows.append({"sector": sec, "category": cat, **_describe_group(g, h, session_pos)})
    return pd.DataFrame(rows)


# ===========================================================================
# 4. category vs OTHER (overall, and within-sector)
# ===========================================================================


def _compare_vs_other(
    cat_events: pd.DataFrame, other_events: pd.DataFrame, horizon: int, session_pos: pd.Series
) -> dict:
    ret_c, valid_c = abnormal_ret_col(horizon), valid_col(horizon)
    cat_series = _date_aggregate(cat_events, ret_c, valid_c)
    oth_series = _date_aggregate(other_events, ret_c, valid_c)
    common = cat_series.index.intersection(oth_series.index)

    cat_valid = cat_events[cat_events[valid_c]]
    n_events, n_tickers = len(cat_valid), int(cat_valid["ticker"].nunique())
    sufficient = (
        n_events >= MIN_EVENTS and n_tickers >= MIN_TICKERS and len(common) >= MIN_OVERLAP_DATES
    )
    if not sufficient:
        return {
            "horizon": horizon, "n_events_category": n_events, "n_tickers_category": n_tickers,
            "n_overlapping_dates": int(len(common)), "sufficient_sample": False,
            "exploratory_horizon": horizon in EXPLORATORY_HORIZONS,
            "diff_mean": None, "se": None, "t_stat": None, "p_value": None,
            "ci95_lo": None, "ci95_hi": None,
        }

    diff = (cat_series.loc[common] - oth_series.loc[common]).to_numpy()
    nw = newey_west_mean_ci_irregular(
        diff, session_positions(common, session_pos), lag=max(horizon - 1, 0)
    )
    return {
        "horizon": horizon, "n_events_category": n_events, "n_tickers_category": n_tickers,
        "n_overlapping_dates": int(len(common)), "sufficient_sample": True,
        "exploratory_horizon": horizon in EXPLORATORY_HORIZONS,
        "diff_mean": nw["mean"], "se": nw["se"], "t_stat": nw["t_stat"], "p_value": nw["p_value"],
        "ci95_lo": nw["ci95"][0], "ci95_hi": nw["ci95"][1],
    }


def category_vs_other(
    events: pd.DataFrame, session_pos: pd.Series, horizons=HORIZONS
) -> pd.DataFrame:
    other = events[events["category"] == "OTHER"]
    rows = []
    for cat in NON_OTHER_CATEGORIES:
        cat_events = events[events["category"] == cat]
        for h in horizons:
            rows.append({"category": cat, **_compare_vs_other(cat_events, other, h, session_pos)})
    df = pd.DataFrame(rows)
    pvals = [None if not r["sufficient_sample"] else r["p_value"] for r in rows]
    df["significant_fdr"] = benjamini_hochberg(pvals, FDR_Q)
    df["n_tests_in_family"] = int(sum(v is not None and not np.isnan(v) for v in pvals))
    return df


def sector_category_vs_other(
    events: pd.DataFrame, session_pos: pd.Series, horizons=HORIZONS
) -> pd.DataFrame:
    known = events[events["sector"].notna()]
    rows = []
    for sec, sg in known.groupby("sector", sort=True):
        other = sg[sg["category"] == "OTHER"]
        for cat in NON_OTHER_CATEGORIES:
            cat_events = sg[sg["category"] == cat]
            for h in horizons:
                cmp = _compare_vs_other(cat_events, other, h, session_pos)
                rows.append({"sector": sec, "category": cat, **cmp})
    df = pd.DataFrame(rows)
    pvals = [None if not r["sufficient_sample"] else r["p_value"] for r in rows]
    df["significant_fdr"] = benjamini_hochberg(pvals, FDR_Q)
    df["n_tests_in_family"] = int(sum(v is not None and not np.isnan(v) for v in pvals))
    return df


# ===========================================================================
# 5. Invalidity reason breakdown (reuses Phase 0's own primitives, applied
#    only to the small subset of EVENT rows, not the full panel)
# ===========================================================================


def invalidity_reasons(
    events: pd.DataFrame,
    panel_dir: str | Path,
    horizon: int,
    panel: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """For invalid events at `horizon`, attribute WHY: beyond the known
    calendar, a missing price at one of the four endpoints, or a window
    crossing a registered seam -- using the exact same primitives Phase 0
    uses (app.data.seams), not a re-derivation.

    Pass `panel` (already loaded) when calling this repeatedly across
    horizons for the same panel_dir -- load_curated() re-reads and re-sorts
    the full multi-million-row panel from disk on every call otherwise,
    which dominates runtime when this is invoked once per horizon.
    """
    if panel is None:
        panel = load_curated(panel_dir)
    sessions = pd.DatetimeIndex(sorted(panel["date"].unique()))
    n = len(sessions)
    valid_c = valid_col(horizon)
    invalid = events[~events[valid_c]].copy()
    if invalid.empty:
        return invalid.assign(reason=pd.Series(dtype=str))

    # Only the tickers that actually have an invalid row here are worth
    # examining -- iterating the full 447-ticker panel when, say,
    # EARNINGS_RESULTS has invalid rows for a handful of tickers wastes
    # almost all of the work.
    wanted = set(invalid["ticker"].unique())
    reasons = []
    for ticker, g in panel[panel["ticker"].isin(wanted)].groupby("ticker"):
        seams = seams_for_frame(g)
        own_present = set(g["date"])
        # Computed ONCE per ticker, not once per invalid row of that ticker.
        mask = forward_window_invalid(sessions, horizon, seams) if seams else None
        rows = invalid[invalid["ticker"] == ticker]
        for date in rows["date"]:
            pos = sessions.searchsorted(date)
            if pos + horizon >= n:
                reasons.append((ticker, date, "beyond_calendar"))
                continue
            if mask is not None and pos < n and mask[pos]:
                reasons.append((ticker, date, "seam_crossing"))
                continue
            end_date = sessions[pos + horizon]
            if date not in own_present or end_date not in own_present:
                reasons.append((ticker, date, "missing_price"))
            else:
                reasons.append((ticker, date, "missing_benchmark_price"))
    reason_df = pd.DataFrame(reasons, columns=["ticker", "date", "reason"])
    return invalid.merge(reason_df, on=["ticker", "date"], how="left")


# ===========================================================================
# 6. Orchestration
# ===========================================================================


def run(
    classified_path: str | Path = DEFAULT_CLASSIFIED,
    returns_path: str | Path = DEFAULT_RETURNS,
    panel_dir: str | Path = DEFAULT_PANEL_DIR,
    sector_path: str | Path = SECTOR_MAP_PATH,
    out_dir: str | Path = DEFAULT_OUT_DIR,
    version: str = OUTPUT_VERSION,
    horizons: tuple[int, ...] = HORIZONS,
) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    events, build_stats = build_event_table(classified_path, returns_path, panel_dir, sector_path)
    events_path = out_dir / f"phase1_events_{version}.parquet"
    events.to_parquet(events_path, index=False)

    # Loaded ONCE and reused everywhere a master session calendar is needed:
    # the invalidity-reason breakdown below, AND -- critically -- the
    # date->session-position lookup that makes the Newey-West lag below
    # track real trading sessions rather than array position (see
    # newey_west_mean_ci_irregular's docstring for why that distinction
    # matters; it was a genuine bug found auditing the sector-level results).
    reason_panel = load_curated(panel_dir)
    sessions = pd.DatetimeIndex(sorted(reason_panel["date"].unique()))
    session_pos = pd.Series(np.arange(len(sessions)), index=sessions)

    cat_summary = category_horizon_summary(events, session_pos, horizons)
    cat_summary_path = out_dir / f"phase1_category_horizon_summary_{version}.parquet"
    cat_summary.to_parquet(cat_summary_path, index=False)

    cat_vs_other = category_vs_other(events, session_pos, horizons)
    cat_vs_other_path = out_dir / f"phase1_category_vs_other_{version}.parquet"
    cat_vs_other.to_parquet(cat_vs_other_path, index=False)

    sector_summary = sector_category_horizon_summary(events, session_pos, horizons)
    sector_summary_path = out_dir / f"phase1_sector_category_horizon_summary_{version}.parquet"
    sector_summary.to_parquet(sector_summary_path, index=False)

    sector_vs_other = sector_category_vs_other(events, session_pos, horizons)
    sector_vs_other_path = out_dir / f"phase1_sector_category_vs_other_{version}.parquet"
    sector_vs_other.to_parquet(sector_vs_other_path, index=False)

    # invalidity reasons for the two most-scrutinised cases: EARNINGS_RESULTS
    # at every horizon (the PEAD question) and the overall grid at h=252
    # (where exclusions are largest) -- not every category x horizon, to
    # keep this step's cost proportionate to what the report actually uses.
    earnings_events = events[events["category"] == "EARNINGS_RESULTS"]
    earnings_reasons = {
        h: invalidity_reasons(earnings_events, panel_dir, h, panel=reason_panel)["reason"]
        .value_counts()
        .to_dict()
        for h in horizons
    }

    coverage = {
        "purpose": "Phase 1: news category x investment horizon (and x sector) returns, "
                   "built on Phase 0 benchmark-adjusted returns. Associations, not causation.",
        "inputs": {
            "classified_articles": str(classified_path),
            "phase0_returns": str(returns_path),
            "panel_dir": str(panel_dir),
        },
        "build_stats": build_stats,
        "sector_source": "configs/sp500_sectors.json (Wikipedia GICS, current membership only)",
        "minimum_sample_rules": {
            "min_events": MIN_EVENTS, "min_tickers": MIN_TICKERS,
            "min_overlap_dates_for_comparison": MIN_OVERLAP_DATES,
        },
        "inference_method": (
            "Newey-West HAC standard error (Bartlett kernel, lag = horizon-1 TRADING "
            "SESSIONS) on a date-aggregated series (one point per calendar date, removing "
            "same-day cross-sectional dependence before the HAC correction for window "
            "overlap). Lag is measured by each observation's real position on the master "
            "session calendar (newey_west_mean_ci_irregular), not by array adjacency in the "
            "date-aggregated series -- event-dates for sparse groups (most sector-level "
            "cells) are not evenly spaced, so array-position lag would smear the overlap "
            "correction across a far wider real-time window than intended (found and fixed "
            "in a Phase 1 audit). Benjamini-Hochberg FDR control at q=0.10, applied "
            "separately within the overall category-vs-OTHER family and the "
            "sector-category-vs-OTHER family."
        ),
        "exploratory_horizons": list(EXPLORATORY_HORIZONS),
        "earnings_results_invalidity_reasons_by_horizon": earnings_reasons,
        "outputs": {
            "events": str(events_path),
            "category_horizon_summary": str(cat_summary_path),
            "category_vs_other": str(cat_vs_other_path),
            "sector_category_horizon_summary": str(sector_summary_path),
            "sector_category_vs_other": str(sector_vs_other_path),
        },
        "n_sector_tests_run": (
            int(sector_vs_other["n_tests_in_family"].iloc[0]) if len(sector_vs_other) else 0
        ),
        "n_overall_tests_run": (
            int(cat_vs_other["n_tests_in_family"].iloc[0]) if len(cat_vs_other) else 0
        ),
    }
    coverage_path = out_dir / f"phase1_coverage_{version}.json"
    coverage_path.write_text(json.dumps(coverage, indent=1, default=str))
    coverage["coverage_path"] = str(coverage_path)

    report_path = out_dir / f"phase1_report_{version}.md"
    report_md = render_markdown_report(
        events, cat_summary, cat_vs_other, sector_vs_other, coverage, horizons
    )
    report_path.write_text(report_md)
    coverage["report_path"] = str(report_path)
    return coverage


# ===========================================================================
# 7. Markdown report
# ===========================================================================


def _fmt_pct(x) -> str:
    return "n/a" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x*100:+.2f}%"


def render_markdown_report(
    events: pd.DataFrame,
    cat_summary: pd.DataFrame,
    cat_vs_other: pd.DataFrame,
    sector_vs_other: pd.DataFrame,
    coverage: dict,
    horizons=HORIZONS,
) -> str:
    lines = [
        "# Phase 1: News Category x Investment Horizon",
        "",
        "Historical associations between classified news categories and "
        "benchmark-adjusted forward returns. **These are observed "
        "associations, not evidence that the news caused the return.**",
        "",
        f"- Counting unit: {coverage['build_stats']['counting_unit']}",
        f"- {coverage['build_stats']['timestamp_precision_note']}",
        f"- Final event rows: {coverage['build_stats']['events_final']:,} "
        f"(excluded: {coverage['build_stats']['events_excluded_no_price_bar_at_event_date']:,} "
        "with no price bar at the event date, "
        f"{coverage['build_stats']['dropped_as_misattributed_entity']:,} misattributed-entity "
        "news, see coverage JSON for full breakdown)",
        f"- Sector coverage: {coverage['build_stats']['events_with_unknown_sector']:,} event rows "
        "have no current GICS sector match and are excluded from sector-level tables only "
        "(current membership, not historical -- see sector_map.py)",
        "",
        "## Overall category x horizon",
        "",
        "| category | horizon | n events (valid) | n tickers | mean | median | "
        "p25 | p75 | % positive | 95% CI (NW) |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for _, r in cat_summary.sort_values(["category", "horizon"]).iterrows():
        flag = "" if r["sufficient_sample"] else " *(insufficient sample)*"
        expl = " *(exploratory)*" if r["exploratory_horizon"] else ""
        ci = (
            f"[{_fmt_pct(r['nw_ci95_lo'])}, {_fmt_pct(r['nw_ci95_hi'])}]"
            if r["nw_ci95_lo"] is not None else "n/a"
        )
        pct_pos = f"{r['pct_positive']*100:.1f}%" if r["pct_positive"] is not None else "n/a"
        lines.append(
            f"| {r['category']} | {r['horizon']}d{expl} | {r['n_events_valid']:,}{flag} | "
            f"{r['n_tickers']:,} | {_fmt_pct(r['mean_abnormal_return'])} | "
            f"{_fmt_pct(r['median_abnormal_return'])} | {_fmt_pct(r['p25_abnormal_return'])} | "
            f"{_fmt_pct(r['p75_abnormal_return'])} | {pct_pos} | {ci} |"
        )

    lines += [
        "",
        "## Category vs. OTHER (same horizon)",
        "",
        f"Newey-West HAC inference, date-aggregated. Benjamini-Hochberg FDR "
        f"control at q={FDR_Q} across {coverage['n_overall_tests_run']} tests actually run. "
        f"Horizons {EXPLORATORY_HORIZONS} are labelled exploratory: overlapping windows this "
        "long leave a much smaller effective independent sample than the raw event count suggests.",
        "",
        "| category | horizon | diff vs OTHER | 95% CI | p (raw) | FDR-significant "
        "| n events | n overlap dates |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for _, r in cat_vs_other.sort_values(["category", "horizon"]).iterrows():
        if not r["sufficient_sample"]:
            lines.append(
                f"| {r['category']} | {r['horizon']}d | *insufficient sample* | - | - | - | "
                f"{r['n_events_category']:,} | {r['n_overlapping_dates']} |"
            )
            continue
        expl = " *(exploratory)*" if r["exploratory_horizon"] else ""
        sig = "**yes**" if r["significant_fdr"] else "no"
        lines.append(
            f"| {r['category']} | {r['horizon']}d{expl} | {_fmt_pct(r['diff_mean'])} | "
            f"[{_fmt_pct(r['ci95_lo'])}, {_fmt_pct(r['ci95_hi'])}] | {r['p_value']:.3g} | "
            f"{sig} | {r['n_events_category']:,} | {r['n_overlapping_dates']} |"
        )

    # ---- EARNINGS_RESULTS / PEAD section -----------------------------
    lines += ["", "## EARNINGS_RESULTS across all six horizons (PEAD check)", ""]
    earn = cat_summary[cat_summary["category"] == "EARNINGS_RESULTS"].sort_values("horizon")
    earn_vs = cat_vs_other[cat_vs_other["category"] == "EARNINGS_RESULTS"].set_index("horizon")
    lines.append("| horizon | mean | median | 95% CI (NW) | n events | vs OTHER | FDR-sig |")
    lines.append("|---|---|---|---|---|---|---|")
    for _, r in earn.iterrows():
        h = r["horizon"]
        vs = earn_vs.loc[h] if h in earn_vs.index else None
        vs_ok = vs is not None and vs["sufficient_sample"]
        vs_str = _fmt_pct(vs["diff_mean"]) if vs_ok else "n/a"
        sig = "**yes**" if vs_ok and vs["significant_fdr"] else "no"
        ci = (
            f"[{_fmt_pct(r['nw_ci95_lo'])}, {_fmt_pct(r['nw_ci95_hi'])}]"
            if r["nw_ci95_lo"] is not None else "n/a"
        )
        lines.append(
            f"| {h}d | {_fmt_pct(r['mean_abnormal_return'])} | "
            f"{_fmt_pct(r['median_abnormal_return'])} | {ci} | {r['n_events_valid']:,} | "
            f"{vs_str} | {sig} |"
        )
    lines.append("")
    lines.append(
        "No assumption of drift (PEAD) is made in advance; the table above is read "
        "as-is to see whether the typical response appears immediately, persists, "
        "grows, or reverses across the six horizons."
    )

    # ---- sector highlights --------------------------------------------
    lines += ["", "## Sector x category highlights", ""]
    sig_sector = sector_vs_other[
        sector_vs_other["sufficient_sample"] & sector_vs_other["significant_fdr"]
    ].sort_values(["category", "sector", "horizon"])
    if sig_sector.empty:
        lines.append(
            "No sector x category x horizon cell was both sufficiently sampled "
            f"(>= {MIN_EVENTS} events, >= {MIN_TICKERS} tickers, >= {MIN_OVERLAP_DATES} "
            "overlapping dates) and FDR-significant. See the full "
            "phase1_sector_category_vs_other table for all computed cells, "
            "including those below the significance threshold."
        )
    else:
        lines.append(
            "| sector | category | horizon | diff vs OTHER (in-sector) | 95% CI "
            "| n events | n tickers |"
        )
        lines.append("|---|---|---|---|---|---|---|")
        for _, r in sig_sector.iterrows():
            expl = " *(exploratory)*" if r["exploratory_horizon"] else ""
            ci_str = f"[{_fmt_pct(r['ci95_lo'])}, {_fmt_pct(r['ci95_hi'])}]"
            lines.append(
                f"| {r['sector']} | {r['category']} | {r['horizon']}d{expl} | "
                f"{_fmt_pct(r['diff_mean'])} | {ci_str} | "
                f"{r['n_events_category']:,} | {r['n_tickers_category']:,} |"
            )

    lines += [
        "",
        "## Limitations",
        "",
        "- Sector classification is CURRENT GICS membership only, applied retroactively; "
        "it is not point-in-time and will misclassify companies that changed sector "
        "(spin-offs, reclassifications) at some point in the 2008-2024 window.",
        "- Timestamps are effectively date-only for 99.68% of articles (see above); "
        "the next-session convention is conservative but coarser than true intraday timing.",
        f"- {EXPLORATORY_HORIZONS} horizons rest on far fewer independent observations than "
        "their raw event counts suggest, because adjacent daily windows overlap by up to "
        "horizon-1 sessions. Treat results at these horizons as exploratory.",
        "- Associations only. News coinciding with a return is not evidence the news caused it.",
    ]
    return "\n".join(lines)
