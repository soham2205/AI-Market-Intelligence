"""Phase 3: rolling 365-calendar-day news-impact intelligence.

    phase1_events_v1.parquet          (audit-corrected Phase 1 event table,
                                       READ-ONLY -- no re-ingestion, no
                                       re-classification, no FinBERT re-run,
                                       no model retraining)
            |
            |  window_grid(): 365-calendar-day windows, anchored at the LAST
            |  event date and stepped BACKWARDS in documented increments
            |  (91d for category level, 182d for stock level)
            v
    per-window event subsets (Phase 0 `valid_{h}d` flags untouched: an event
    contributes at horizon h only if its full forward window is valid)
            |
            +--> rolling_category_horizon()  window x category x horizon
            |       reuses _describe_group + _compare_vs_other VERBATIM
            +--> rolling_stock()             window x ticker x category x horizon
            |       numpy fast path, test-pinned equal to _describe_group
            +--> classify_temporal_change()  strengthen/weaken/reverse/consistent
            |       on NON-OVERLAPPING windows only
            +--> candidate_concentration()   are the Phase 2 hits event-dominated?
            v
    phase3_rolling_category_horizon_v1.parquet
    phase3_rolling_stock_v1.parquet
    phase3_temporal_change_v1.parquet
    phase3_freshness_v1.json
    phase3_report_v1.md
    phase3_figures/*.png                      (optional; skipped if no matplotlib)

STALENESS, STATED UP FRONT: this phase adds no new data. The underlying news
archive ends 2024-01-08, and because an event needs its FULL forward window to
be valid, the newest assessable event date recedes further with every horizon
(1d: 2024-01-05 ... 252d: 2023-01-05). Every "recent" statement in Phase 3
means "recent relative to a news archive that stopped in January 2024", NOT
relative to today. This is rolling HISTORICAL analysis, not real-time
intelligence, and nothing here should be described as the latter.

WHY ROLLING WINDOWS NEED THEIR OWN CARE (and what is deliberately NOT claimed):

1. Overlapping windows are not independent. With a 365-day window stepped 91
   days, four consecutive windows share up to 3/4 of their events. Any
   statement that compares windows therefore uses only a NON-OVERLAPPING
   subset (every ceil(365/stride)-th window, selected backwards from the most
   recent so the newest window is always included), where the two sides share
   no events at all.
2. Overlapping forward-return windows WITHIN a window are handled the same way
   Phase 1 and Phase 2 handle them: date-aggregate first (removes same-day
   cross-sectional dependence), then a Newey-West HAC standard error with lag =
   horizon - 1 measured in REAL trading-session distance via
   newey_west_mean_ci_irregular (the Phase 1 audit fix, reused unchanged).
3. At long horizons inside a 365-day window the two collide: a 252-session
   forward window is itself ~ a calendar year, so a 365-day window holds barely
   more than one non-overlapping outcome period. The HAC correction is applied,
   but its asymptotics assume many more effective observations than exist here.
   Horizons 60/126/252 are therefore flagged `exploratory_horizon` (inherited
   from Phase 1) AND, in a rolling context, should be read as descriptive
   series rather than as a sequence of independent tests.
4. Rolling multiplies comparisons enormously (55 windows x 7 categories x 6
   horizons). Benjamini-Hochberg is applied to the rolling vs-OTHER family as a
   screen, but window overlap breaks the independence/PRDS conditions BH relies
   on, so those flags are labelled exploratory. The one inferential claim this
   phase does make -- `change_established` in the temporal-change table -- is
   BH-corrected over a family built from DISJOINT windows, where the
   assumptions are defensible.

A statistically inconclusive or empty recent window is NEVER evidence of no
impact. It usually means the sample is too small, and at long horizons it means
the outcome window has not finished inside the available data. Those two cases
are reported as `insufficient_recent_data`, never as "no effect".
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from app.data.curate import load_curated
from app.events.benchmark_returns import abnormal_ret_col, valid_col
from app.events.category_horizon_analysis import (
    EXPLORATORY_HORIZONS,
    FDR_Q,
    HORIZONS,
    _compare_vs_other,
    _describe_group,
    benjamini_hochberg,
    newey_west_mean_ci_irregular,
    session_positions,
)

# --- window geometry -------------------------------------------------------
WINDOW_DAYS = 365
CATEGORY_STRIDE_DAYS = 91  # ~quarterly: fine enough to see change, coarse
#                            enough that the window count stays interpretable
STOCK_STRIDE_DAYS = 182  # ~semi-annual: per-ticker windows are sparse, so a
#                          finer stride would multiply near-duplicate tests
#                          without adding independent information

# --- minimum-sample gates, fixed BEFORE looking at any ranking -------------
# Category level. MIN_EVENTS matches Phase 1's 30; the ticker bar is RAISED
# from Phase 1's 5 to 10 because a category-wide claim inside a single year
# should rest on breadth across names, not on a handful of busy tickers.
ROLL_MIN_EVENTS = 30
ROLL_MIN_TICKERS = 10
# Stock level. Deliberately permissive, and permissive on purpose: corporate
# news arrives roughly 1-16 times a YEAR per ticker per category (measured on
# this archive: MANAGEMENT ~1.4/yr median, EARNINGS_RESULTS ~16/yr median), so
# a 365-day window simply cannot hold a Phase-2-sized sample. 8 is low enough
# to show what little signal exists and is reported as `exploratory`, never as
# an adequately-supported result.
ROLL_STOCK_MIN_EVENTS = 8
ROLL_STOCK_EXPLORATORY_MIN = 4

# --- temporal-change rules, also fixed in advance --------------------------
TREND_MIN_WINDOWS = 4  # non-overlapping sufficient windows needed to say anything
STRENGTHEN_RATIO = 1.25  # |late| >= 1.25 x |early| reads as strengthening
WEAKEN_RATIO = 0.75  # |late| <= 0.75 x |early| reads as weakening

# The two Phase 2 candidates, carried in by (ticker, category, horizon) so
# their rolling rows are emitted even when far too sparse to analyse -- their
# sparsity is the finding, and silently dropping them would hide it.
PHASE2_CANDIDATES: tuple[tuple[str, str, int], ...] = (
    ("BR", "MANAGEMENT", 20),
    ("BEN", "MAJOR_COMPANY_NEWS", 1),
)

DEFAULT_EVENTS = Path("data/curated/phase1_events_v1.parquet")
DEFAULT_PANEL_DIR = "data/curated_sp500"
DEFAULT_OUT_DIR = Path("data/curated")
OUTPUT_VERSION = "v1"


# ===========================================================================
# 1. Window geometry
# ===========================================================================


def window_grid(
    first_date, last_date, window_days: int = WINDOW_DAYS, stride_days: int = CATEGORY_STRIDE_DAYS
) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """365-calendar-day windows as inclusive [start, end] pairs, chronological.

    Anchored at `last_date` and stepped BACKWARDS, so the final window is
    always exactly the last `window_days` of available data -- the window a
    reader cares most about. A window is kept only if it fits entirely inside
    the data range, so every window covers the same calendar span and window
    sizes are comparable.
    """
    first_date, last_date = pd.Timestamp(first_date), pd.Timestamp(last_date)
    span = pd.Timedelta(days=window_days - 1)
    out: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    end = last_date
    while end - span >= first_date:
        out.append((end - span, end))
        end -= pd.Timedelta(days=stride_days)
    return list(reversed(out))


def non_overlapping_subset(
    window_ends: list[pd.Timestamp], window_days: int = WINDOW_DAYS
) -> list[int]:
    """Positions of a maximal set of windows that share NO calendar days,
    chosen backwards from the most recent so the newest window is always in.

    Returned chronologically. Used wherever two windows are compared, because
    adjacent overlapping windows share most of their events and cannot be
    treated as independent observations of the same effect.
    """
    if not window_ends:
        return []
    keep: list[int] = []
    cutoff: pd.Timestamp | None = None
    for i in range(len(window_ends) - 1, -1, -1):
        end = pd.Timestamp(window_ends[i])
        if cutoff is None or end <= cutoff:
            keep.append(i)
            cutoff = end - pd.Timedelta(days=window_days)
    return sorted(keep)


# ===========================================================================
# 2. Freshness / coverage
# ===========================================================================


def data_freshness(events: pd.DataFrame, horizons=HORIZONS, as_of=None) -> dict:
    """Last news date, last VALID outcome date per horizon, and how stale the
    archive is relative to `as_of` (default: today).

    The per-horizon frontier is the honest answer to "how recent can this
    analysis be?": a 252-session outcome cannot be observed for an event whose
    forward window runs past the end of the price data, so the newest usable
    event date is much older at long horizons than at 1 day.
    """
    as_of = (
        pd.Timestamp(as_of).normalize() if as_of is not None
        else pd.Timestamp.today().normalize()
    )
    last_news = pd.Timestamp(events["date"].max())
    first_news = pd.Timestamp(events["date"].min())
    per_horizon = {}
    for h in horizons:
        v = events[events[valid_col(h)]]
        last_valid = pd.Timestamp(v["date"].max()) if len(v) else None
        per_horizon[f"{h}d"] = {
            "n_events_valid": int(len(v)),
            "n_events_invalid": int(len(events) - len(v)),
            "pct_invalid": round(float((len(events) - len(v)) / max(len(events), 1)) * 100, 3),
            "last_valid_event_date": str(last_valid.date()) if last_valid is not None else None,
            "days_of_outcome_frontier_lost": (
                int((last_news - last_valid).days) if last_valid is not None else None
            ),
            "staleness_days_vs_as_of": (
                int((as_of - last_valid).days) if last_valid is not None else None
            ),
        }
    return {
        "as_of": str(as_of.date()),
        "first_news_date": str(first_news.date()),
        "last_news_date": str(last_news.date()),
        "news_staleness_days_vs_as_of": int((as_of - last_news).days),
        "news_staleness_years_vs_as_of": round((as_of - last_news).days / 365.25, 2),
        "n_events_total": int(len(events)),
        "n_tickers": int(events["ticker"].nunique()),
        "n_categories": int(events["category"].nunique()),
        "real_time": False,
        "real_time_note": (
            "Phase 3 refreshed NO data. It re-slices the existing archive into "
            "rolling windows. The archive ends at last_news_date above, so this is "
            "rolling historical analysis and must not be described as real-time or "
            "current-market intelligence."
        ),
        "per_horizon": per_horizon,
    }


# ===========================================================================
# 3. Rolling category x horizon  (reuses Phase 1 primitives verbatim)
# ===========================================================================


def _direction_label(lo, hi) -> str:
    """positive / negative / inconclusive from a confidence interval alone.

    Per-window and UNCORRECTED for the many windows examined, so this is a
    descriptive read of one window, not a significance claim.
    """
    if lo is None or hi is None:
        return "inconclusive"
    if isinstance(lo, float) and (np.isnan(lo) or np.isnan(hi)):
        return "inconclusive"
    if lo > 0:
        return "positive"
    if hi < 0:
        return "negative"
    return "inconclusive"


def rolling_category_horizon(
    events: pd.DataFrame,
    session_pos: pd.Series,
    horizons=HORIZONS,
    window_days: int = WINDOW_DAYS,
    stride_days: int = CATEGORY_STRIDE_DAYS,
) -> pd.DataFrame:
    """One row per (window, category, horizon).

    `_describe_group` and `_compare_vs_other` are called UNCHANGED on each
    window's event subset -- the per-window statistics and the same-date paired
    vs-OTHER difference are exactly Phase 1's, just computed on a time slice.
    (The paired same-date device is valid here because a category and OTHER
    contain DIFFERENT TICKERS on a shared date; Phase 2 documents why that same
    device collapses for a single ticker, which is why the stock-level path
    below does something different.)
    """
    windows = window_grid(events["date"].min(), events["date"].max(), window_days, stride_days)
    by_cat = {cat: g for cat, g in events.groupby("category", sort=True)}
    rows = []
    for w_start, w_end in windows:
        in_win = events[(events["date"] >= w_start) & (events["date"] <= w_end)]
        other_w = in_win[in_win["category"] == "OTHER"]
        for cat in sorted(by_cat):
            cat_w = in_win[in_win["category"] == cat]
            for h in horizons:
                desc = _describe_group(cat_w, h, session_pos)
                phase1_gate = desc.pop("sufficient_sample")
                window_sufficient = (
                    desc["n_events_valid"] >= ROLL_MIN_EVENTS
                    and desc["n_tickers"] >= ROLL_MIN_TICKERS
                )
                row = {
                    "window_start": w_start,
                    "window_end": w_end,
                    "category": cat,
                    **desc,
                    "phase1_gate_sufficient": phase1_gate,
                    "window_sufficient": window_sufficient,
                    "own_direction": _direction_label(desc["nw_ci95_lo"], desc["nw_ci95_hi"]),
                }
                if cat == "OTHER":
                    row.update({
                        "vs_other_n_overlapping_dates": None,
                        "vs_other_sufficient": False,
                        "vs_other_diff_mean": None,
                        "vs_other_se": None,
                        "vs_other_p_value": None,
                        "vs_other_ci95_lo": None,
                        "vs_other_ci95_hi": None,
                        "vs_other_direction": "n/a",
                    })
                else:
                    cmp = _compare_vs_other(cat_w, other_w, h, session_pos)
                    row.update({
                        "vs_other_n_overlapping_dates": cmp["n_overlapping_dates"],
                        "vs_other_sufficient": cmp["sufficient_sample"],
                        "vs_other_diff_mean": cmp["diff_mean"],
                        "vs_other_se": cmp["se"],
                        "vs_other_p_value": cmp["p_value"],
                        "vs_other_ci95_lo": cmp["ci95_lo"],
                        "vs_other_ci95_hi": cmp["ci95_hi"],
                        "vs_other_direction": _direction_label(cmp["ci95_lo"], cmp["ci95_hi"]),
                    })
                rows.append(row)
    df = pd.DataFrame(rows)
    if df.empty:
        # No window fits inside the data range (span < window_days). Return an
        # empty frame rather than failing in the BH step below.
        return df

    # BH across the whole rolling vs-OTHER family. Flagged exploratory in the
    # column name because overlapping windows violate the independence/PRDS
    # conditions BH assumes -- it is a screen here, not a licence to claim a
    # discovery. The defensible inferential statement is in the
    # temporal-change table, which uses disjoint windows.
    pvals = [
        p if (s and w) else None
        for p, s, w in zip(
            df["vs_other_p_value"], df["vs_other_sufficient"], df["window_sufficient"], strict=True
        )
    ]
    df["vs_other_significant_fdr_exploratory"] = benjamini_hochberg(pvals, FDR_Q)
    df["n_tests_in_rolling_family"] = int(
        sum(v is not None and not np.isnan(v) for v in pvals)
    )
    return df


# ===========================================================================
# 4. Rolling stock x category x horizon
# ===========================================================================


def _stock_describe_fast(
    ret: np.ndarray, pos: np.ndarray, horizon: int
) -> dict:
    """Per-window stats for ONE ticker-category, computed on numpy arrays of
    already-valid observations.

    Equivalent to `_describe_group` restricted to a single ticker, and pinned
    to it by test_stock_fast_path_matches_describe_group. The equivalence holds
    because the Phase 1 event table carries exactly one row per (ticker, date,
    category) -- verified, 0 duplicates -- so `_date_aggregate`'s "one point
    per date" step is an IDENTITY operation at stock level and the per-event
    series can go straight into the HAC routine. This path exists only for
    speed: the stock grid is ~2,100 ticker-categories x 28 windows x 6
    horizons, where per-cell pandas groupby overhead dominates everything else.
    """
    n = len(ret)
    out = {
        "n_events_valid": n,
        "mean_abnormal_return": float(ret.mean()) if n else None,
        "median_abnormal_return": float(np.median(ret)) if n else None,
        "p25_abnormal_return": float(np.quantile(ret, 0.25)) if n else None,
        "p75_abnormal_return": float(np.quantile(ret, 0.75)) if n else None,
        "std_abnormal_return": float(ret.std(ddof=1)) if n > 1 else None,
        "pct_positive": float((ret > 0).mean()) if n else None,
        "pct_negative": float((ret < 0).mean()) if n else None,
        "nw_lag_sessions": max(horizon - 1, 0),
    }
    if n >= 2:
        nw = newey_west_mean_ci_irregular(ret, pos, lag=max(horizon - 1, 0))
        out.update({
            "nw_mean": nw["mean"],
            "nw_se": nw["se"],
            "nw_ci95_lo": nw["ci95"][0],
            "nw_ci95_hi": nw["ci95"][1],
        })
    else:
        out.update({"nw_mean": out["mean_abnormal_return"], "nw_se": None,
                    "nw_ci95_lo": None, "nw_ci95_hi": None})
    return out


def _stock_evidence_tier(n: int) -> str:
    if n >= ROLL_STOCK_MIN_EVENTS:
        return "exploratory"  # the CEILING at stock level in a 365-day window
    if n >= ROLL_STOCK_EXPLORATORY_MIN:
        return "sparse"
    return "insufficient"


def _pair_max_window_events(dates: np.ndarray, window_days: int, threshold: int) -> bool:
    """Could this ticker-category EVER reach `threshold` events in any window?
    Cheap O(n log n) pre-screen that skips pairs no window can qualify."""
    if len(dates) < threshold:
        return False
    hi = np.searchsorted(dates, dates + np.timedelta64(window_days - 1, "D"), side="right")
    return bool((hi - np.arange(len(dates))).max() >= threshold)


def rolling_stock(
    events: pd.DataFrame,
    session_pos: pd.Series,
    horizons=HORIZONS,
    window_days: int = WINDOW_DAYS,
    stride_days: int = STOCK_STRIDE_DAYS,
    candidates: tuple[tuple[str, str, int], ...] = PHASE2_CANDIDATES,
) -> tuple[pd.DataFrame, dict]:
    """One row per (window, ticker, category, horizon) where the window holds
    enough events to say anything at all.

    Scope rules, documented because they decide what is absent from the output:
    * A ticker-category is considered only if SOME window could reach
      ROLL_STOCK_MIN_EVENTS (cheap pre-screen), or it is a Phase 2 candidate.
    * A window-cell is emitted only if it has >= ROLL_STOCK_EXPLORATORY_MIN
      valid events, or it belongs to a Phase 2 candidate -- whose rows are
      always emitted so their sparsity is visible as data rather than as a
      silent omission. Suppressed counts are returned in the scope dict, so
      nothing is hidden by this filter.
    * No rolling vs-OTHER test at stock level: a single ticker's 365-day window
      holds single-digit category events, far below what the Phase 2 two-sample
      comparison needs, so computing it would only manufacture noise.
    """
    windows = window_grid(events["date"].min(), events["date"].max(), window_days, stride_days)
    win_starts = np.array([w[0] for w in windows], dtype="datetime64[ns]")
    win_ends = np.array([w[1] for w in windows], dtype="datetime64[ns]")
    forced_pairs = {(t, c) for t, c, _ in candidates}

    rows = []
    n_pairs_screened = n_pairs_kept = n_cells_suppressed = 0
    for (ticker, cat), g in events.groupby(["ticker", "category"], sort=True):
        n_pairs_screened += 1
        is_forced = (ticker, cat) in forced_pairs
        g = g.sort_values("date")
        dates = g["date"].to_numpy(dtype="datetime64[ns]")
        if not is_forced and not _pair_max_window_events(
            dates, window_days, ROLL_STOCK_MIN_EVENTS
        ):
            continue
        n_pairs_kept += 1
        pos_all = session_positions(g["date"], session_pos).astype(float)
        # Window membership depends only on the event dates, not the horizon,
        # so resolve each window to a [lo, hi) slice ONCE per pair.
        lo_idx = np.searchsorted(dates, win_starts, side="left")
        hi_idx = np.searchsorted(dates, win_ends, side="right")
        for h in horizons:
            ret_all = g[abnormal_ret_col(h)].to_numpy(dtype=float)
            valid_all = g[valid_col(h)].to_numpy(dtype=bool)
            for wi in range(len(win_starts)):
                lo, hi = int(lo_idx[wi]), int(hi_idx[wi])
                vmask = valid_all[lo:hi]
                n = int(vmask.sum())
                if n < ROLL_STOCK_EXPLORATORY_MIN and not is_forced:
                    n_cells_suppressed += 1
                    continue
                desc = _stock_describe_fast(
                    ret_all[lo:hi][vmask], pos_all[lo:hi][vmask], h
                )
                rows.append({
                    "window_start": pd.Timestamp(win_starts[wi]),
                    "window_end": pd.Timestamp(win_ends[wi]),
                    "ticker": ticker,
                    "category": cat,
                    "horizon": h,
                    "n_events_in_window_total": hi - lo,
                    **desc,
                    "exploratory_horizon": h in EXPLORATORY_HORIZONS,
                    "window_sufficient": n >= ROLL_STOCK_MIN_EVENTS,
                    "evidence_tier": _stock_evidence_tier(n),
                    "phase2_candidate": is_forced,
                    "own_direction": _direction_label(desc["nw_ci95_lo"], desc["nw_ci95_hi"]),
                })
    scope = {
        "stride_days": stride_days,
        "n_windows": len(windows),
        "n_ticker_category_pairs_screened": n_pairs_screened,
        "n_ticker_category_pairs_analysed": n_pairs_kept,
        "n_window_cells_emitted": len(rows),
        "n_window_cells_suppressed_below_min": n_cells_suppressed,
        "emit_rule": (
            f">= {ROLL_STOCK_EXPLORATORY_MIN} valid events in the window, or a Phase 2 "
            "candidate (always emitted). Suppressed cells are counted here, not hidden."
        ),
        "vs_other_note": (
            "No rolling vs-OTHER test at stock level -- a single ticker's 365-day window "
            "holds single-digit category events, far below the Phase 2 two-sample "
            "requirement."
        ),
    }
    return pd.DataFrame(rows), scope


# ===========================================================================
# 5. Temporal change: strengthen / weaken / reverse / consistent / insufficient
# ===========================================================================


def _two_window_diff(late: pd.Series, early: pd.Series, mean_c: str, se_c: str) -> dict:
    """late - early for two NON-OVERLAPPING windows.

    The windows share no events, so each side's HAC-corrected SE can be
    combined as se = sqrt(se_late^2 + se_early^2) -- the same two-sample device
    Phase 2 uses for disjoint date sets, and the reason the comparison is
    restricted to non-overlapping windows in the first place.
    """
    m_l, m_e = late[mean_c], early[mean_c]
    s_l, s_e = late[se_c], early[se_c]
    if pd.isna(m_l) or pd.isna(m_e):
        return {"diff": None, "se": None, "t_stat": None, "p_value": None,
                "ci95_lo": None, "ci95_hi": None}
    diff = float(m_l - m_e)
    if pd.isna(s_l) or pd.isna(s_e):
        return {"diff": diff, "se": None, "t_stat": None, "p_value": None,
                "ci95_lo": None, "ci95_hi": None}
    se = float(np.sqrt(float(s_l) ** 2 + float(s_e) ** 2))
    if se <= 0:
        return {"diff": diff, "se": se, "t_stat": None, "p_value": None,
                "ci95_lo": diff, "ci95_hi": diff}
    t_stat = diff / se
    df_eff = max(min(int(late["n_dates_or_events"]), int(early["n_dates_or_events"])) - 1, 1)
    return {
        "diff": diff,
        "se": se,
        "t_stat": float(t_stat),
        "p_value": float(2 * (1 - stats.t.cdf(abs(t_stat), df=df_eff))),
        "ci95_lo": diff - 1.96 * se,
        "ci95_hi": diff + 1.96 * se,
    }


def _label_change(early: pd.Series, late: pd.Series, mean_c: str, lo_c: str, hi_c: str) -> str:
    """Descriptive label from effect SIZES and their intervals -- deliberately
    not from p-values, and deliberately separate from whether the change is
    statistically established (see `change_established`)."""
    m_e, m_l = early[mean_c], late[mean_c]
    if pd.isna(m_e) or pd.isna(m_l):
        return "inconclusive"
    m_e, m_l = float(m_e), float(m_l)
    e_excl = _direction_label(early[lo_c], early[hi_c]) != "inconclusive"
    l_excl = _direction_label(late[lo_c], late[hi_c]) != "inconclusive"
    # A sign change is handled BEFORE any magnitude ratio: comparing |late| to
    # |early| across a sign boundary would call a flip from -0.09pp to +0.05pp
    # "weakened", which misdescribes it. A flip is only "reversed" when both
    # sides individually exclude zero; otherwise the direction is simply not
    # pinned down and says so.
    if m_e != 0 and m_l != 0 and np.sign(m_e) != np.sign(m_l):
        return "reversed" if (e_excl and l_excl) else "direction_unstable"
    if abs(m_e) <= 0:
        return "strengthened" if abs(m_l) > 0 else "broadly_consistent"
    ratio = abs(m_l) / abs(m_e)
    if ratio >= STRENGTHEN_RATIO:
        return "strengthened"
    if ratio <= WEAKEN_RATIO:
        return "weakened"
    return "broadly_consistent"


def classify_temporal_change(
    rolling: pd.DataFrame, horizons=HORIZONS, window_days: int = WINDOW_DAYS
) -> pd.DataFrame:
    """Per (category, horizon, metric): has the effect strengthened, weakened,
    reversed, stayed broadly consistent, or is there too little to tell?

    Two metrics are reported side by side, long-format:
    * `vs_other_diff` -- the category's same-date paired difference against
      OTHER news. Primary, because differencing against other news on the same
      dates removes that window's market-wide return level.
    * `own_nw_mean` -- the category's own HAC mean abnormal return. The only
      metric available for OTHER itself.

    Only NON-OVERLAPPING, sufficiently-sampled windows are compared, and the
    earliest and latest such windows form the two sides. `change_label` is a
    descriptive read of effect sizes; `change_established` additionally
    requires a BH-corrected significant difference AND non-overlapping 95%
    intervals, and is the only column that should be read as a finding.
    """
    specs = [
        ("vs_other_diff", "vs_other_diff_mean", "vs_other_se", "vs_other_ci95_lo",
         "vs_other_ci95_hi", "vs_other_n_overlapping_dates", "vs_other_sufficient"),
        ("own_nw_mean", "nw_mean", "nw_se", "nw_ci95_lo", "nw_ci95_hi", "n_dates", None),
    ]
    rows = []
    for metric, mean_c, se_c, lo_c, hi_c, n_c, extra_suff in specs:
        for cat in sorted(rolling["category"].unique()):
            if metric == "vs_other_diff" and cat == "OTHER":
                continue
            for h in horizons:
                sub = rolling[
                    (rolling["category"] == cat) & (rolling["horizon"] == h)
                ].sort_values("window_end").reset_index(drop=True)
                if sub.empty:
                    continue
                keep = non_overlapping_subset(list(sub["window_end"]), window_days)
                nonov = sub.loc[keep].copy()
                ok = nonov["window_sufficient"]
                if extra_suff is not None:
                    ok = ok & nonov[extra_suff].fillna(False).astype(bool)
                ok = ok & nonov[mean_c].notna()
                usable = nonov[ok].copy()
                usable["n_dates_or_events"] = usable[n_c]

                recent = sub.iloc[-1]
                recent_ok = bool(recent["window_sufficient"]) and (
                    extra_suff is None or bool(recent[extra_suff])
                )
                base = {
                    "metric": metric,
                    "category": cat,
                    "horizon": h,
                    "exploratory_horizon": h in EXPLORATORY_HORIZONS,
                    "n_windows_total": int(len(sub)),
                    "n_windows_sufficient": int(sub["window_sufficient"].sum()),
                    "n_nonoverlapping_windows": int(len(nonov)),
                    "n_nonoverlapping_sufficient": int(len(usable)),
                    "recent_window_end": recent["window_end"],
                    "recent_window_sufficient": recent_ok,
                    "recent_n_events_valid": int(recent["n_events_valid"]),
                    "recent_estimate": (
                        None if pd.isna(recent[mean_c]) else float(recent[mean_c])
                    ),
                }
                if not recent_ok:
                    base["recent_status"] = "insufficient_recent_data"
                else:
                    base["recent_status"] = "assessable"

                if len(usable) < TREND_MIN_WINDOWS:
                    rows.append({
                        **base,
                        "change_label": "insufficient_windows",
                        "early_window_end": None, "late_window_end": None,
                        "early_estimate": None, "late_estimate": None,
                        "diff_late_minus_early": None, "diff_se": None,
                        "diff_t_stat": None, "diff_p_value": None,
                        "diff_ci95_lo": None, "diff_ci95_hi": None,
                        "intervals_disjoint": False,
                    })
                    continue

                early, late = usable.iloc[0], usable.iloc[-1]
                d = _two_window_diff(late, early, mean_c, se_c)
                disjoint = bool(
                    pd.notna(early[lo_c]) and pd.notna(late[lo_c])
                    and (float(early[hi_c]) < float(late[lo_c])
                         or float(late[hi_c]) < float(early[lo_c]))
                )
                rows.append({
                    **base,
                    "change_label": _label_change(early, late, mean_c, lo_c, hi_c),
                    "early_window_end": early["window_end"],
                    "late_window_end": late["window_end"],
                    "early_estimate": float(early[mean_c]),
                    "late_estimate": float(late[mean_c]),
                    "diff_late_minus_early": d["diff"],
                    "diff_se": d["se"],
                    "diff_t_stat": d["t_stat"],
                    "diff_p_value": d["p_value"],
                    "diff_ci95_lo": d["ci95_lo"],
                    "diff_ci95_hi": d["ci95_hi"],
                    "intervals_disjoint": disjoint,
                })
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    # BH over the temporal-change family. Unlike the rolling screen above, the
    # two sides of every test here are disjoint windows, so this is the one
    # place a corrected claim is defensible.
    df["diff_significant_fdr"] = benjamini_hochberg(list(df["diff_p_value"]), FDR_Q)
    df["change_established"] = df["diff_significant_fdr"] & df["intervals_disjoint"]
    df["n_tests_in_change_family"] = int(df["diff_p_value"].notna().sum())
    return df


# ===========================================================================
# 6. Phase 2 candidate concentration: is the historical hit a few events?
# ===========================================================================


def candidate_concentration(
    events: pd.DataFrame,
    session_pos: pd.Series,
    candidates: tuple[tuple[str, str, int], ...] = PHASE2_CANDIDATES,
    window_days: int = WINDOW_DAYS,
) -> list[dict]:
    """For each Phase 2 candidate: how much of its full-history estimate rests
    on a handful of events, and is there any window dense enough to re-check it?

    Leave-one-out is the direct test of "dominated by a few events": if dropping
    the single most extreme observation moves the mean by a large fraction, the
    estimate is not a stable property of the category for that stock.
    """
    out = []
    for ticker, cat, horizon in candidates:
        sub = events[(events["ticker"] == ticker) & (events["category"] == cat)]
        ret_c, valid_c = abnormal_ret_col(horizon), valid_col(horizon)
        v = sub[sub[valid_c]].sort_values("date")
        r = v[ret_c].to_numpy(dtype=float)
        n = len(r)
        rec: dict = {
            "ticker": ticker,
            "category": cat,
            "horizon": horizon,
            "sector": (sub["sector"].dropna().iloc[0] if sub["sector"].notna().any() else None),
            "n_events_valid_full_history": n,
            "first_event_date": str(v["date"].min().date()) if n else None,
            "last_event_date": str(v["date"].max().date()) if n else None,
            "events_per_year": (
                round(n / max((v["date"].max() - v["date"].min()).days / 365.25, 1e-9), 2)
                if n > 1 else None
            ),
        }
        if n == 0:
            out.append({**rec, "verdict": "no valid events"})
            continue
        mean_all = float(r.mean())
        extreme_i = int(np.argmax(np.abs(r - mean_all)))
        r_drop1 = np.delete(r, extreme_i)
        mean_drop1 = float(r_drop1.mean()) if len(r_drop1) else None
        # drop the 3 most extreme
        order = np.argsort(-np.abs(r - mean_all))
        r_drop3 = np.delete(r, order[:3])
        rec.update({
            "mean_abnormal_return": mean_all,
            "mean_excl_most_extreme_event": mean_drop1,
            "pct_change_in_mean_excl_most_extreme": (
                round(abs(mean_drop1 - mean_all) / abs(mean_all) * 100, 1)
                if mean_drop1 is not None and mean_all != 0 else None
            ),
            "mean_excl_3_most_extreme": float(r_drop3.mean()) if len(r_drop3) else None,
            "largest_single_event_abnormal_return": float(r[extreme_i]),
            "largest_single_event_date": str(v["date"].iloc[extreme_i].date()),
            "largest_single_event_share_of_sum_pct": (
                round(abs(r[extreme_i]) / np.abs(r).sum() * 100, 1)
                if np.abs(r).sum() > 0 else None
            ),
        })
        # densest 365-day window, and the most recent one
        dates = v["date"].to_numpy(dtype="datetime64[ns]")
        hi = np.searchsorted(dates, dates + np.timedelta64(window_days - 1, "D"), side="right")
        counts = hi - np.arange(len(dates))
        best_i = int(np.argmax(counts))
        last_news = pd.Timestamp(events["date"].max())
        recent_start = last_news - pd.Timedelta(days=window_days - 1)
        n_recent = int(
            (
                (dates >= np.datetime64(recent_start))
                & (dates <= np.datetime64(last_news))
            ).sum()
        )
        rec.update({
            "max_events_in_any_365d_window": int(counts[best_i]),
            "densest_window_start": str(pd.Timestamp(dates[best_i]).date()),
            "events_in_most_recent_365d_window": n_recent,
            "rolling_min_events": ROLL_STOCK_MIN_EVENTS,
            "any_window_meets_rolling_min": bool(counts.max() >= ROLL_STOCK_MIN_EVENTS),
        })
        dominated = (
            rec["pct_change_in_mean_excl_most_extreme"] is not None
            and rec["pct_change_in_mean_excl_most_extreme"] >= 25.0
        ) or n < 30
        rec["event_dominated"] = bool(dominated)
        if not rec["any_window_meets_rolling_min"]:
            rec["verdict"] = (
                f"INSUFFICIENT EVIDENCE for rolling assessment: densest 365-day window holds "
                f"{int(counts.max())} valid events, below the {ROLL_STOCK_MIN_EVENTS} minimum, "
                f"and the most recent window holds {n_recent}. Recent stability cannot be "
                f"evaluated at this horizon."
            )
        elif n_recent < ROLL_STOCK_MIN_EVENTS:
            rec["verdict"] = (
                f"Historically assessable (densest 365-day window holds {int(counts.max())} "
                f"events) but RECENT stability is NOT assessable: the most recent 365-day "
                f"window holds only {n_recent} valid event(s), below the "
                f"{ROLL_STOCK_MIN_EVENTS} minimum."
                + (
                    " The full-history estimate is also sensitive to individual events "
                    "(see leave-one-out)." if dominated else ""
                )
            )
        elif dominated:
            rec["verdict"] = (
                "Rolling assessment possible in the densest windows, but the full-history "
                "estimate is sensitive to individual events (see leave-one-out)."
            )
        else:
            rec["verdict"] = "Rolling assessment possible and not dominated by single events."
        out.append(rec)
    return out


# ===========================================================================
# 7. Optional figures (skipped cleanly when matplotlib is absent)
# ===========================================================================


def make_figures(rolling: pd.DataFrame, out_dir: Path, horizons=HORIZONS) -> list[str]:
    """Rolling mean + HAC interval time series for EARNINGS_RESULTS (the
    strongest aggregate Phase 1 association) at every horizon.

    matplotlib is NOT a declared project dependency, so this is best-effort:
    if the import fails the report still stands on its in-text numeric series
    and ASCII sparklines, and the run does not fail.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return []

    fig_dir = Path(out_dir) / "phase3_figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for cat in ("EARNINGS_RESULTS",):
        fig, axes = plt.subplots(len(horizons), 1, figsize=(10, 2.1 * len(horizons)), sharex=True)
        axes = np.atleast_1d(axes)
        for ax, h in zip(axes, horizons, strict=True):
            sub = rolling[
                (rolling["category"] == cat) & (rolling["horizon"] == h)
            ].sort_values("window_end")
            ok = sub[sub["window_sufficient"]]
            ax.axhline(0, color="black", lw=0.8, ls=":")
            if len(ok):
                ax.plot(ok["window_end"], ok["vs_other_diff_mean"] * 100, lw=1.4,
                        color="#1f4e79", label="vs OTHER")
                ax.fill_between(
                    ok["window_end"], ok["vs_other_ci95_lo"] * 100,
                    ok["vs_other_ci95_hi"] * 100, alpha=0.2, color="#1f4e79",
                )
            ax.set_ylabel(f"{h}d\n(pp)", fontsize=8)
            ax.tick_params(labelsize=7)
        axes[0].set_title(
            f"{cat}: rolling 365-day difference vs OTHER news (percentage points)\n"
            "shaded = 95% HAC interval; gaps = window below minimum sample",
            fontsize=9,
        )
        fig.tight_layout()
        path = fig_dir / f"phase3_rolling_{cat.lower()}_vs_other.png"
        fig.savefig(path, dpi=110)
        plt.close(fig)
        written.append(str(path))
    return written


# ===========================================================================
# 8. Orchestration
# ===========================================================================


def run(
    events_path: str | Path = DEFAULT_EVENTS,
    panel_dir: str | Path = DEFAULT_PANEL_DIR,
    out_dir: str | Path = DEFAULT_OUT_DIR,
    version: str = OUTPUT_VERSION,
    horizons: tuple[int, ...] = HORIZONS,
    category_stride_days: int = CATEGORY_STRIDE_DAYS,
    stock_stride_days: int = STOCK_STRIDE_DAYS,
    window_days: int = WINDOW_DAYS,
    as_of=None,
    figures: bool = True,
) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    events = pd.read_parquet(events_path)
    panel = load_curated(panel_dir)
    sessions = pd.DatetimeIndex(sorted(panel["date"].unique()))
    session_pos = pd.Series(np.arange(len(sessions)), index=sessions)

    freshness = data_freshness(events, horizons, as_of=as_of)

    rolling = rolling_category_horizon(
        events, session_pos, horizons, window_days, category_stride_days
    )
    rolling_path = out_dir / f"phase3_rolling_category_horizon_{version}.parquet"
    rolling.to_parquet(rolling_path, index=False)

    stock, stock_scope = rolling_stock(
        events, session_pos, horizons, window_days, stock_stride_days
    )
    stock_path = out_dir / f"phase3_rolling_stock_{version}.parquet"
    stock.to_parquet(stock_path, index=False)

    change = classify_temporal_change(rolling, horizons, window_days)
    change_path = out_dir / f"phase3_temporal_change_{version}.parquet"
    change.to_parquet(change_path, index=False)

    candidates = candidate_concentration(events, session_pos, PHASE2_CANDIDATES, window_days)
    figure_paths = make_figures(rolling, out_dir, horizons) if figures else []

    windows = window_grid(
        events["date"].min(), events["date"].max(), window_days, category_stride_days
    )
    label_counts = (
        change.groupby(["metric", "change_label"]).size().reset_index(name="n").to_dict("records")
        if not change.empty else []
    )
    coverage = {
        "purpose": (
            "Phase 3: rolling 365-calendar-day news-impact intelligence. Asks whether the "
            "Phase 1/Phase 2 historical relationships hold steady over time or change. "
            "Associations, not causation, and NOT real-time intelligence."
        ),
        "inputs": {"events": str(events_path), "panel_dir": str(panel_dir)},
        "freshness": freshness,
        "window_design": {
            "window_days": window_days,
            "category_stride_days": category_stride_days,
            "stock_stride_days": stock_stride_days,
            "n_category_windows": len(windows),
            "first_window": f"{windows[0][0].date()}..{windows[0][1].date()}" if windows else None,
            "last_window": f"{windows[-1][0].date()}..{windows[-1][1].date()}" if windows else None,
            "anchoring": (
                "Windows are anchored at the last event date and stepped backwards, so the "
                "final window is exactly the most recent 365 days of available news."
            ),
            "overlap_note": (
                (
                    f"Consecutive category windows overlap by "
                    f"{window_days - category_stride_days} of {window_days} days and are NOT "
                    "independent. Every window-to-window comparison uses only a "
                    "non-overlapping subset."
                )
                if category_stride_days < window_days
                else (
                    f"Category stride ({category_stride_days}d) is at least the window length "
                    f"({window_days}d), so consecutive windows are already disjoint."
                )
            ),
        },
        "minimum_samples": {
            "category_window": (
                f">= {ROLL_MIN_EVENTS} valid events AND >= {ROLL_MIN_TICKERS} tickers"
            ),
            "stock_window_exploratory_ceiling": f">= {ROLL_STOCK_MIN_EVENTS} valid events",
            "stock_window_emitted_at": f">= {ROLL_STOCK_EXPLORATORY_MIN} valid events",
            "trend_min_nonoverlapping_windows": TREND_MIN_WINDOWS,
            "note": (
                "Thresholds were fixed before any ranking. A window below them is reported "
                "as insufficient, never as evidence of no impact."
            ),
        },
        "multiple_testing": {
            "rolling_vs_other_family": (
                f"Benjamini-Hochberg at q={FDR_Q} across all sufficient rolling vs-OTHER "
                "tests, reported as `vs_other_significant_fdr_exploratory`. EXPLORATORY: "
                "overlapping windows break BH's independence/PRDS assumptions."
            ),
            "temporal_change_family": (
                f"Benjamini-Hochberg at q={FDR_Q} across the temporal-change tests, whose two "
                "sides are DISJOINT windows. `change_established` additionally requires "
                "non-overlapping 95% intervals and is the only column to read as a finding."
            ),
            "n_rolling_tests": int(rolling["n_tests_in_rolling_family"].iloc[0])
            if len(rolling) else 0,
            "n_change_tests": int(change["n_tests_in_change_family"].iloc[0])
            if not change.empty else 0,
        },
        "rolling_category_rows": int(len(rolling)),
        "rolling_stock_scope": stock_scope,
        "change_label_counts": label_counts,
        "n_change_established": int(change["change_established"].sum())
        if not change.empty else 0,
        "phase2_candidates": candidates,
        "caveats": [
            "NOT real-time: no data was refreshed in this phase; the archive ends "
            f"{freshness['last_news_date']} ({freshness['news_staleness_years_vs_as_of']} years "
            f"before {freshness['as_of']}).",
            "Long horizons (60/126/252 sessions) inside a 365-day window have very few "
            "effectively independent outcome periods; read them as descriptive series.",
            "Sector labels inherited from Phase 1 are CURRENT GICS membership, not "
            "point-in-time historical membership.",
            "Historical association only -- not causal, not a guarantee of future response.",
        ],
        "outputs": {
            "rolling_category_horizon": str(rolling_path),
            "rolling_stock": str(stock_path),
            "temporal_change": str(change_path),
        },
        "figures": figure_paths,
    }

    freshness_path = out_dir / f"phase3_freshness_{version}.json"
    freshness_path.write_text(json.dumps(coverage, indent=1, default=str), encoding="utf-8")
    coverage["freshness_path"] = str(freshness_path)

    report_path = out_dir / f"phase3_report_{version}.md"
    report_path.write_text(
        render_markdown_report(rolling, stock, change, candidates, coverage), encoding="utf-8"
    )
    coverage["report_path"] = str(report_path)
    return coverage


# ===========================================================================
# 9. Markdown report
# ===========================================================================

_SPARK_LEVELS = "_.-=+*#"  # ASCII only: these reports are read in plain terminals


def _sparkline(values) -> str:
    """Compact ASCII time-series summary. Gaps (insufficient windows) show as
    a space, so a reader can see WHERE evidence is missing, not just the shape."""
    vals = [None if v is None or pd.isna(v) else float(v) for v in values]
    present = [v for v in vals if v is not None]
    if not present:
        return ""
    lo, hi = min(present), max(present)
    if hi == lo:
        mid = _SPARK_LEVELS[len(_SPARK_LEVELS) // 2]
        return "".join(mid if v is not None else " " for v in vals)
    out = []
    for v in vals:
        if v is None:
            out.append(" ")
        else:
            k = int((v - lo) / (hi - lo) * (len(_SPARK_LEVELS) - 1))
            out.append(_SPARK_LEVELS[min(max(k, 0), len(_SPARK_LEVELS) - 1)])
    return "".join(out)


def _fmt_pct(x) -> str:
    return "n/a" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x * 100:+.2f}%"


def _stride_label(days: int) -> str:
    """Plain-language cadence for a stride, derived rather than hardcoded so an
    overridden --stride never gets described with the default's wording."""
    for cutoff, label in ((40, "~monthly"), (115, "~quarterly"), (220, "~semi-annual"),
                          (400, "~annual")):
        if days <= cutoff:
            return label
    return f"every {days} days"


def render_markdown_report(
    rolling: pd.DataFrame,
    stock: pd.DataFrame,
    change: pd.DataFrame,
    candidates: list[dict],
    coverage: dict,
) -> str:
    f = coverage["freshness"]
    wd = coverage["window_design"]
    lines = [
        "# Phase 3: Rolling 365-Day News-Impact Intelligence",
        "",
        "Do the Phase 1 / Phase 2 historical relationships between news categories and "
        "benchmark-adjusted returns hold steady over time, or do they strengthen, weaken, "
        "or reverse? **Historical association, not causation.**",
        "",
        "## Data freshness (read this first)",
        "",
        "- **This is NOT real-time intelligence.** No data was refreshed in Phase 3.",
        f"- News archive covers **{f['first_news_date']} to {f['last_news_date']}** "
        f"({f['n_events_total']:,} events, {f['n_tickers']} tickers).",
        f"- As of **{f['as_of']}**, the newest news is "
        f"**{f['news_staleness_days_vs_as_of']:,} days "
        f"({f['news_staleness_years_vs_as_of']} years) old**.",
        "- Because an event needs its FULL forward window to be valid, the newest *assessable* "
        "event date recedes as the horizon lengthens:",
        "",
        "| horizon | last valid event date | valid events | invalid | % invalid | "
        "frontier lost vs last news |",
        "|---|---|---|---|---|---|",
    ]
    for key, ph in f["per_horizon"].items():
        lines.append(
            f"| {key} | {ph['last_valid_event_date']} | {ph['n_events_valid']:,} | "
            f"{ph['n_events_invalid']:,} | {ph['pct_invalid']}% | "
            f"{ph['days_of_outcome_frontier_lost']} days |"
        )
    h_keys = list(f["per_horizon"])
    h_short, h_long = h_keys[0], h_keys[-1]
    lines += [
        "",
        f"So \"the most recent window\" means something different at each horizon: at "
        f"{h_short} it ends {f['per_horizon'][h_short]['last_valid_event_date']}, but at "
        f"{h_long} it cannot extend past "
        f"{f['per_horizon'][h_long]['last_valid_event_date']}.",
        "",
        "## Window design",
        "",
        f"- {wd['window_days']}-calendar-day windows; category level stepped "
        f"**{wd['category_stride_days']} days** ({_stride_label(wd['category_stride_days'])}), "
        f"stock level **{wd['stock_stride_days']} days** "
        f"({_stride_label(wd['stock_stride_days'])}).",
        f"- {wd['n_category_windows']} category windows, {wd['first_window']} to "
        f"{wd['last_window']}.",
        f"- {wd['anchoring']}",
        f"- {wd['overlap_note']}",
        f"- Minimum sample, category window: {coverage['minimum_samples']['category_window']}.",
        "",
        "## Temporal change by category and horizon",
        "",
        "`change_label` is a descriptive read of effect sizes and their intervals. "
        "`established` is the strict column: Benjamini-Hochberg-corrected significance on "
        "disjoint windows AND non-overlapping 95% intervals. Primary metric is the same-date "
        "paired difference versus OTHER news.",
        "",
        "| category | horizon | label | established | early | late | diff (late-early) | "
        "95% CI of diff | non-overlap windows used | recent window |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    primary = change[change["metric"] == "vs_other_diff"] if not change.empty else change
    for _, r in primary.sort_values(["category", "horizon"]).iterrows():
        est = "**yes**" if r.get("change_established") else "no"
        expl = " *(exploratory h)*" if r["exploratory_horizon"] else ""
        ci = (
            f"[{_fmt_pct(r['diff_ci95_lo'])}, {_fmt_pct(r['diff_ci95_hi'])}]"
            if pd.notna(r["diff_ci95_lo"]) else "n/a"
        )
        lines.append(
            f"| {r['category']} | {r['horizon']}d{expl} | {r['change_label']} | {est} | "
            f"{_fmt_pct(r['early_estimate'])} | {_fmt_pct(r['late_estimate'])} | "
            f"{_fmt_pct(r['diff_late_minus_early'])} | {ci} | "
            f"{r['n_nonoverlapping_sufficient']}/{r['n_nonoverlapping_windows']} | "
            f"{r['recent_status']} |"
        )

    lines += [
        "",
        f"**{coverage['n_change_established']} of {len(change)} temporal-change tests are "
        "statistically established** under that strict definition. Everything else is either "
        "descriptive or under-sampled.",
        "",
    ]

    # Why the primary metric is differenced against OTHER, demonstrated from
    # this run's own numbers rather than asserted.
    if not change.empty:
        own = change[change["metric"] == "own_nw_mean"]
        est_own = own[own["change_established"]]
        est_prim = primary[primary["change_established"]]
        long_h = max(own["horizon"])
        own_at_long = int((est_own["horizon"] == long_h).sum())
        shared = set(zip(est_own["category"], est_own["horizon"], strict=True)) & set(
            zip(est_prim["category"], est_prim["horizon"], strict=True)
        )
        lines += [
            "### Why the primary metric is differenced against OTHER news",
            "",
            f"Established changes: **{len(est_own)} on the raw own-category mean**, "
            f"**{len(est_prim)} on the difference versus OTHER**, with "
            f"**{len(shared)} category/horizon cell(s) established on both**. "
            f"{own_at_long} of the {len(est_own)} own-mean ones sit at the single longest "
            f"horizon ({long_h} sessions) -- which is the tell.",
            "",
        ]
        long_own = own[(own["horizon"] == long_h)]
        rose = long_own[long_own["late_estimate"] > long_own["early_estimate"]]
        other_row = long_own[long_own["category"] == "OTHER"]
        if len(rose) and len(other_row):
            o = other_row.iloc[0]
            lines += [
                f"At the {int(o['horizon'])}-session horizon, {len(rose)} of {len(long_own)} "
                f"categories show a higher own-mean in the late window -- **including OTHER "
                f"itself** ({_fmt_pct(o['early_estimate'])} to {_fmt_pct(o['late_estimate'])}). "
                "When the baseline category moves with everything else, the common cause is "
                "market-wide drift over the late window, not a change in how any one kind of "
                "news is followed by returns. Differencing each category against OTHER news on "
                "the SAME dates removes that common component, which is why own-mean "
                "\"strengthening\" at long horizons largely disappears in the primary column "
                "above. Read the own-mean rows as context, not as findings.",
                "",
            ]

    lines += [
        "## Rolling series (ASCII sparklines, oldest window on the left)",
        "",
        f"Levels `{_SPARK_LEVELS}` run low to high within each row; a blank means that window "
        "fell below the minimum sample and was deliberately left empty rather than "
        "interpolated. Each row is scaled to its own min/max, so compare SHAPE across a row, "
        "not height between rows.",
        "",
        "| category | horizon | rolling vs-OTHER difference | min | max | windows ok |",
        "|---|---|---|---|---|---|",
    ]
    for cat in sorted(rolling["category"].unique()):
        if cat == "OTHER":
            continue
        for h in sorted(rolling["horizon"].unique()):
            sub = rolling[
                (rolling["category"] == cat) & (rolling["horizon"] == h)
            ].sort_values("window_end")
            vals = [
                r["vs_other_diff_mean"] if (r["window_sufficient"] and r["vs_other_sufficient"])
                else None
                for _, r in sub.iterrows()
            ]
            present = [v for v in vals if v is not None and not pd.isna(v)]
            if not present:
                continue
            lines.append(
                f"| {cat} | {h}d | `{_sparkline(vals)}` | {_fmt_pct(min(present))} | "
                f"{_fmt_pct(max(present))} | {len(present)}/{len(vals)} |"
            )

    lines += [
        "",
        "## Earnings news across all six horizons",
        "",
        "Phase 1 found the strongest aggregate association for EARNINGS_RESULTS, so it gets "
        "the closest look here.",
        "",
        "| horizon | windows ok | label | established | earliest vs-OTHER | latest vs-OTHER | "
        "recent window status |",
        "|---|---|---|---|---|---|---|",
    ]
    for h in sorted(rolling["horizon"].unique()):
        cr = primary[(primary["category"] == "EARNINGS_RESULTS") & (primary["horizon"] == h)]
        sub = rolling[
            (rolling["category"] == "EARNINGS_RESULTS") & (rolling["horizon"] == h)
        ]
        n_ok = int((sub["window_sufficient"] & sub["vs_other_sufficient"].fillna(False)).sum())
        if cr.empty:
            lines.append(f"| {h}d | {n_ok}/{len(sub)} | n/a | no | n/a | n/a | n/a |")
            continue
        r = cr.iloc[0]
        lines.append(
            f"| {h}d | {n_ok}/{len(sub)} | {r['change_label']} | "
            f"{'**yes**' if r['change_established'] else 'no'} | "
            f"{_fmt_pct(r['early_estimate'])} | {_fmt_pct(r['late_estimate'])} | "
            f"{r['recent_status']} |"
        )

    lines += [
        "",
        "## The two Phase 2 candidates",
        "",
        "Phase 2 flagged these as its only adequately-supported, FDR-significant stock-level "
        "results. The question here is whether those estimates rest on a few events, and "
        "whether any 365-day window is dense enough to re-check them.",
        "",
    ]
    for c in candidates:
        lines += [
            f"### {c['ticker']} / {c.get('sector')} / {c['category']} / {c['horizon']}d",
            "",
            f"- Full history: **{c['n_events_valid_full_history']} valid events** "
            f"({c['first_event_date']} to {c['last_event_date']}), "
            f"~{c['events_per_year']} events/year.",
            f"- Mean abnormal return {_fmt_pct(c.get('mean_abnormal_return'))}; excluding its "
            f"single most extreme event {_fmt_pct(c.get('mean_excl_most_extreme_event'))} "
            f"(**{c.get('pct_change_in_mean_excl_most_extreme')}% change**); excluding the 3 "
            f"most extreme {_fmt_pct(c.get('mean_excl_3_most_extreme'))}.",
            f"- Largest single event: {_fmt_pct(c.get('largest_single_event_abnormal_return'))} "
            f"on {c.get('largest_single_event_date')}, "
            f"{c.get('largest_single_event_share_of_sum_pct')}% of total absolute return.",
            f"- Densest 365-day window holds **{c.get('max_events_in_any_365d_window')} events** "
            f"(from {c.get('densest_window_start')}); most recent 365-day window holds "
            f"**{c.get('events_in_most_recent_365d_window')}**. Rolling minimum is "
            f"{c.get('rolling_min_events')}.",
            f"- Event-dominated: **{c.get('event_dominated')}**.",
            f"- **{c['verdict']}**",
            "",
        ]
        # Window-by-window series: the direct answer to "is there enough data to
        # evaluate recent stability?" -- counts first, estimates second.
        cs = stock[
            (stock["ticker"] == c["ticker"])
            & (stock["category"] == c["category"])
            & (stock["horizon"] == c["horizon"])
        ].sort_values("window_end")
        if not cs.empty:
            n_suff = int(cs["window_sufficient"].sum())
            excl_zero = int(
                (cs["window_sufficient"] & (cs["own_direction"] != "inconclusive")).sum()
            )
            vals = [
                r["nw_mean"] if r["window_sufficient"] else None for _, r in cs.iterrows()
            ]
            lines += [
                f"- Rolling windows: **{n_suff} of {len(cs)}** reach the "
                f"{ROLL_STOCK_MIN_EVENTS}-event minimum; of those, **{excl_zero}** have a 95% "
                f"interval excluding zero.",
                f"- Sufficient-window series (blank = below minimum): `{_sparkline(vals)}`",
                "",
                "| window ending | events | meets min | mean | 95% interval | direction |",
                "|---|---|---|---|---|---|",
            ]
            for _, r in cs.iterrows():
                if r["n_events_valid"] == 0:
                    continue
                ci = (
                    f"[{_fmt_pct(r['nw_ci95_lo'])}, {_fmt_pct(r['nw_ci95_hi'])}]"
                    if pd.notna(r["nw_ci95_lo"]) else "n/a"
                )
                lines.append(
                    f"| {pd.Timestamp(r['window_end']).date()} | {int(r['n_events_valid'])} | "
                    f"{'yes' if r['window_sufficient'] else 'no'} | "
                    f"{_fmt_pct(r['mean_abnormal_return'])} | {ci} | {r['own_direction']} |"
                )
            lines.append("")

    stock_scope = coverage["rolling_stock_scope"]
    lines += [
        "## Rolling stock-level coverage",
        "",
        f"- {stock_scope['n_ticker_category_pairs_analysed']:,} of "
        f"{stock_scope['n_ticker_category_pairs_screened']:,} ticker-category pairs could reach "
        f"the {ROLL_STOCK_MIN_EVENTS}-event rolling minimum in any window and were analysed.",
        f"- {stock_scope['n_window_cells_emitted']:,} window cells emitted; "
        f"{stock_scope['n_window_cells_suppressed_below_min']:,} suppressed below "
        f"{ROLL_STOCK_EXPLORATORY_MIN} events (counted, not hidden).",
        f"- {stock_scope['vs_other_note']}",
        "",
    ]
    if len(stock):
        tier = stock.groupby("evidence_tier").size().to_dict()
        lines += [
            "| evidence tier | window cells |",
            "|---|---|",
            *[f"| {k} | {v:,} |" for k, v in sorted(tier.items())],
            "",
            "Note the ceiling: at stock level a 365-day window can at best reach "
            "`exploratory`. Corporate news for one ticker in one category arrives a few times "
            "a year, so no single year can supply a Phase-2-sized per-stock sample. That is a "
            "structural property of news arrival, not a gap in this archive.",
            "",
        ]
    if coverage["figures"]:
        lines += ["## Figures", ""]
        lines += [f"- `{p}`" for p in coverage["figures"]]
        lines += [""]

    lines += [
        "## Limitations",
        "",
        "- **Not real-time.** Phase 3 re-slices a static archive ending "
        f"{f['last_news_date']}; it adds no new data and supports no claim about current "
        "market conditions.",
        f"- Consecutive windows overlap by {wd['window_days'] - wd['category_stride_days']} of "
        f"{wd['window_days']} days. Window-to-window comparisons use only disjoint windows; the "
        "rolling FDR flags remain exploratory because the family itself is overlapping.",
        "- At 60/126/252 sessions the forward-return window is a large fraction of (or longer "
        "than) the 365-day analysis window, so a window contains barely more than one "
        "independent outcome period. HAC corrects the standard error but cannot manufacture "
        "independent observations; treat those rows as descriptive.",
        "- An insufficient or inconclusive recent window is NOT evidence of no impact. At long "
        "horizons it usually means the outcome window has not finished inside the data.",
        "- No shrinkage or pooling is applied in Phase 3. Rolling estimates are raw per-window "
        "HAC estimates with their intervals, so nothing here is pulled toward a prior.",
        "- Sector labels are current GICS membership, inherited from Phase 1, not "
        "point-in-time.",
        "- Historical association only -- never causation, and never a guarantee that the same "
        "news will be followed by the same return again.",
    ]
    return "\n".join(lines)
