"""Phase 2: stock-specific news-category sensitivity.

    phase1_events_v1.parquet (READ-ONLY -- already built, already audit-fixed)
            |
            |  group by (ticker, category) instead of (sector, category)
            v
    per-(ticker, category, horizon): mean/median/percentiles, NW CI,
    evidence tier (adequately_supported / exploratory / insufficient)
            |
            +--> stock_category_horizon_summary  (own stats, + empirical-Bayes
            |                                      shrinkage toward the Phase 1
            |                                      category-wide mean)
            +--> stock_vs_other                  (vs this ticker's own OTHER,
            |                                      same machinery as Phase 1,
            |                                      BH-FDR within each
            |                                      (category, horizon) family
            |                                      across all tickers)
            v
    phase2_stock_category_horizon_summary_v1.parquet
    phase2_stock_vs_other_v1.parquet
    phase2_stock_ranking_v1.parquet
    phase2_coverage_v1.json
    phase2_report_v1.md

Reuses Phase 1's computational primitives directly: `_describe_group` (for
a ticker's own per-category stats) is called UNCHANGED, and the irregular-
spacing Newey-West HAC estimator (`newey_west_mean_ci_irregular`) and its
date-position helper (`session_positions`) are reused unchanged for every
inference in this module. Nothing here re-derives event-date assignment,
re-scores sentiment, or rebuilds the event table; `build_event_table` is
never called -- phase1_events_v1.parquet is read directly.

Two things do NOT carry over unchanged from Phase 1, both discovered while
building this module (see each function's docstring for the full story):

1. Phase 1's `_compare_vs_other` has a sufficiency gate requiring
   n_tickers >= MIN_TICKERS (5) -- a sector/overall-level pooling rule. A
   single ticker's own group always has n_tickers == 1, and the gate is
   baked into whether the real numbers get computed at all, not just a
   label that can be overridden afterward. This module computes its own
   evidence tiering (STOCK_MIN_EVENTS / STOCK_EXPLORATORY_MIN_EVENTS) and
   never calls `_compare_vs_other` for stock-level work.

2. Phase 1's comparison methodology itself -- same-date paired difference
   between two POOLS OF DIFFERENT TICKERS -- does not transfer to a single
   ticker at all: for one ticker, a given date's return is the same number
   regardless of which category tag a news item that day carries, so a
   paired same-date difference is trivially zero whenever the two groups
   share a date. `_stock_compare_vs_other` below implements the correct
   two-sample version instead.

Evidence tiers (explicit, not binary):
  adequately_supported  n_valid_events >= STOCK_MIN_EVENTS (20) -- eligible
                         for the ranking and for FDR-corrected significance
  exploratory            STOCK_EXPLORATORY_MIN_EVENTS (8) <= n < 20 -- a mean
                         is shown, clearly labelled, never ranked
  insufficient            n < 8 -- no summary statistic is presented beyond
                         the raw count

Shrinkage: a standard empirical-Bayes (DerSimonian-Laird random-effects)
partial pooling of each ticker's raw mean toward the Phase 1 category-wide
mean (reused from phase1_category_horizon_summary_v1.parquet, not
recomputed), weighted by how much between-ticker variance in true effects
exists versus how uncertain each ticker's own estimate is. This is the
textbook justification requested: it needs no assumption beyond the normal-
approximation already used throughout this project's inference, and it
degrades gracefully (shrinkage weight -> 0, i.e. no shrinkage) when too few
tickers are sufficiently sampled to estimate the between-ticker variance --
in that case the category/horizon is skipped and raw estimates are reported
with their uncertainty instead, per the task's own documented fallback.
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
    NON_OTHER_CATEGORIES,
    _describe_group,
    benjamini_hochberg,
    newey_west_mean_ci_irregular,
    session_positions,
)
from app.events.sector_map import sector_series

STOCK_MIN_EVENTS = 20
STOCK_EXPLORATORY_MIN_EVENTS = 8
STOCK_MIN_OTHER_DATES = 10
SHRINKAGE_MIN_TICKERS = 20  # tickers needed, sufficiently sampled, to fit tau^2

DEFAULT_EVENTS = Path("data/curated/phase1_events_v1.parquet")
DEFAULT_CATEGORY_SUMMARY = Path("data/curated/phase1_category_horizon_summary_v1.parquet")
DEFAULT_SECTOR_SUMMARY = Path("data/curated/phase1_sector_category_horizon_summary_v1.parquet")
DEFAULT_PANEL_DIR = "data/curated_sp500"
DEFAULT_OUT_DIR = Path("data/curated")
OUTPUT_VERSION = "v1"


def _evidence_tier(n_valid: int) -> str:
    if n_valid >= STOCK_MIN_EVENTS:
        return "adequately_supported"
    if n_valid >= STOCK_EXPLORATORY_MIN_EVENTS:
        return "exploratory"
    return "insufficient"


# ===========================================================================
# 1. Per-ticker descriptive stats (reuses _describe_group verbatim)
# ===========================================================================


def stock_category_horizon_summary(
    events: pd.DataFrame, session_pos: pd.Series, horizons=HORIZONS
) -> pd.DataFrame:
    """One row per (ticker, category, horizon): same statistics Phase 1
    computes for a sector or the whole market, computed per ticker instead,
    with stock-appropriate evidence tiering layered on top."""
    rows = []
    for (ticker, cat), g in events.groupby(["ticker", "category"], sort=True):
        for h in horizons:
            stats_h = _describe_group(g, h, session_pos)
            stats_h["evidence_tier"] = _evidence_tier(stats_h["n_events_valid"])
            rows.append({"ticker": ticker, "category": cat, **stats_h})
    return pd.DataFrame(rows)


def attach_sector_baseline(
    stock_summary: pd.DataFrame, sector_summary: pd.DataFrame, sector_by_ticker: pd.Series
) -> pd.DataFrame:
    """Attach each ticker's CURRENT-GICS-sector baseline (Phase 1's sector-level
    mean/NW estimate for the same category and horizon) as context columns.

    Only attached where Phase 1 itself judged that (sector, category, horizon)
    cell sufficiently sampled (`sufficient_sample`) -- this never invents a
    baseline where Phase 1 found none, per "sector/category baseline for
    context, where sufficiently supported." Sector membership is current, not
    point-in-time (see app.events.sector_map).
    """
    out = stock_summary.copy()
    out["sector"] = out["ticker"].map(sector_by_ticker)
    sec = sector_summary[
        [
            "sector",
            "category",
            "horizon",
            "mean_abnormal_return",
            "nw_mean",
            "nw_se",
            "n_tickers",
            "sufficient_sample",
        ]
    ].rename(
        columns={
            "mean_abnormal_return": "sector_mean_abnormal_return",
            "nw_mean": "sector_nw_mean",
            "nw_se": "sector_nw_se",
            "n_tickers": "sector_n_tickers",
            "sufficient_sample": "sector_sufficient_sample",
        }
    )
    out = out.merge(sec, on=["sector", "category", "horizon"], how="left")
    insufficient = ~out["sector_sufficient_sample"].eq(True)
    sector_cols = [
        "sector_mean_abnormal_return", "sector_nw_mean", "sector_nw_se", "sector_n_tickers",
    ]
    for col in sector_cols:
        out.loc[insufficient, col] = np.nan
    return out


# ===========================================================================
# 2. Per-ticker category vs that SAME ticker's OTHER
# ===========================================================================


def _stock_compare_vs_other(
    cat_events: pd.DataFrame, other_events: pd.DataFrame, horizon: int, session_pos: pd.Series
) -> dict:
    """Compare ONE ticker's returns on category-tagged dates to its returns
    on OTHER-tagged dates. Deliberately NOT a port of Phase 1's
    `_compare_vs_other` (same-date paired difference between two POOLS OF
    DIFFERENT TICKERS) -- that device relies on the two groups containing
    different entities on the same date, so a genuine difference can exist
    date-by-date. For a SINGLE ticker, `abnormal_ret_{h}d` on a given date
    is just that stock's one realized return, independent of which category
    tag a news article that day carries: if a ticker has both an
    EARNINGS_RESULTS and an OTHER article on the same date, both rows carry
    the IDENTICAL return value (both come from the same Phase 0 (ticker,
    date) row). A same-date paired difference is therefore trivially zero
    on every date the two groups share -- caught in testing, see
    test_stock_comparison_is_not_the_sector_style_paired_date_difference.

    The correct comparison here is a two-sample one: the category's own
    dates vs. OTHER's own dates (with any date common to both excluded from
    the OTHER side, so the two samples are disjoint), each Newey-West HAC
    adjusted independently for its own window-overlap serial correlation,
    combined as an approximately-independent two-sample difference
    (se = sqrt(se_cat^2 + se_other^2), the standard combination for two
    samples drawn from non-overlapping date sets).
    """
    ret_c, valid_c = abnormal_ret_col(horizon), valid_col(horizon)
    cat_valid = cat_events[cat_events[valid_c]]
    oth_valid = other_events[other_events[valid_c]]

    cat_dates = set(cat_valid["date"])
    oth_only = oth_valid[~oth_valid["date"].isin(cat_dates)]
    n_overlap_excluded = len(oth_valid) - len(oth_only)

    cat_series = cat_valid.drop_duplicates("date").set_index("date")[ret_c].sort_index()
    oth_series = oth_only.drop_duplicates("date").set_index("date")[ret_c].sort_index()

    n_events = len(cat_valid)
    sufficient = n_events >= STOCK_MIN_EVENTS and len(oth_series) >= STOCK_MIN_OTHER_DATES
    base = {
        "horizon": horizon, "n_events_category": n_events,
        "n_tickers_category": int(cat_valid["ticker"].nunique()),
        "n_other_dates": int(len(oth_series)),
        "n_same_day_overlap_excluded": int(n_overlap_excluded),
        "sufficient_sample": sufficient, "exploratory_horizon": horizon in EXPLORATORY_HORIZONS,
        "evidence_tier": _evidence_tier(n_events),
    }
    if len(cat_series) < 2 or len(oth_series) < 2:
        return {**base, "diff_mean": None, "se": None, "t_stat": None, "p_value": None,
                "ci95_lo": None, "ci95_hi": None}

    lag = max(horizon - 1, 0)
    nw_cat = newey_west_mean_ci_irregular(
        cat_series.to_numpy(), session_positions(cat_series.index, session_pos), lag
    )
    nw_oth = newey_west_mean_ci_irregular(
        oth_series.to_numpy(), session_positions(oth_series.index, session_pos), lag
    )
    diff_mean = nw_cat["mean"] - nw_oth["mean"]
    if nw_cat["se"] is None or nw_oth["se"] is None:
        return {**base, "diff_mean": diff_mean, "se": None, "t_stat": None, "p_value": None,
                "ci95_lo": None, "ci95_hi": None}

    se = float(np.sqrt(nw_cat["se"] ** 2 + nw_oth["se"] ** 2))
    if se > 0:
        t_stat = diff_mean / se
        df = max(min(len(cat_series), len(oth_series)) - 1, 1)
        p_value = float(2 * (1 - stats.t.cdf(abs(t_stat), df=df)))
        ci = [diff_mean - 1.96 * se, diff_mean + 1.96 * se]
    else:
        t_stat, p_value, ci = None, None, [diff_mean, diff_mean]
    return {
        **base, "diff_mean": diff_mean, "se": se, "t_stat": t_stat, "p_value": p_value,
        "ci95_lo": ci[0], "ci95_hi": ci[1],
    }


def stock_vs_other(
    events: pd.DataFrame, session_pos: pd.Series, horizons=HORIZONS
) -> pd.DataFrame:
    """One row per (ticker, category, horizon) comparing a category's events
    to the SAME TICKER's own OTHER-category events at the same horizon.
    BH-FDR correction is applied separately within each (category, horizon)
    family across all tickers -- "of the ~440 tickers tested for a reaction
    to ANALYST_ACTION at 20 sessions, how many survive correction" -- which
    is the natural per-cell family for a ticker-sensitivity screen.
    """
    rows = []
    for ticker, tg in events.groupby("ticker", sort=True):
        other = tg[tg["category"] == "OTHER"]
        for cat in NON_OTHER_CATEGORIES:
            cat_events = tg[tg["category"] == cat]
            if cat_events.empty:
                continue
            for h in horizons:
                cmp = _stock_compare_vs_other(cat_events, other, h, session_pos)
                rows.append({"ticker": ticker, "category": cat, **cmp})
    df = pd.DataFrame(rows)

    df["significant_fdr"] = False
    for _key, idx in df.groupby(["category", "horizon"]).groups.items():
        sub = df.loc[idx]
        pvals = [
            p if s else None
            for p, s in zip(sub["p_value"], sub["sufficient_sample"], strict=True)
        ]
        flags = benjamini_hochberg(pvals, FDR_Q)
        df.loc[idx, "significant_fdr"] = flags
        df.loc[idx, "n_tests_in_family"] = int(
            sum(v is not None and not np.isnan(v) for v in pvals)
        )
    return df


# ===========================================================================
# 3. Empirical-Bayes shrinkage (DerSimonian-Laird random-effects)
# ===========================================================================


def _der_simonian_laird_tau2(thetas: np.ndarray, ses: np.ndarray) -> float:
    """Standard random-effects between-group variance estimator (DerSimonian
    & Laird, 1986). thetas/ses are the raw per-ticker effect estimates and
    their own standard errors, restricted to sufficiently-sampled tickers."""
    w = 1.0 / np.square(ses)
    theta_bar = float(np.sum(w * thetas) / np.sum(w))
    q = float(np.sum(w * np.square(thetas - theta_bar)))
    k = len(thetas)
    c = float(np.sum(w) - np.sum(np.square(w)) / np.sum(w))
    if c <= 0:
        return 0.0
    return max(0.0, (q - (k - 1)) / c)


def shrink_category_horizon(
    stock_summary: pd.DataFrame, category_summary: pd.DataFrame
) -> pd.DataFrame:
    """Add shrunk_mean / shrinkage_weight / tau2 columns to the stock-level
    summary: each ticker's raw mean is pulled toward the Phase 1 category-
    wide mean (REUSED from phase1_category_horizon_summary_v1.parquet, not
    recomputed), by shrinkage weight w = tau2/(tau2+se^2) -- the standard
    empirical-Bayes partial-pooling formula. w -> 0 (full shrinkage to the
    prior) when a ticker's own estimate is noisy relative to the genuine
    spread of effects across tickers; w -> 1 (trust the data) when its own
    estimate is precise.

    Skips shrinkage (shrunk_mean = raw mean, tau2 = NaN) for any
    (category, horizon) with fewer than SHRINKAGE_MIN_TICKERS sufficiently-
    sampled tickers to fit tau2 reliably -- the task's own documented
    fallback rather than forcing an unsupported variance estimate.
    """
    out = stock_summary.copy()
    out["shrunk_mean"] = out["mean_abnormal_return"]
    out["shrinkage_weight"] = np.nan
    out["tau2"] = np.nan

    prior = category_summary.set_index(["category", "horizon"])["mean_abnormal_return"]

    for (cat, h), idx in out.groupby(["category", "horizon"]).groups.items():
        if (cat, h) not in prior.index:
            continue
        prior_mean = float(prior.loc[(cat, h)])
        sub = out.loc[idx]
        fit = sub[(sub["evidence_tier"] == "adequately_supported") & sub["nw_se"].notna()]
        if len(fit) < SHRINKAGE_MIN_TICKERS:
            continue
        tau2 = _der_simonian_laird_tau2(
            fit["mean_abnormal_return"].to_numpy(), fit["nw_se"].to_numpy()
        )
        has_se = sub["nw_se"].notna() & (sub["nw_se"] > 0)
        se2 = sub.loc[has_se, "nw_se"] ** 2
        w = tau2 / (tau2 + se2)
        shrunk = prior_mean + w * (sub.loc[has_se, "mean_abnormal_return"] - prior_mean)
        out.loc[sub.loc[has_se].index, "shrunk_mean"] = shrunk
        out.loc[sub.loc[has_se].index, "shrinkage_weight"] = w
        out.loc[idx, "tau2"] = tau2
    return out


# ===========================================================================
# 4. Ranking: adequately-supported, FDR-significant, sorted by shrunk effect
# ===========================================================================


def rank_stock_sensitivities(
    stock_summary_shrunk: pd.DataFrame, vs_other: pd.DataFrame, top_n: int | None = None
) -> pd.DataFrame:
    """Ranks (ticker, category, horizon) cells that are BOTH adequately
    sampled AND FDR-significant vs that ticker's own OTHER baseline, by
    |NW-adjusted vs-OTHER difference| -- never by raw mean alone, and never
    including a cell that failed the minimum-sample or significance bar,
    however large its raw number looks.

    This sorts by the (NW-adjusted) two-sample difference itself, not by
    `shrunk_mean` -- shrinkage here is only fit on each ticker's own
    category-mean return (toward the category-wide prior), not on the
    vs-OTHER difference, so there is no "shrunk diff" to sort by without
    also shrinking the OTHER-side estimate, which this phase does not do.
    """
    merge_cols = [
        "ticker", "category", "horizon", "shrunk_mean", "shrinkage_weight", "tau2", "n_tickers",
    ]
    merged = vs_other.merge(
        stock_summary_shrunk[merge_cols],
        on=["ticker", "category", "horizon"],
        how="left",
    )
    eligible = merged[
        (merged["evidence_tier"] == "adequately_supported") & merged["significant_fdr"]
    ].copy()
    eligible["abs_diff_mean"] = eligible["diff_mean"].abs()
    eligible = eligible.sort_values(
        ["abs_diff_mean", "n_events_category"], ascending=[False, False]
    ).reset_index(drop=True)
    eligible["rank"] = np.arange(1, len(eligible) + 1)
    if top_n is not None:
        eligible = eligible.head(top_n)
    return eligible


# ===========================================================================
# 5. Orchestration
# ===========================================================================


def run(
    events_path: str | Path = DEFAULT_EVENTS,
    category_summary_path: str | Path = DEFAULT_CATEGORY_SUMMARY,
    sector_summary_path: str | Path = DEFAULT_SECTOR_SUMMARY,
    panel_dir: str | Path = DEFAULT_PANEL_DIR,
    out_dir: str | Path = DEFAULT_OUT_DIR,
    version: str = OUTPUT_VERSION,
    horizons: tuple[int, ...] = HORIZONS,
) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    events = pd.read_parquet(events_path)
    category_summary = pd.read_parquet(category_summary_path)
    sector_summary = pd.read_parquet(sector_summary_path)

    panel = load_curated(panel_dir)
    sessions = pd.DatetimeIndex(sorted(panel["date"].unique()))
    session_pos = pd.Series(np.arange(len(sessions)), index=sessions)

    sector_by_ticker = sector_series(events["ticker"].unique())

    stock_summary = stock_category_horizon_summary(events, session_pos, horizons)
    stock_summary = shrink_category_horizon(stock_summary, category_summary)
    stock_summary = attach_sector_baseline(stock_summary, sector_summary, sector_by_ticker)
    stock_summary_path = out_dir / f"phase2_stock_category_horizon_summary_{version}.parquet"
    stock_summary.to_parquet(stock_summary_path, index=False)

    vs_other = stock_vs_other(events, session_pos, horizons)
    vs_other_path = out_dir / f"phase2_stock_vs_other_{version}.parquet"
    vs_other.to_parquet(vs_other_path, index=False)

    ranking = rank_stock_sensitivities(stock_summary, vs_other)
    ranking_path = out_dir / f"phase2_stock_ranking_{version}.parquet"
    ranking.to_parquet(ranking_path, index=False)

    # ---- coverage -------------------------------------------------------
    tier_counts = (
        stock_summary.groupby(["category", "horizon", "evidence_tier"]).size().unstack(fill_value=0)
    )
    shrinkage_fit = (
        stock_summary.dropna(subset=["tau2"])
        .groupby(["category", "horizon"])["tau2"]
        .first()
        .to_dict()
    )
    coverage = {
        "purpose": "Phase 2: stock-specific news-category sensitivity, built on the "
                   "audit-corrected Phase 1 event table. Associations, not causation, "
                   "and not a guarantee of future response.",
        "inputs": {
            "events": str(events_path),
            "category_summary": str(category_summary_path),
            "sector_summary": str(sector_summary_path),
            "panel_dir": str(panel_dir),
        },
        "n_tickers": int(events["ticker"].nunique()),
        "n_events_total": int(len(events)),
        "evidence_tiers": {
            "adequately_supported": f">= {STOCK_MIN_EVENTS} valid events",
            "exploratory": f"{STOCK_EXPLORATORY_MIN_EVENTS}-{STOCK_MIN_EVENTS - 1} valid events",
            "insufficient": f"< {STOCK_EXPLORATORY_MIN_EVENTS} valid events",
        },
        "min_other_dates_for_comparison": STOCK_MIN_OTHER_DATES,
        "shrinkage_method": (
            "Empirical-Bayes / DerSimonian-Laird random-effects partial pooling toward "
            "the Phase 1 category-wide mean. Skipped (raw estimate reported instead) for "
            f"any (category, horizon) with fewer than {SHRINKAGE_MIN_TICKERS} adequately-"
            "sampled tickers to fit the between-ticker variance."
        ),
        "shrinkage_tau2_by_category_horizon": {
            f"{k[0]}|{k[1]}d": v for k, v in shrinkage_fit.items()
        },
        "multiple_testing": (
            "Benjamini-Hochberg FDR control at q=0.10, applied separately within each "
            "(category, horizon) family across all tickers (i.e. one correction per "
            "36 category x horizon cells, each over up to ~440 ticker-level tests)."
        ),
        "exploratory_horizons": list(EXPLORATORY_HORIZONS),
        "tier_counts_by_category_horizon": tier_counts.reset_index().to_dict(orient="records"),
        "n_fdr_significant_adequately_supported": int(
            len(
                vs_other[
                    (vs_other["evidence_tier"] == "adequately_supported")
                    & vs_other["significant_fdr"]
                ]
            )
        ),
        "outputs": {
            "stock_category_horizon_summary": str(stock_summary_path),
            "stock_vs_other": str(vs_other_path),
            "stock_ranking": str(ranking_path),
        },
        "caveats": [
            "Sector classifications (where referenced) are CURRENT GICS membership, "
            "not point-in-time historical membership.",
            "Historical association only -- not causal, not a guaranteed future pattern.",
        ],
    }
    coverage_path = out_dir / f"phase2_coverage_{version}.json"
    coverage_path.write_text(json.dumps(coverage, indent=1, default=str))
    coverage["coverage_path"] = str(coverage_path)

    report_path = out_dir / f"phase2_report_{version}.md"
    report_path.write_text(render_markdown_report(stock_summary, vs_other, ranking, coverage))
    coverage["report_path"] = str(report_path)
    return coverage


# ===========================================================================
# 6. Markdown report
# ===========================================================================


def _fmt_pct(x) -> str:
    return "n/a" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x * 100:+.2f}%"


def render_markdown_report(
    stock_summary: pd.DataFrame, vs_other: pd.DataFrame, ranking: pd.DataFrame, coverage: dict
) -> str:
    lines = [
        "# Phase 2: Stock-Specific News-Category Sensitivity",
        "",
        "For each stock, which categories of news have historically been followed by a "
        "benchmark-adjusted return large enough, and consistent enough, to distinguish "
        "from noise -- and over what horizon. **Historical association, not causation, "
        "and not a guarantee of future response.**",
        "",
        f"- {coverage['n_tickers']} tickers, {coverage['n_events_total']:,} total events "
        "(reused unchanged from the audit-corrected Phase 1 event table)",
        f"- Evidence tiers: adequately_supported (>= {STOCK_MIN_EVENTS} events), "
        f"exploratory ({STOCK_EXPLORATORY_MIN_EVENTS}-{STOCK_MIN_EVENTS - 1}), "
        f"insufficient (< {STOCK_EXPLORATORY_MIN_EVENTS})",
        f"- {coverage['shrinkage_method']}",
        f"- {coverage['multiple_testing']}",
        f"- **{coverage['n_fdr_significant_adequately_supported']} (ticker, category, "
        "horizon) cells are both adequately sampled and FDR-significant**, out of "
        f"{len(vs_other):,} computed.",
        "",
        "## Top adequately-supported, FDR-significant stock sensitivities",
        "",
        "Ranked by |effect size| among cells that passed BOTH the sample-size bar and "
        "FDR correction -- never by raw mean alone.",
        "",
        "| rank | ticker | category | horizon | diff vs own OTHER | 95% CI | n events "
        "| n other dates |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for _, r in ranking.head(15).iterrows():
        expl = " *(exploratory horizon)*" if r["exploratory_horizon"] else ""
        lines.append(
            f"| {int(r['rank'])} | {r['ticker']} | {r['category']} | {r['horizon']}d{expl} | "
            f"{_fmt_pct(r['diff_mean'])} | [{_fmt_pct(r['ci95_lo'])}, {_fmt_pct(r['ci95_hi'])}] | "
            f"{r['n_events_category']:,} | {r['n_other_dates']} |"
        )

    lines += ["", "## Illustrative stock profiles (selected from the actual ranking above)", ""]
    top_tickers = ranking.head(3)["ticker"].unique() if len(ranking) else []
    for t in top_tickers:
        ts = stock_summary[stock_summary["ticker"] == t].sort_values(["category", "horizon"])
        sector = ts["sector"].iloc[0] if len(ts) and pd.notna(ts["sector"].iloc[0]) else "unknown"
        lines.append(f"### {t} (sector: {sector})")
        lines.append("")
        lines.append(
            "| category | horizon | n events | mean | shrunk mean | tier | vs OTHER | FDR-sig "
            "| sector baseline |"
        )
        lines.append("|---|---|---|---|---|---|---|---|---|")
        tv = vs_other[vs_other["ticker"] == t].set_index(["category", "horizon"])
        for _, r in ts.iterrows():
            key = (r["category"], r["horizon"])
            vrow = tv.loc[key] if key in tv.index else None
            vrow_ok = vrow is not None and vrow["sufficient_sample"]
            vs_str = _fmt_pct(vrow["diff_mean"]) if vrow_ok else "n/a"
            sig = "**yes**" if vrow_ok and vrow["significant_fdr"] else "no"
            sector_str = _fmt_pct(r["sector_nw_mean"]) if pd.notna(r["sector_nw_mean"]) else "n/a"
            lines.append(
                f"| {r['category']} | {r['horizon']}d | {r['n_events_valid']:,} | "
                f"{_fmt_pct(r['mean_abnormal_return'])} | {_fmt_pct(r['shrunk_mean'])} | "
                f"{r['evidence_tier']} | {vs_str} | {sig} | {sector_str} |"
            )
        lines.append("")

    lines += [
        "## Where evidence is too sparse",
        "",
        "Per-category coverage at the adequately-supported bar, across all tickers:",
        "",
        "| category | horizon | adequately supported | exploratory | insufficient |",
        "|---|---|---|---|---|",
    ]
    for row in coverage["tier_counts_by_category_horizon"]:
        lines.append(
            f"| {row['category']} | {row['horizon']}d | {row.get('adequately_supported', 0)} | "
            f"{row.get('exploratory', 0)} | {row.get('insufficient', 0)} |"
        )

    lines += [
        "",
        "## Limitations",
        "",
        "- Shrinkage pulls noisy per-ticker estimates toward the Phase 1 category-wide "
        "mean; it reduces false alarms from a handful of extreme events but does not "
        "eliminate them, and is skipped (raw estimate shown) wherever too few tickers "
        "exist to fit the between-ticker variance reliably.",
        "- Sector context, where shown, uses CURRENT GICS membership (see Phase 1), "
        "not point-in-time historical sector assignment.",
        f"- Horizons {EXPLORATORY_HORIZONS} rest on fewer independent observations than "
        "their raw event counts suggest, because adjacent return windows overlap.",
        "- Historical association only -- never evidence of causation, and never a "
        "guarantee that the same stock will respond the same way to similar news again.",
    ]
    return "\n".join(lines)
