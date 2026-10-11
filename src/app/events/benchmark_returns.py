"""Phase 0 of the event-impact analysis: benchmark-adjusted forward returns.

    curated_sp500 panel  +  SPY benchmark series
            |
            |  forward_returns_on_calendar()  <- shared master-calendar shift
            v
    stock forward returns            benchmark forward returns
            |_________________________________|
                            |
                      abnormal_ret_{k}d = stock_ret_{k}d - benchmark_ret_{k}d
                            |
                            v
            benchmark_adjusted_returns_v{N}.parquet  +  _summary.json

These are OUTCOMES/MEASUREMENTS for the event-impact analysis that follows
(news-category return studies, rolling signal diagnostics, stock-sensitivity
ranking) -- never predictive model features. Nothing here is wired into
features.build, features.matrix, or models.train, and nothing here writes to
an existing curated artifact.

Design notes
------------
* Six horizons in TRADING SESSIONS: 1, 5, 20, 60, 126, 252. This is a
  deliberately different set from labeling.targets.DEFAULT_HORIZONS
  (1, 5, 21, 63, 126, 252) -- the modelling pipeline's month/quarter
  approximations -- because this task specifies 20/60 explicitly. Both use
  the same Close(T+k)/Close(T)-1 formula via fwd_ret_col()/label-style
  naming; only the horizon SET differs, and it is named HORIZONS here rather
  than imported, so the two conventions cannot silently drift into each
  other.
* Calendar alignment: forward_returns_on_calendar() reindexes each series
  (every ticker's close, and the benchmark's close) onto the SAME master
  session calendar before shifting by horizon POSITIONS on that shared
  index. This differs from labeling.targets.forward_return(), which shifts
  positionally within each ticker's own (possibly gappy) row sequence --
  fine for same-ticker labels, but NOT sufficient here: if a ticker is
  missing one session that the benchmark has, a per-ticker positional shift
  would silently pair "T+k sessions for this ticker" with a different
  calendar date than "T+k sessions for the benchmark", violating the
  explicit requirement to compare identical start/end dates. Reindexing
  onto one shared calendar first removes that risk by construction: Close
  is NaN at a ticker's missing session, and shifting by position on the
  common index guarantees T+k always means the same calendar date for
  every series.
* Seam masking reuses app.data.seams exactly as features.build does: a
  horizon's STOCK forward return is invalidated (NaN) when its window
  crosses a seam registered for that ticker's (source, ticker) pair. The
  benchmark is sourced fresh via yfinance, which resolves to NO_SEAMS, so
  it never needs masking itself -- but a stock-side seam still invalidates
  the abnormal return at that horizon, since the stock leg is unusable.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from app.data.curate import load_curated, save_curated
from app.data.seams import SOURCE_COLUMN, forward_window_invalid, seams_for_frame
from app.data.validate import validate_prices

HORIZONS: tuple[int, ...] = (1, 5, 20, 60, 126, 252)

BENCHMARK_TICKER = "SPY"
DEFAULT_PANEL_DIR = "data/curated_sp500"
DEFAULT_BENCHMARK_DIR = "data/benchmark_spy"
DEFAULT_OUT_DIR = "data/curated"
OUTPUT_NAME = "benchmark_adjusted_returns"
OUTPUT_VERSION = "v1"
BENCHMARK_FETCH_START = "2008-01-01"  # matches the panel's earliest session


def stock_ret_col(h: int) -> str:
    return f"stock_fwd_ret_{h}d"


def bench_ret_col(h: int) -> str:
    return f"benchmark_fwd_ret_{h}d"


def abnormal_ret_col(h: int) -> str:
    return f"abnormal_ret_{h}d"


def valid_col(h: int) -> str:
    return f"valid_{h}d"


# ---------------------------------------------------------------------------
# Benchmark ingestion
# ---------------------------------------------------------------------------


def ensure_benchmark(
    benchmark_dir: str | Path = DEFAULT_BENCHMARK_DIR,
    panel_dir: str | Path = DEFAULT_PANEL_DIR,
    start: str = BENCHMARK_FETCH_START,
) -> pd.DataFrame:
    """Load a cached SPY series covering the panel's date range, or fetch it.

    Reuses the project's existing validated price path verbatim --
    YFinanceSource.fetch -> validate_prices -> save_curated(source="yfinance")
    -- the same pipeline every ticker recovery in this project has used.
    SPY is kept in its OWN directory, never under curated_sp500/prices, so it
    can never be accidentally picked up by load_curated(curated_sp500) as if
    it were a 448th stock in the training universe.

    A pre-existing legacy SPY file lives at data/curated/prices/ticker=SPY.parquet
    (from the original 9-ticker pipeline) but is NOT reused: it starts at
    2015-01-02 (the panel needs 2008-01-02) and predates the source/seam
    provenance column entirely. Refetching through the current pipeline is
    cheap (one ticker) and keeps provenance consistent.
    """
    benchmark_dir = Path(benchmark_dir)
    sessions = load_curated(panel_dir)["date"]
    panel_start, panel_end = sessions.min(), sessions.max()

    cached = load_curated(benchmark_dir, tickers=[BENCHMARK_TICKER])
    if len(cached) and cached["date"].min() <= panel_start and cached["date"].max() >= panel_end:
        return cached

    from app.data.sources.yfinance_source import YFinanceSource

    raw = YFinanceSource().fetch(BENCHMARK_TICKER, start, None)
    if raw.empty:
        raise RuntimeError(
            f"yfinance returned no data for {BENCHMARK_TICKER}; cannot build a benchmark "
            "without a paid API or a large archive re-download (both excluded by Phase 0's "
            "constraints). This is a genuine blocker, not a silently-substituted benchmark."
        )
    clean, report = validate_prices(raw, ticker=BENCHMARK_TICKER)
    if clean.empty or clean["date"].min() > panel_start or clean["date"].max() < panel_end:
        raise RuntimeError(
            f"{BENCHMARK_TICKER} coverage ({clean['date'].min() if len(clean) else 'empty'} .. "
            f"{clean['date'].max() if len(clean) else 'empty'}) does not span the panel's "
            f"range ({panel_start.date()} .. {panel_end.date()}); issues={report.issues}. "
            "Refusing to silently substitute a different benchmark."
        )
    save_curated(clean, benchmark_dir, BENCHMARK_TICKER, source="yfinance")
    # save_curated writes the source column to disk but does not mutate the
    # frame it was given; set it here too so a freshly-fetched return value
    # has the identical schema a cache-hit (reloaded from disk) would have.
    clean[SOURCE_COLUMN] = "yfinance"
    return clean


# ---------------------------------------------------------------------------
# Calendar-aligned forward returns
# ---------------------------------------------------------------------------


def forward_returns_on_calendar(
    close_by_date: pd.Series,
    sessions: pd.DatetimeIndex,
    horizons: tuple[int, ...] = HORIZONS,
) -> pd.DataFrame:
    """Forward returns for ONE series, shifted by position on `sessions`.

    `close_by_date` is indexed by date (need not cover every session --
    missing sessions become NaN after reindexing). Returns one column per
    horizon, indexed by `sessions`, holding Close(T+k)/Close(T) - 1 where
    T and T+k are both POSITIONS on the shared calendar `sessions` -- so
    two different series (e.g. a stock and the benchmark) always compare
    the identical pair of calendar dates for a given horizon and row.
    """
    aligned = close_by_date.reindex(sessions)
    out = pd.DataFrame(index=sessions)
    arr = aligned.to_numpy()
    n = len(arr)
    for h in horizons:
        fwd = np.full(n, np.nan)
        if h < n:
            with np.errstate(invalid="ignore", divide="ignore"):
                fwd[: n - h] = arr[h:] / arr[:-h] - 1
        out[h] = fwd
    return out


# ---------------------------------------------------------------------------
# Full Phase 0 pipeline
# ---------------------------------------------------------------------------


def build_abnormal_returns(
    panel_dir: str | Path = DEFAULT_PANEL_DIR,
    benchmark_dir: str | Path = DEFAULT_BENCHMARK_DIR,
    horizons: tuple[int, ...] = HORIZONS,
    out_dir: str | Path = DEFAULT_OUT_DIR,
    version: str = OUTPUT_VERSION,
) -> dict:
    """Build and persist the benchmark-adjusted forward-return table.

    One row per (ticker, date) present in the curated panel ("eligible"
    ticker-dates -- every session a ticker actually has a valid price bar
    on). For each horizon: the stock forward return, the benchmark forward
    return over the IDENTICAL calendar interval, the abnormal return, and a
    boolean validity flag. A horizon is marked invalid (all three columns
    NaN) rather than dropped when: the window runs past the end of the
    data, the stock's own window crosses a registered seam, or either
    series is missing a close at either endpoint.
    """
    panel = load_curated(panel_dir)
    benchmark = ensure_benchmark(benchmark_dir, panel_dir)

    sessions = pd.DatetimeIndex(sorted(panel["date"].unique()))
    bench_close = benchmark.set_index("date")["close"].sort_index()
    bench_fwd = forward_returns_on_calendar(bench_close, sessions, horizons)
    bench_has_close = bench_close.reindex(sessions).notna()

    frames: list[pd.DataFrame] = []
    n_seam_masked = {h: 0 for h in horizons}
    n_incomplete = {h: 0 for h in horizons}
    n_missing_price = {h: 0 for h in horizons}
    n_valid = {h: 0 for h in horizons}

    n_sessions = len(sessions)
    bench_present = bench_has_close.to_numpy()

    for ticker, g in panel.groupby("ticker"):
        g = g.sort_values("date")
        close = g.set_index("date")["close"]
        seams = seams_for_frame(g)

        stock_fwd = forward_returns_on_calendar(close, sessions, horizons)
        row = pd.DataFrame({"date": sessions, "ticker": ticker})
        row["source"] = g["source"].iloc[0] if "source" in g.columns else None

        own_present = close.reindex(sessions).notna().to_numpy()

        for h in horizons:
            stock_r = stock_fwd[h].to_numpy()
            bench_r = bench_fwd[h].to_numpy()

            # Three DISTINCT, mutually exclusive, priority-ordered reasons a
            # horizon can be unavailable (matches the task's own categories):
            #   1. the window runs past the end of known history
            #   2. the window is within history but a required close (stock
            #      at T or T+h, or benchmark at T or T+h) is simply absent
            #      -- a dropped/invalid bar, not a seam
            #   3. both endpoints exist, but the interval between them
            #      crosses a REGISTERED discontinuity for this ticker
            beyond_calendar = np.zeros(n_sessions, dtype=bool)
            beyond_calendar[n_sessions - h :] = True

            end_present = np.zeros(n_sessions, dtype=bool)
            end_present[: n_sessions - h] = own_present[h:] & bench_present[h:]
            price_missing = ~beyond_calendar & ~(own_present & bench_present & end_present)

            seam_bad = (
                forward_window_invalid(sessions, h, seams)
                if seams
                else np.zeros(n_sessions, dtype=bool)
            )
            seam_bad = seam_bad & ~beyond_calendar & ~price_missing

            invalid = beyond_calendar | price_missing | seam_bad
            abn = stock_r - bench_r

            row[stock_ret_col(h)] = np.where(invalid, np.nan, stock_r)
            row[bench_ret_col(h)] = np.where(invalid, np.nan, bench_r)
            row[abnormal_ret_col(h)] = np.where(invalid, np.nan, abn)
            row[valid_col(h)] = ~invalid

            # Count reasons only over sessions this ticker is actually
            # ELIGIBLE for (own_present) -- the same filter applied to `row`
            # below. Without this gate, every session outside a ticker's own
            # listed range (which never appears in the output at all) would
            # still be tallied as "missing price", inflating that count by
            # the full non-eligible population rather than describing the
            # rows the output table actually contains.
            n_incomplete[h] += int((beyond_calendar & own_present).sum())
            n_missing_price[h] += int((price_missing & own_present).sum())
            n_seam_masked[h] += int((seam_bad & own_present).sum())
            n_valid[h] += int((~invalid & own_present).sum())

        # restrict rows to sessions where this ticker actually has a priced
        # bar -- "eligible ticker-date" means the panel itself has it, not
        # every master session (a ticker can be unlisted on some of them).
        row = row[own_present]
        frames.append(row)

    out = (
        pd.concat(frames, ignore_index=True)
        .sort_values(["ticker", "date"])
        .reset_index(drop=True)
    )

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{OUTPUT_NAME}_{version}.parquet"
    out.to_parquet(out_path, index=False)

    # ----- sanity checks -----------------------------------------------
    sanity = {}
    for h in horizons:
        s, b = out[stock_ret_col(h)], out[bench_ret_col(h)]
        mask = s.notna() & b.notna()
        sanity[f"{h}d_pooled_corr_stock_vs_benchmark"] = (
            float(s[mask].corr(b[mask])) if mask.sum() > 1 else None
        )
        sanity[f"{h}d_mean_stock_return"] = float(s.mean()) if s.notna().any() else None
        sanity[f"{h}d_mean_benchmark_return"] = float(b[mask].mean()) if mask.any() else None
        abn_col = out[abnormal_ret_col(h)]
        sanity[f"{h}d_mean_abnormal_return"] = (
            float(abn_col.mean()) if abn_col.notna().any() else None
        )

    summary = {
        "input_panel_dir": str(panel_dir),
        "input_date_range": [str(sessions.min().date()), str(sessions.max().date())],
        "input_tickers": int(panel["ticker"].nunique()),
        "input_rows": int(len(panel)),
        "benchmark_ticker": BENCHMARK_TICKER,
        "benchmark_date_range": [
            str(bench_close.index.min().date()),
            str(bench_close.index.max().date()),
        ],
        "benchmark_rows": int(len(bench_close)),
        "horizons": list(horizons),
        "output_rows": int(len(out)),
        "output_path": str(out_path),
        "per_horizon": {
            h: {
                "n_valid": n_valid[h],
                "n_incomplete_window_end_of_data": n_incomplete[h],
                "n_missing_price_at_endpoint": n_missing_price[h],
                "n_seam_masked": n_seam_masked[h],
                "pct_valid": round(n_valid[h] / len(out), 4) if len(out) else None,
            }
            for h in horizons
        },
        "sanity_checks": sanity,
        "excludes": [
            "Rows are never dropped for being predictive features; this table is "
            "outcomes/measurements only and is not wired into features.build, "
            "features.matrix, or models.train.",
        ],
    }
    summary_path = out_dir / f"{OUTPUT_NAME}_{version}_summary.json"
    summary_path.write_text(json.dumps(summary, indent=1, default=str))
    summary["summary_path"] = str(summary_path)
    return summary
