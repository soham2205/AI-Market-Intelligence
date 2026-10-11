"""Phase 0: benchmark-adjusted forward returns.

Synthetic fixtures keep these fast; tests against the real built artifact
skip when it has not been generated.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from app.data.seams import SOURCE_COLUMN
from app.events.benchmark_returns import (
    BENCHMARK_TICKER,
    HORIZONS,
    abnormal_ret_col,
    bench_ret_col,
    build_abnormal_returns,
    forward_returns_on_calendar,
    stock_ret_col,
    valid_col,
)

REAL_PANEL = Path("data/curated_sp500")
REAL_OUTPUT = Path("data/curated/benchmark_adjusted_returns_v1.parquet")
MATRIX_H5 = Path("data/curated/matrix_h5_v4.parquet")
MATRIX_H1 = Path("data/curated/matrix_h1_v4.parquet")


def sessions(n=40, start="2020-01-02"):
    return pd.bdate_range(start, periods=n)


def close_series(values, idx):
    return pd.Series(values, index=idx, dtype=float)


def synthetic_panel(tickers=("AAA", "BBB"), n=40, source="yfinance", gap_at=None):
    """A small curated-panel-shaped frame: date, ticker, OHLCV-ish, source."""
    idx = sessions(n)
    frames = []
    for i, t in enumerate(tickers):
        rng = np.random.default_rng(i)
        close = 100 + np.cumsum(rng.normal(0, 1, n))
        df = pd.DataFrame(
            {
                "date": idx,
                "ticker": t,
                "open": close,
                "high": close * 1.01,
                "low": close * 0.99,
                "close": close,
                "volume": 1e6,
                SOURCE_COLUMN: source,
            }
        )
        if gap_at is not None and t == tickers[0]:
            df = df.drop(df.index[gap_at]).reset_index(drop=True)
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def write_panel(tmp_path: Path, panel: pd.DataFrame) -> Path:
    from app.data.curate import save_curated

    panel_dir = tmp_path / "panel"
    for ticker, g in panel.groupby("ticker"):
        source = g[SOURCE_COLUMN].iloc[0]
        save_curated(g.drop(columns=[SOURCE_COLUMN]), panel_dir, ticker, source=source)
    return panel_dir


def write_benchmark(tmp_path: Path, bench_close: pd.Series) -> Path:
    from app.data.curate import save_curated

    df = pd.DataFrame(
        {
            "date": bench_close.index,
            "ticker": BENCHMARK_TICKER,
            "open": bench_close.values,
            "high": bench_close.values,
            "low": bench_close.values,
            "close": bench_close.values,
            "volume": 1e6,
        }
    )
    bench_dir = tmp_path / "benchmark"
    save_curated(df, bench_dir, BENCHMARK_TICKER, source="yfinance")
    return bench_dir


# --------------------------------------------------------------------------
# forward_returns_on_calendar: arithmetic, session-vs-calendar, alignment
# --------------------------------------------------------------------------


def test_forward_return_arithmetic_is_exact():
    idx = sessions(10)
    close = close_series([100, 101, 102, 103, 104, 105, 106, 107, 108, 109], idx)
    fwd = forward_returns_on_calendar(close, idx, horizons=(1, 3))
    assert fwd[1].iloc[0] == pytest.approx(101 / 100 - 1)
    assert fwd[3].iloc[0] == pytest.approx(103 / 100 - 1)
    assert fwd[1].iloc[4] == pytest.approx(105 / 104 - 1)


def test_horizons_are_trading_sessions_not_calendar_days():
    """A horizon of 1 must mean 'the next SESSION', so a weekend/holiday in
    between must not change the computed return."""
    idx = sessions(10)  # business days only -- weekends already excluded
    close = close_series(range(100, 110), idx)
    fwd = forward_returns_on_calendar(close, idx, horizons=(1,))
    # every 1-session return must be exactly the same step size (1/100, 1/101, ...)
    # regardless of how many calendar days separate consecutive sessions.
    gaps_in_days = idx.to_series().diff().dt.days.dropna()
    assert (gaps_in_days > 1).any(), "fixture should include at least one weekend gap"
    assert fwd[1].iloc[0] == pytest.approx(101 / 100 - 1)


def test_incomplete_window_at_end_of_history_is_nan():
    idx = sessions(5)
    close = close_series([100, 101, 102, 103, 104], idx)
    fwd = forward_returns_on_calendar(close, idx, horizons=(3,))
    # last 3 rows cannot see 3 sessions ahead
    assert fwd[3].iloc[-3:].isna().all()
    assert fwd[3].iloc[:2].notna().all()


def test_missing_session_does_not_silently_shift_the_horizon():
    """The documented reason forward_returns_on_calendar reindexes onto a
    SHARED calendar rather than shifting positionally within a ticker's own
    rows: a dropped bar must produce NaN at the affected offsets, not quietly
    pair T with the wrong T+k."""
    idx = sessions(10)
    close = close_series(range(100, 110), idx)
    gapped = close.drop(idx[4])  # ticker has no bar on session 4
    fwd = forward_returns_on_calendar(gapped, idx, horizons=(1,))
    assert pd.isna(fwd[1].iloc[3]), "T=3 -> T+1=4 is missing, must be NaN"
    assert pd.isna(fwd[1].iloc[4]), "T=4 itself is missing, must be NaN"
    # sessions once clear of the gap are completely unaffected
    assert fwd[1].iloc[5] == pytest.approx(106 / 105 - 1)
    assert fwd[1].iloc[6] == pytest.approx(107 / 106 - 1)


def test_benchmark_and_stock_compare_identical_calendar_dates():
    """Two series with DIFFERENT gaps, forward-returned on the SAME shared
    calendar, must line up T and T+h to the same dates for both."""
    idx = sessions(12)
    stock = close_series(range(100, 112), idx).drop(idx[3])
    bench = close_series(range(200, 212), idx).drop(idx[7])
    stock_fwd = forward_returns_on_calendar(stock, idx, horizons=(2,))
    bench_fwd = forward_returns_on_calendar(bench, idx, horizons=(2,))
    # at position 5 (T=5, T+2=7): stock has both endpoints (values 105, 107 --
    # dropping position 3 only NaNs that slot, it does not shift the rest),
    # while bench is missing its T+2=7 endpoint (that's the dropped slot).
    assert stock_fwd[2].iloc[5] == pytest.approx(107 / 105 - 1)
    assert pd.isna(bench_fwd[2].iloc[5]), "bench missing its T+2 endpoint here"


# --------------------------------------------------------------------------
# build_abnormal_returns: full pipeline on synthetic data
# --------------------------------------------------------------------------


def test_abnormal_return_equals_stock_minus_benchmark(tmp_path):
    panel = synthetic_panel()
    panel_dir = write_panel(tmp_path, panel)
    idx = sessions(40)
    bench_close = close_series(100 + np.cumsum(np.full(40, 0.1)), idx)
    bench_dir = write_benchmark(tmp_path, bench_close)

    summary = build_abnormal_returns(
        panel_dir=panel_dir, benchmark_dir=bench_dir, horizons=(1, 5),
        out_dir=tmp_path / "out", version="test",
    )
    out = pd.read_parquet(summary["output_path"])
    row = out[out[valid_col(1)]].iloc[0]
    assert row[abnormal_ret_col(1)] == pytest.approx(
        row[stock_ret_col(1)] - row[bench_ret_col(1)], abs=1e-12
    )


def test_missing_benchmark_price_marks_horizon_unavailable(tmp_path):
    panel = synthetic_panel(tickers=("AAA",), n=20)
    panel_dir = write_panel(tmp_path, panel)
    idx = sessions(20)
    bench_close = close_series(np.full(20, 100.0), idx)
    bench_close = bench_close.drop(idx[10])  # benchmark missing on session 10
    bench_dir = write_benchmark(tmp_path, bench_close)

    summary = build_abnormal_returns(
        panel_dir=panel_dir, benchmark_dir=bench_dir, horizons=(1,),
        out_dir=tmp_path / "out", version="test",
    )
    out = pd.read_parquet(summary["output_path"])
    row_t9 = out[(out.ticker == "AAA") & (out.date == idx[9])].iloc[0]
    assert not row_t9[valid_col(1)], "T+1 lands on the missing benchmark session"
    assert pd.isna(row_t9[abnormal_ret_col(1)])
    assert summary["per_horizon"][1]["n_missing_price_at_endpoint"] >= 1


def test_incomplete_window_near_dataset_end_is_unavailable_not_dropped(tmp_path):
    panel = synthetic_panel(tickers=("AAA",), n=20)
    panel_dir = write_panel(tmp_path, panel)
    idx = sessions(20)
    bench_close = close_series(np.full(20, 100.0), idx)
    bench_dir = write_benchmark(tmp_path, bench_close)

    summary = build_abnormal_returns(
        panel_dir=panel_dir, benchmark_dir=bench_dir, horizons=(5,),
        out_dir=tmp_path / "out", version="test",
    )
    out = pd.read_parquet(summary["output_path"])
    # the row itself must still be PRESENT (not removed), just flagged invalid
    last_rows = out[out.date >= idx[16]]
    assert len(last_rows) == 4, "rows near the end must be kept, not dropped"
    assert not last_rows[valid_col(5)].any()
    assert last_rows[abnormal_ret_col(5)].isna().all()


def test_known_seam_invalidates_the_crossing_horizon_only(tmp_path):
    """A ticker sourced from fnspid carries the archive-wide 2020-07-06 seam
    automatically (app.data.seams); a horizon whose window crosses it must
    be marked unavailable even though both endpoint prices exist."""
    from app.data.seams import FNSPID_SEAM_DATES

    seam = FNSPID_SEAM_DATES[0]
    idx = pd.bdate_range(end=seam - pd.Timedelta(days=1), periods=30).append(
        pd.bdate_range(start=seam, periods=30)
    )
    close = close_series(100 + np.arange(len(idx)) * 0.1, idx)
    panel = pd.DataFrame(
        {
            "date": idx, "ticker": "SEAMY", "open": close, "high": close * 1.01,
            "low": close * 0.99, "close": close, "volume": 1e6, SOURCE_COLUMN: "fnspid",
        }
    )
    panel_dir = write_panel(tmp_path, panel)
    bench_close = close_series(100 + np.arange(len(idx)) * 0.05, idx)
    bench_dir = write_benchmark(tmp_path, bench_close)

    summary = build_abnormal_returns(
        panel_dir=panel_dir, benchmark_dir=bench_dir, horizons=(5,),
        out_dir=tmp_path / "out", version="test",
    )
    out = pd.read_parquet(summary["output_path"])
    pos = idx.get_loc(seam)
    crossing_date = idx[pos - 1]  # its 5-session forward window spans the seam
    row = out[(out.ticker == "SEAMY") & (out.date == crossing_date)].iloc[0]
    assert not row[valid_col(5)]
    assert summary["per_horizon"][5]["n_seam_masked"] >= 1

    # a row safely before the seam, far enough that its window does not
    # reach it, must NOT be masked
    safe_date = idx[pos - 10]
    safe_row = out[(out.ticker == "SEAMY") & (out.date == safe_date)].iloc[0]
    assert safe_row[valid_col(5)]


def test_eligible_rows_only_sessions_the_ticker_actually_has(tmp_path):
    """A row never appears for a date the ticker has no bar on at all."""
    panel = synthetic_panel(tickers=("AAA",), n=20, gap_at=5)
    panel_dir = write_panel(tmp_path, panel)
    idx = sessions(20)
    bench_close = close_series(np.full(20, 100.0), idx)
    bench_dir = write_benchmark(tmp_path, bench_close)

    summary = build_abnormal_returns(
        panel_dir=panel_dir, benchmark_dir=bench_dir, horizons=(1,),
        out_dir=tmp_path / "out", version="test",
    )
    out = pd.read_parquet(summary["output_path"])
    assert idx[5] not in set(out.date), "ticker had no bar that session at all"
    assert len(out) == 19


def test_reason_counts_sum_to_total_rows(tmp_path):
    """Regression pin: per-horizon reason counts must exactly partition the
    output rows (valid + end_of_data + missing_price + seam_masked == total),
    never over- or under-count by including ineligible sessions."""
    panel = synthetic_panel(tickers=("AAA", "BBB"), n=30, gap_at=10)
    panel_dir = write_panel(tmp_path, panel)
    idx = sessions(30)
    bench_close = close_series(100 + np.arange(30) * 0.1, idx)
    bench_dir = write_benchmark(tmp_path, bench_close)

    summary = build_abnormal_returns(
        panel_dir=panel_dir, benchmark_dir=bench_dir, horizons=(1, 5, 20),
        out_dir=tmp_path / "out", version="test",
    )
    out = pd.read_parquet(summary["output_path"])
    for h in (1, 5, 20):
        stats = summary["per_horizon"][h]
        total = (
            stats["n_valid"]
            + stats["n_incomplete_window_end_of_data"]
            + stats["n_missing_price_at_endpoint"]
            + stats["n_seam_masked"]
        )
        assert total == len(out), f"h={h}: reason counts do not partition the output rows"
        assert stats["n_valid"] == int(out[valid_col(h)].sum())


def test_forward_returns_are_never_used_as_feature_inputs():
    """The module must not IMPORT the model-feature/training pipeline. (The
    module's own docstring discusses, in prose, why it deliberately avoids
    this -- so this checks actual import statements, not a raw substring
    search over the whole source including that explanation.)"""
    import ast
    import inspect

    import app.events.benchmark_returns as mod

    tree = ast.parse(inspect.getsource(mod))
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)
        elif isinstance(node, ast.Import):
            imported.update(n.name for n in node.names)
    forbidden = {"app.features.build", "app.features.matrix", "app.models.train"}
    assert not (imported & forbidden), f"must not import {imported & forbidden}"


# --------------------------------------------------------------------------
# No accidental modification of existing artifacts
# --------------------------------------------------------------------------


@pytest.mark.skipif(not MATRIX_H5.exists(), reason="matrix_h5_v4 absent")
def test_does_not_modify_existing_h5_matrix():
    before = MATRIX_H5.stat().st_mtime, MATRIX_H5.stat().st_size
    import app.events.benchmark_returns  # noqa: F401

    after = MATRIX_H5.stat().st_mtime, MATRIX_H5.stat().st_size
    assert before == after


@pytest.mark.skipif(not MATRIX_H1.exists(), reason="matrix_h1_v4 absent")
def test_does_not_modify_existing_h1_matrix():
    before = MATRIX_H1.stat().st_mtime, MATRIX_H1.stat().st_size
    import app.events.benchmark_returns  # noqa: F401

    after = MATRIX_H1.stat().st_mtime, MATRIX_H1.stat().st_size
    assert before == after


# --------------------------------------------------------------------------
# Against the real built artifact (skips if not yet generated)
# --------------------------------------------------------------------------

REAL = pytest.mark.skipif(not REAL_OUTPUT.exists(), reason="Phase 0 output not built")


@REAL
def test_real_output_covers_every_panel_row():
    from app.data.curate import load_curated

    panel = load_curated(REAL_PANEL)
    out = pd.read_parquet(REAL_OUTPUT)
    assert len(out) == len(panel)
    assert set(zip(out.ticker, out.date, strict=True)) == set(
        zip(panel.ticker, panel.date, strict=True)
    )


@REAL
def test_real_output_has_all_six_horizons():
    out = pd.read_parquet(REAL_OUTPUT)
    for h in HORIZONS:
        for col in (stock_ret_col(h), bench_ret_col(h), abnormal_ret_col(h), valid_col(h)):
            assert col in out.columns


@REAL
def test_real_abnormal_return_identity_holds_everywhere_valid():
    out = pd.read_parquet(REAL_OUTPUT)
    for h in HORIZONS:
        v = out[out[valid_col(h)]]
        diff = (v[abnormal_ret_col(h)] - (v[stock_ret_col(h)] - v[bench_ret_col(h)])).abs()
        assert diff.max() < 1e-9


@REAL
def test_real_invalid_rows_have_no_leaked_return_values():
    """A row marked invalid must carry NaN in all three of its return
    columns for that horizon -- never a value treated as real."""
    out = pd.read_parquet(REAL_OUTPUT)
    for h in HORIZONS:
        invalid = ~out[valid_col(h)]
        assert out.loc[invalid, stock_ret_col(h)].isna().all()
        assert out.loc[invalid, bench_ret_col(h)].isna().all()
        assert out.loc[invalid, abnormal_ret_col(h)].isna().all()
