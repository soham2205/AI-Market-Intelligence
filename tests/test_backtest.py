from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.backtest.baselines import buy_hold_returns, sma_crossover_predictions
from app.backtest.engine import performance_metrics, run_backtest


def make_prices(closes, opens):
    dates = pd.bdate_range("2024-01-01", periods=len(closes))
    return pd.DataFrame(
        {
            "date": dates,
            "ticker": "A",
            "open": opens,
            "high": np.maximum(opens, closes),
            "low": np.minimum(opens, closes),
            "close": closes,
            "volume": 1000.0,
        }
    )


def make_preds(dates, probas, ticker="A"):
    n = len(probas)
    return pd.DataFrame(
        {"date": list(dates)[:n], "ticker": ticker, "proba": probas}
    )


def test_entry_earns_open_to_close():
    prices = make_prices([100.0, 110.0, 100.0], [100.0, 105.0, 105.0])
    preds = make_preds(prices["date"], [0.9, 0.9])
    result = run_backtest(preds, prices, cost_bps=0)
    expected_total = (110 / 105) * (100 / 110) - 1
    assert result["metrics"]["total_return"] == pytest.approx(expected_total)


def test_exit_fills_at_open_and_costs_charged():
    prices = make_prices([100.0, 110.0, 100.0], [100.0, 105.0, 105.0])
    preds = make_preds(prices["date"], [0.9, 0.1])
    result = run_backtest(preds, prices, cost_bps=100)
    assert result["metrics"]["total_cost_drag"] == pytest.approx(2 * 0.01)
    a, b, c = 110 / 105, 105 / 110, 0.01
    assert result["metrics"]["total_return"] == pytest.approx((a - c) * (b - c) - 1)


def test_no_signal_means_flat():
    prices = make_prices([100.0, 101.0, 102.0], [100.0, 101.0, 102.0])
    preds = make_preds(prices["date"], [0.3, 0.3])
    result = run_backtest(preds, prices, entry_threshold=0.55, cost_bps=0)
    assert result["metrics"]["total_return"] == pytest.approx(0.0)
    assert result["positions"].to_numpy().sum() == 0


def test_performance_metrics_math():
    r = np.array([0.01, -0.01, 0.02, 0.0, 0.01])
    dates = pd.bdate_range("2024-01-01", periods=len(r))
    m = performance_metrics(r, dates)
    equity = np.cumprod(1 + r)
    assert m["total_return"] == pytest.approx(equity[-1] - 1)
    assert m["sharpe"] == pytest.approx(
        float(np.mean(r) / np.std(r, ddof=1) * np.sqrt(252))
    )
    peak = np.maximum.accumulate(equity)
    assert m["max_drawdown"] == pytest.approx(float(np.min(equity / peak - 1)))


def test_buy_hold_baseline(make_panel):
    panel = make_panel(tickers=("AAPL", "MSFT"), n=60)
    bh = buy_hold_returns(panel)
    px = panel.pivot(index="date", columns="ticker", values="close")
    expected = px.pct_change().fillna(0).mean(axis=1)
    assert np.allclose(bh.to_numpy(), expected.to_numpy())


def test_sma_baseline_signals_are_zero_or_one(make_panel):
    panel = make_panel(n=80)
    preds = sma_crossover_predictions(panel)
    assert set(preds["proba"].unique()).issubset({0.0, 1.0})
    assert len(preds) > 0
