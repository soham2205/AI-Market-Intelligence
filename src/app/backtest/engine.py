from __future__ import annotations

import numpy as np
import pandas as pd


def run_backtest(
    predictions: pd.DataFrame,
    prices: pd.DataFrame,
    entry_threshold: float = 0.55,
    exit_threshold: float = 0.50,
    cost_bps: float = 10.0,
) -> dict:
    """Event-driven daily backtest with T+1 execution.

    Signal formed on day T (info <= close T) changes the position for day
    T+1. Entry fills at the T+1 open (earning open->close that day), exits
    fill at the T+1 open (earning only the overnight leg). Costs are charged
    on position turnover.
    """
    px = prices.pivot(index="date", columns="ticker", values="close").sort_index()
    opx = prices.pivot(index="date", columns="ticker", values="open").sort_index()
    sig = (
        predictions.pivot(index="date", columns="ticker", values="proba")
        .reindex(px.index)
        .sort_index()
    )
    first_signal = sig.dropna(how="all").index.min()
    if first_signal is not None:
        px = px.loc[px.index >= first_signal]
        opx = opx.loc[opx.index >= first_signal]
        sig = sig.loc[sig.index >= first_signal]
    tickers = px.columns
    dates = px.index
    n_assets = len(tickers)
    cost_rate = cost_bps / 1e4

    pxc, opc, sigv = (
        px.to_numpy(dtype=float),
        opx.reindex(columns=tickers).to_numpy(dtype=float),
        sig.reindex(columns=tickers).to_numpy(dtype=float),
    )

    positions = np.zeros_like(pxc)
    for t in range(len(dates) - 1):
        prev = positions[t]
        new = prev.copy()
        today = sigv[t]
        long_mask = today >= entry_threshold
        flat_mask = today <= exit_threshold
        new[long_mask] = 1.0
        new[flat_mask & ~long_mask] = 0.0
        new[np.isnan(today)] = prev[np.isnan(today)]
        positions[t + 1] = new

    asset_ret = np.zeros_like(pxc)
    turnover = np.zeros(len(dates))
    for t in range(1, len(dates)):
        prev, cur = positions[t - 1], positions[t]
        with np.errstate(invalid="ignore", divide="ignore"):
            overnight = np.where(pxc[t - 1] > 0, opc[t] / pxc[t - 1] - 1, 0.0)
            intraday = np.where(opc[t] > 0, pxc[t] / opc[t] - 1, 0.0)
            cc = np.where(pxc[t - 1] > 0, pxc[t] / pxc[t - 1] - 1, 0.0)
        entered = (prev == 0) & (cur == 1)
        exited = (prev == 1) & (cur == 0)
        held = (prev == 1) & (cur == 1)
        day_ret = np.where(entered, intraday, np.where(exited, overnight, cc * held))
        asset_ret[t] = np.nan_to_num(day_ret, nan=0.0, posinf=0.0, neginf=0.0)
        turnover[t] = np.abs(cur - prev).sum() / n_assets

    costs = turnover * cost_rate
    port_ret = asset_ret.mean(axis=1) - costs

    equity = (1 + pd.Series(port_ret, index=dates)).cumprod()
    metrics = performance_metrics(port_ret, dates)
    metrics["total_cost_drag"] = float(costs.sum())
    metrics["avg_daily_turnover"] = float(turnover.mean())

    return {
        "metrics": metrics,
        "equity_curve": equity,
        "daily_returns": pd.Series(port_ret, index=dates),
        "positions": pd.DataFrame(positions, index=dates, columns=tickers),
    }


def performance_metrics(daily_returns: np.ndarray, dates: pd.DatetimeIndex) -> dict:
    r = np.asarray(daily_returns, dtype=float)
    equity = np.cumprod(1 + r)
    total_return = float(equity[-1] - 1)
    years = max(len(r) / 252.0, 1e-9)
    cagr = float(equity[-1] ** (1 / years) - 1)
    std = float(np.std(r, ddof=1)) if len(r) > 1 else 0.0
    sharpe = float(np.mean(r) / std * np.sqrt(252)) if std > 0 else 0.0
    downside = r[r < 0]
    dstd = float(np.std(downside, ddof=1)) if len(downside) > 1 else 0.0
    sortino = float(np.mean(r) / dstd * np.sqrt(252)) if dstd > 0 else 0.0
    peak = np.maximum.accumulate(equity)
    max_drawdown = float(np.min(equity / peak - 1))
    return {
        "total_return": total_return,
        "cagr": cagr,
        "sharpe": sharpe,
        "sortino": sortino,
        "max_drawdown": max_drawdown,
        "n_days": len(r),
    }
