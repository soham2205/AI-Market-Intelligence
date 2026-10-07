from __future__ import annotations

import pandas as pd


def buy_hold_returns(
    prices: pd.DataFrame,
    start: str | None = None,
    end: str | None = None,
) -> pd.Series:
    """Equal-weight buy-and-hold over the universe (close-to-close)."""
    px = prices.pivot(index="date", columns="ticker", values="close").sort_index()
    rets = px.pct_change(fill_method=None).fillna(0.0)
    if start:
        rets = rets[rets.index >= pd.Timestamp(start)]
    if end:
        rets = rets[rets.index <= pd.Timestamp(end)]
    return rets.mean(axis=1)


def sma_crossover_predictions(prices: pd.DataFrame) -> pd.DataFrame:
    """Classic SMA20>SMA50 rule expressed as pseudo-probabilities (0/1).

    Signals use data through day T; the backtest engine executes at T+1.
    """
    rows = []
    for ticker, g in prices.sort_values("date").groupby("ticker"):
        close = g.set_index("date")["close"]
        sma20 = close.rolling(20).mean()
        sma50 = close.rolling(50).mean()
        proba = (sma20 > sma50).astype(float)
        frame = pd.DataFrame({"date": proba.index, "ticker": ticker, "proba": proba.values})
        rows.append(frame.dropna())
    return pd.concat(rows, ignore_index=True)
