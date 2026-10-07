from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def make_ohlcv():
    def _factory(n: int = 10, start: str = "2024-01-01", seed: int = 42) -> pd.DataFrame:
        rng = np.random.default_rng(seed)
        dates = pd.bdate_range(start, periods=n)
        close = 100 + np.cumsum(rng.normal(0, 1, n))
        open_ = close + rng.normal(0, 0.5, n)
        high = np.maximum(open_, close) + np.abs(rng.normal(0.5, 0.1, n))
        low = np.minimum(open_, close) - np.abs(rng.normal(0.5, 0.1, n))
        volume = rng.integers(1_000_000, 5_000_000, n).astype(float)
        df = pd.DataFrame(
            {
                "open": open_,
                "high": high,
                "low": low,
                "close": close,
                "volume": volume,
            },
            index=dates,
        )
        df.index.name = "date"
        return df

    return _factory


@pytest.fixture
def make_panel():
    """Synthetic multi-ticker curated price panel with a 'date' column."""

    def _factory(
        tickers: tuple[str, ...] = ("AAPL", "MSFT"),
        n: int = 120,
        start: str = "2023-01-02",
    ) -> pd.DataFrame:
        frames = []
        for i, ticker in enumerate(tickers):
            rng = np.random.default_rng(i)
            dates = pd.bdate_range(start, periods=n)
            close = 100 + np.abs(np.cumsum(rng.normal(0.05, 1.2, n)))
            open_ = close * (1 + rng.normal(0, 0.002, n))
            high = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, 0.003, n)))
            low = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, 0.003, n)))
            volume = rng.integers(1_000_000, 5_000_000, n).astype(float)
            frames.append(
                pd.DataFrame(
                    {
                        "date": dates,
                        "ticker": ticker,
                        "open": open_,
                        "high": high,
                        "low": low,
                        "close": close,
                        "volume": volume,
                    }
                )
            )
        return pd.concat(frames, ignore_index=True)

    return _factory
