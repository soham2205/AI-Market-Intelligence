from __future__ import annotations

import pandas as pd

from app.data.sources.base import REQUIRED_OHLCV


class YFinanceSource:
    name = "yfinance"

    def fetch(self, ticker: str, start: str, end: str | None) -> pd.DataFrame:
        import yfinance as yf

        raw = yf.download(
            ticker,
            start=start,
            end=end,
            auto_adjust=True,
            progress=False,
        )
        if raw is None or raw.empty:
            return pd.DataFrame(columns=REQUIRED_OHLCV)

        if isinstance(raw.columns, pd.MultiIndex):
            first = raw.columns.get_level_values(0)
            second = raw.columns.get_level_values(1)
            level = first if set(first.str.lower()) & set(REQUIRED_OHLCV) else second
            raw.columns = level
        df = raw.rename(columns=lambda c: str(c).lower())
        df.index = pd.to_datetime(df.index).tz_localize(None).normalize()
        df.index.name = "date"
        return df[[c for c in REQUIRED_OHLCV if c in df.columns]]
