from __future__ import annotations

from typing import Protocol

import pandas as pd

REQUIRED_OHLCV = ["open", "high", "low", "close", "volume"]

# Canonical news schema every NewsSource must return, in this order.
NEWS_SCHEMA = ["article_id", "headline", "source", "url", "published_at", "ticker"]


class PriceSource(Protocol):
    def fetch(self, ticker: str, start: str, end: str | None) -> pd.DataFrame:
        """Return OHLCV rows indexed by date for a single ticker."""
        ...


class NewsSource(Protocol):
    def fetch(
        self, ticker: str, start: str | None = None, end: str | None = None
    ) -> pd.DataFrame:
        """Return NEWS_SCHEMA rows for one ticker within [start, end].

        `published_at` must be a real, tz-aware publication timestamp taken
        from the source. Never synthesise one: an article whose timestamp
        cannot be recovered is dropped, because a fabricated timestamp is a
        look-ahead bug waiting to happen.
        """
        ...
