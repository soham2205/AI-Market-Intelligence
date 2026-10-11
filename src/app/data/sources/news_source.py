from __future__ import annotations

import hashlib

import pandas as pd

from app.data.sources.base import NEWS_SCHEMA


class YFinanceNewsSource:
    """Ticker news headlines via yfinance (no API key required).

    Live/recent only -- the endpoint returns roughly the last two days, so
    this is an inference-time source, not a source of training history.
    """

    name = "yfinance"

    def fetch(
        self,
        ticker: str,
        start: str | None = None,
        end: str | None = None,
    ) -> pd.DataFrame:
        import yfinance as yf

        items = yf.Ticker(ticker).news or []
        rows = []
        for item in items:
            content = item.get("content", item)
            title = content.get("headline") or content.get("title") or item.get("title")
            # Current yfinance exposes pubDate/displayTime; the *Iso variants
            # are kept as fallbacks so older payloads still parse.
            published = (
                content.get("pubDate")
                or content.get("displayTime")
                or content.get("pubIsoDate")
                or content.get("displayTimeIso")
            )
            epoch = content.get("providerPublishTime") or item.get("providerPublishTime")
            if not title:
                continue
            if published is None and epoch is not None:
                published = pd.Timestamp(epoch, unit="s", tz="UTC").isoformat()
            if published is None:
                continue
            url = (content.get("clickThroughUrl") or content.get("canonicalUrl") or {}).get(
                "url", ""
            )
            source = (content.get("provider") or {}).get("displayName", "unknown")
            article_id = hashlib.sha1(f"{url}|{title}".encode()).hexdigest()[:16]
            rows.append(
                {
                    "article_id": article_id,
                    "headline": title,
                    "source": source,
                    "url": url,
                    "published_at": pd.Timestamp(published),
                    "ticker": ticker.upper(),
                }
            )
        df = pd.DataFrame(rows, columns=NEWS_SCHEMA)
        if len(df) and (start is not None or end is not None):
            ts = pd.to_datetime(df["published_at"], utc=True, errors="coerce")
            keep = ts.notna()
            if start is not None:
                keep &= ts >= pd.Timestamp(start, tz="UTC")
            if end is not None:
                keep &= ts <= pd.Timestamp(end, tz="UTC")
            df = df[keep].reset_index(drop=True)
        return df


def dedupe_news(news: pd.DataFrame) -> pd.DataFrame:
    return news.drop_duplicates(subset=["article_id"]).reset_index(drop=True)
