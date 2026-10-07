from __future__ import annotations

import pandas as pd


def aggregate_daily(scored_news: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per-article sentiment to per-(ticker, trade_date) features."""
    df = scored_news.copy()
    grouped = df.groupby(["ticker", "trade_date"])
    out = pd.DataFrame(
        {
            "n_articles": grouped.size(),
            "sent_mean": grouped["score"].mean(),
            "sent_pos_share": grouped["score"].apply(lambda s: float((s > 0).mean())),
            "sent_neg_share": grouped["score"].apply(lambda s: float((s < 0).mean())),
            "sent_dispersion": grouped["score"].std(),
        }
    ).reset_index()
    out["sent_dispersion"] = out["sent_dispersion"].fillna(0.0)
    return out.sort_values(["ticker", "trade_date"]).reset_index(drop=True)


def add_ewma(sentiment_daily: pd.DataFrame, span: int = 5) -> pd.DataFrame:
    """Carry a decaying sentiment level across days with no coverage."""
    out = sentiment_daily.sort_values(["ticker", "trade_date"]).copy()
    out["_masked"] = out["sent_mean"].where(out["n_articles"] > 0)
    out["ewma_sent"] = (
        out.groupby("ticker")["_masked"]
        .transform(lambda s: s.ewm(span=span, adjust=False).mean().ffill())
    )
    return out.drop(columns=["_masked"])
