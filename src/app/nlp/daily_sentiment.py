"""Article-level FinBERT sentiment -> daily ticker-level features.

    news_sentiment.parquet (one row per article)
            |
            |  first_tradable_session()   <- the project's existing convention
            v
    (ticker, trade_date) groups
            |
            v
    news_sentiment_daily.parquet (five numeric features)

Session assignment follows nlp.bucketing verbatim rather than inventing a new
rule: news published before 09:30 ET lands on that session, anything at or
after the open lands on the NEXT session, and weekends and market holidays
roll forward against the real trading calendar taken from the curated price
panel. That is what keeps a headline from being attributed to a session that
had already opened when it was published.

This module deliberately does NOT reuse nlp.aggregate.aggregate_daily. That
function produces a different feature set (sent_mean / sent_pos_share / ...)
whose shares are computed from the sign of the score, and it is wired into the
live yfinance ingest path. The features here are defined on the FinBERT LABEL
instead, so they are a separate calculation with separate names.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from app.nlp.bucketing import ET, first_tradable_session

LABELS = ("positive", "negative", "neutral")

DAILY_COLUMNS = [
    "ticker",
    "trade_date",
    "mean_sentiment",
    "positive_ratio",
    "negative_ratio",
    "neutral_ratio",
    "article_count",
]

REQUIRED_ARTICLE_COLUMNS = ["ticker", "published_at", "label", "score"]

DEFAULT_ARTICLE_PATH = Path("data/curated/news_sentiment.parquet")
DEFAULT_OUTPUT_PATH = Path("data/curated/news_sentiment_daily.parquet")


def session_calendar(curated_dir: str | Path = "data/curated") -> pd.DatetimeIndex:
    """Trading sessions, taken from the curated price panel.

    Same construction the CLI uses for news ingestion, so both paths agree on
    what counts as a session.
    """
    from app.data.curate import load_curated

    curated = load_curated(curated_dir)
    if curated.empty:
        raise ValueError(f"no curated prices under {curated_dir}; cannot build a calendar")
    return pd.DatetimeIndex(
        pd.to_datetime(curated["date"]).dt.normalize().unique()
    ).sort_values()


def assign_trade_date(
    articles: pd.DataFrame,
    sessions: pd.DatetimeIndex,
) -> tuple[pd.DataFrame, dict]:
    """Attach the session each article is first actionable in.

    Two classes of article cannot be placed and are dropped rather than
    guessed at:

    * published AFTER the last known session -- there is no session yet, which
      `first_tradable_session` already signals with NaT.
    * published BEFORE the first known session -- `_roll_to_session` would
      snap every one of them onto the calendar's opening day, inventing a
      sentiment spike there. Anything that old is outside the modelling panel
      anyway, so it is discarded explicitly instead of silently piling up.
    """
    missing = [c for c in REQUIRED_ARTICLE_COLUMNS if c not in articles.columns]
    if missing:
        raise KeyError(f"article frame is missing columns {missing}")
    if len(sessions) == 0:
        raise ValueError("session calendar is empty")

    work = articles.copy()
    published = pd.to_datetime(work["published_at"], utc=True)
    work["published_at"] = published

    local_day = published.dt.tz_convert(ET).dt.normalize().dt.tz_localize(None)
    before_calendar = local_day < sessions[0]
    n_before = int(before_calendar.sum())
    work = work[~before_calendar]

    work["trade_date"] = [
        first_tradable_session(ts, sessions) for ts in work["published_at"]
    ]
    after_calendar = work["trade_date"].isna()
    n_after = int(after_calendar.sum())
    work = work[~after_calendar].copy()
    work["trade_date"] = pd.to_datetime(work["trade_date"])

    stats = {
        "articles_in": len(articles),
        "dropped_before_calendar": n_before,
        "dropped_after_calendar": n_after,
        "articles_assigned": len(work),
    }
    return work, stats


def aggregate_daily_sentiment(
    articles: pd.DataFrame,
    sessions: pd.DatetimeIndex,
) -> tuple[pd.DataFrame, dict]:
    """Collapse article-level sentiment to one row per (ticker, trade_date).

    Five numeric features, nothing else:
      mean_sentiment  mean of the article-level score
      positive_ratio  share of articles whose FinBERT label is positive
      negative_ratio  share whose label is negative
      neutral_ratio   share whose label is neutral
      article_count   number of articles
    """
    assigned, stats = assign_trade_date(articles, sessions)

    unknown = set(assigned["label"].dropna().unique()) - set(LABELS)
    if unknown:
        raise ValueError(
            f"unexpected FinBERT labels {sorted(unknown)}; the three ratios "
            "would not sum to 1"
        )

    work = assigned
    for label in LABELS:
        work[f"_is_{label}"] = (work["label"] == label).astype(float)

    grouped = work.groupby(["ticker", "trade_date"], sort=True)
    daily = grouped.agg(
        mean_sentiment=("score", "mean"),
        positive_ratio=("_is_positive", "mean"),
        negative_ratio=("_is_negative", "mean"),
        neutral_ratio=("_is_neutral", "mean"),
        article_count=("score", "size"),
    ).reset_index()

    daily["article_count"] = daily["article_count"].astype("int64")
    for col in ("mean_sentiment", "positive_ratio", "negative_ratio", "neutral_ratio"):
        daily[col] = daily[col].astype("float64")

    daily = daily[DAILY_COLUMNS].sort_values(["ticker", "trade_date"]).reset_index(drop=True)

    stats.update(
        {
            "ticker_days": len(daily),
            "tickers": int(daily["ticker"].nunique()),
            "articles_represented": int(daily["article_count"].sum()),
            "earliest_session": str(daily["trade_date"].min().date()) if len(daily) else None,
            "latest_session": str(daily["trade_date"].max().date()) if len(daily) else None,
        }
    )
    return daily, stats


def run(
    article_path: str | Path = DEFAULT_ARTICLE_PATH,
    curated_dir: str | Path = "data/curated",
    output_path: str | Path = DEFAULT_OUTPUT_PATH,
) -> dict:
    """Build the daily sentiment artifact and persist it."""
    articles = pd.read_parquet(article_path)
    sessions = session_calendar(curated_dir)
    daily, stats = aggregate_daily_sentiment(articles, sessions)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    daily.to_parquet(output_path, index=False)

    stats["output_path"] = str(output_path)
    stats["output_bytes"] = output_path.stat().st_size
    stats["sessions_in_calendar"] = len(sessions)
    return stats
