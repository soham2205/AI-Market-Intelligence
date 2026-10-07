"""Combined historical training matrix: technical features + daily news.

    expanded price panel (data/curated_sp500)
            |  build_features()  <- existing engine, per-source seam handling
            v
    technical features + multi-horizon targets
            |  left join on (ticker, trade_date)
            v
    + five daily news features (zero-filled on no-news sessions)
            v
    matrix_h{K}_v1.parquet

Design decisions, all deliberate:

* Ticker identity is NOT a feature. It is kept as metadata for joins and
  reporting only. A one-hot over 447 tickers could not generalise to a ticker
  the model never saw, which is the whole point of a pooled model, and it
  would make the dense design matrix ~5 GB.
* Only scale-free features reach the model. Raw SMA levels, MACD, ATR and OBV
  delta vary by 700x-2100x between a $5 and a $1,600 stock; their scale-free
  counterparts (close_to_sma20, macd_norm, atr_pct, obv_delta_norm, ...) carry
  the same information. See technical.SCALE_DEPENDENT.
* No-news sessions keep their price observation and receive ZERO for all five
  news features. Zero is the honest encoding here: article_count 0 is
  literally true, and the three ratios plus mean_sentiment describe "no
  articles expressed any sentiment". This differs from the NaN-and-impute
  convention used by the older 9-ticker path, and is what requirement 3 asks
  for.
* Labels and purge come from the existing implementations unchanged
  (labeling.targets, splits.purged_cv).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from app.features.build import build_features
from app.features.technical import FEATURE_LOOKBACK, SCALE_DEPENDENT
from app.labeling.targets import fwd_ret_col, label_col

# The five daily news features, in the audited artifact's own order.
NEWS_FEATURES = [
    "mean_sentiment",
    "positive_ratio",
    "negative_ratio",
    "neutral_ratio",
    "article_count",
]

# Carried through for joins, validation and reporting -- never fed to a model.
METADATA_COLUMNS = ["ticker", "date", "source"]

DEFAULT_PANEL_DIR = "data/curated_sp500"
# The v2 audited artifact additionally excludes news tagged with a ticker
# before the current entity took that symbol (see app.data.entities). The v1
# file is kept for comparison but is known to misattribute IR and COR news.
DEFAULT_NEWS = "data/curated/news_sentiment_daily_sp500_audited_v2.parquet"
MATRIX_VERSION = "v1"


def model_feature_columns(df: pd.DataFrame) -> list[str]:
    """Scale-free predictive features present in `df`, in a stable order.

    Excludes metadata, every horizon's targets, the scale-dependent raw
    indicators, and ticker identity.
    """
    ordered = [c for c in FEATURE_LOOKBACK if c not in SCALE_DEPENDENT]
    technical = [c for c in ordered if c in df.columns]
    news = [c for c in NEWS_FEATURES if c in df.columns]
    return technical + news


def build_matrix(
    panel: pd.DataFrame,
    news_daily: pd.DataFrame,
    horizon: int = 5,
    horizons: tuple[int, ...] | None = None,
) -> tuple[pd.DataFrame, dict]:
    """Assemble the matrix for one horizon and report every exclusion.

    Returns (matrix, stats). Rows are dropped only for reasons that are
    counted and named in `stats`; nothing is discarded silently.
    """
    horizons = horizons or (horizon,)
    if horizon not in horizons:
        raise ValueError(f"horizon {horizon} must appear in horizons {horizons}")

    # Seam handling is automatic: build_features resolves seams per ticker from
    # the `source` provenance column written at curation time.
    feats = build_features(panel, horizons=horizons)
    n_sessions = len(feats)

    news = news_daily.rename(columns={"trade_date": "date"})
    missing = [c for c in NEWS_FEATURES if c not in news.columns]
    if missing:
        raise KeyError(f"news frame is missing {missing}")

    merged = feats.merge(
        news[["ticker", "date", *NEWS_FEATURES]],
        on=["ticker", "date"],
        how="left",
        validate="one_to_one",
    )
    if len(merged) != n_sessions:
        raise AssertionError(f"join changed row count: {n_sessions} -> {len(merged)}")

    matched = int(merged["article_count"].notna().sum())
    # Requirement 3: keep every price session; zero-fill the news features.
    merged[NEWS_FEATURES] = merged[NEWS_FEATURES].fillna(0.0)
    merged["article_count"] = merged["article_count"].astype("int64")
    merged["has_news"] = merged["article_count"] > 0

    if "source" in panel.columns:
        merged = merged.merge(
            panel.groupby("ticker", as_index=False)["source"].first(),
            on="ticker",
            how="left",
        )

    label, fwd = label_col(horizon), fwd_ret_col(horizon)
    features = model_feature_columns(merged)

    # --- exclusions, each counted ------------------------------------------
    no_label = merged[label].isna()
    feature_na = merged[features].isna().any(axis=1) & ~no_label
    keep = ~(no_label | feature_na)

    out = merged.loc[keep, [*METADATA_COLUMNS, *features, "has_news", fwd, label]]
    out = out.sort_values(["ticker", "date"]).reset_index(drop=True)

    pos = float(out[label].mean()) if len(out) else float("nan")
    stats = {
        "horizon": horizon,
        "price_sessions_in": n_sessions,
        "tickers_in": int(feats["ticker"].nunique()),
        "news_rows_joined": matched,
        "news_rows_available": int(len(news)),
        "sessions_with_news": int(merged["has_news"].sum()),
        "sessions_without_news": int((~merged["has_news"]).sum()),
        "excluded_no_label": int(no_label.sum()),
        "excluded_feature_nan": int(feature_na.sum()),
        "rows_out": int(len(out)),
        "tickers_out": int(out["ticker"].nunique()),
        "first_session": str(out["date"].min().date()) if len(out) else None,
        "last_session": str(out["date"].max().date()) if len(out) else None,
        "feature_columns": features,
        "n_features": len(features),
        "label_positive_rate": pos,
        "ticker_is_a_feature": "ticker" in features,
    }
    return out, stats


def feature_nan_breakdown(
    panel: pd.DataFrame, news_daily: pd.DataFrame, horizon: int = 5
) -> pd.DataFrame:
    """Which feature drives each NaN exclusion, for the exclusion report."""
    feats = build_features(panel, horizons=(horizon,))
    news = news_daily.rename(columns={"trade_date": "date"})
    merged = feats.merge(
        news[["ticker", "date", *NEWS_FEATURES]], on=["ticker", "date"], how="left"
    )
    merged[NEWS_FEATURES] = merged[NEWS_FEATURES].fillna(0.0)
    features = model_feature_columns(merged)
    sub = merged[merged[label_col(horizon)].notna()]
    counts = sub[features].isna().sum().sort_values(ascending=False)
    return counts[counts > 0].rename("rows_with_nan").to_frame()


def run(
    panel_dir: str | Path = DEFAULT_PANEL_DIR,
    news_path: str | Path = DEFAULT_NEWS,
    horizon: int = 5,
    out_dir: str | Path = "data/curated",
    version: str = MATRIX_VERSION,
) -> dict:
    """Build and persist the matrix as a new versioned artifact."""
    from app.data.curate import load_curated

    panel = load_curated(panel_dir)
    news = pd.read_parquet(news_path)
    matrix, stats = build_matrix(panel, news, horizon=horizon)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"matrix_h{horizon}_{version}.parquet"
    matrix.to_parquet(out_path, index=False)

    stats["version"] = version
    stats["output_path"] = str(out_path)
    stats["output_bytes"] = out_path.stat().st_size
    with open(out_dir / f"matrix_h{horizon}_{version}_summary.json", "w") as fh:
        json.dump({k: v for k, v in stats.items() if not isinstance(v, np.generic)}, fh, indent=1)
    return stats
