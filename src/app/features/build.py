from __future__ import annotations

from collections.abc import Sequence

import pandas as pd

from app.data.seams import forward_window_invalid, seams_for_frame
from app.features.technical import compute_indicators
from app.labeling.targets import (
    fwd_ret_col,
    is_target_column,
    label_col,
)

# Static non-feature columns. Horizon targets are excluded by PREFIX via
# is_target_column() -- never by enumeration, so a new horizon cannot leak.
STATIC_EXCLUDED = {"date", "ticker"}

# Retained for backwards compatibility with the single-horizon era.
EXCLUDED_FROM_FEATURES = STATIC_EXCLUDED | {"fwd_ret_1d", "label_up_1d"}

MAX_FEATURE_LOOKBACK = 63  # vol_63d is the deepest rolling window

SENTIMENT_COLS = [
    "sent_mean",
    "sent_pos_share",
    "sent_neg_share",
    "sent_dispersion",
    "ewma_sent",
    "n_articles",
]

# Absence of news is information ("nothing was written today"), not a missing
# measurement, so these are zero-filled after the join instead of being handed
# to the imputer. Everything else in SENTIMENT_COLS stays NaN on no-news days
# and is imputed per fold inside the model pipeline.
SENTIMENT_COUNT_COLS = ["n_articles"]


def is_feature_column(col: str) -> bool:
    """A column feeds the model iff it is neither metadata nor any target."""
    return col not in STATIC_EXCLUDED and not is_target_column(col)


def build_features(
    curated: pd.DataFrame,
    sentiment: pd.DataFrame | None = None,
    positive_threshold: float = 0.0,
    horizons: Sequence[int] = (1,),
    seam_dates: Sequence[pd.Timestamp] | None = None,
) -> pd.DataFrame:
    """Build point-in-time-correct feature matrix with multi-horizon labels.

    Convention (leak-free by construction):

    - Feature row for day T uses ONLY information observable at the close of
      day T. Rolling indicators and returns are inherently backward-looking;
      daily OHLCV is fully known at T's close.
    - Sentiment aggregates are keyed to their FIRST TRADABLE session (see
      nlp.bucketing), so joining on the same date is leak-free: overnight
      news is known before T's open, during/after-session news is bucketed
      to T+1 by the bucketing step.
    - For each horizon k the label describes the CUMULATIVE forward move:
      fwd_ret_{k}d = Close_{T+k}/Close_T - 1, and
      label_up_{k}d = 1 iff that return exceeds `positive_threshold`.
      The trailing k rows per ticker have no T+k and stay NaN.
    - Every fwd_ret_*/label_up_* column is a TARGET, never a feature. The
      model layer filters by prefix (is_feature_column), so horizon k's
      label can never enter horizon j's feature matrix.
    - Data seams (see app.data.seams) are resolved AUTOMATICALLY, per ticker,
      from the `source` provenance column that curation records. A caller
      cannot forget to pass them, and a source with no known defect resolves
      to no seams, so a clean yfinance series is untouched. Passing
      `seam_dates` explicitly overrides the lookup for every ticker; pass
      NO_SEAMS to force masking off.
      Indicators whose lookback spans a seam, and labels whose forward window
      spans one, are set to NaN. Prices are never altered and no row is
      dropped.
    """
    frames: list[pd.DataFrame] = []
    for ticker, g in curated.groupby("ticker"):
        g = g.sort_values("date").set_index("date")
        # Resolved per ticker: a panel may mix providers, and a yfinance
        # ticker must not inherit another provider's seam.
        seams = seams_for_frame(g) if seam_dates is None else tuple(seam_dates)
        feats = compute_indicators(g, seam_dates=seams)
        feats["day_of_week"] = g.index.dayofweek
        feats["ticker"] = ticker

        close = g["close"]
        for k in horizons:
            fwd = close.shift(-k) / close - 1
            # A forward return that steps across a seam is not a real return.
            spans = forward_window_invalid(g.index, k, seams)
            if spans.any():
                fwd = fwd.mask(spans)
            feats[fwd_ret_col(k)] = fwd
            feats[label_col(k)] = (
                (fwd > positive_threshold).astype(float).where(fwd.notna())
            )

        if sentiment is not None:
            s = sentiment[sentiment["ticker"] == ticker]
            # Tolerate a sentiment table built without add_ewma (or any other
            # subset) rather than raising on a missing column.
            cols = [c for c in SENTIMENT_COLS if c in s.columns]
            s = s.sort_values("trade_date").set_index("trade_date")[cols]
            feats = feats.join(s, how="left")
            # AFTER the join: sessions with no news row get a real zero.
            for col in SENTIMENT_COUNT_COLS:
                if col in feats.columns:
                    feats[col] = feats[col].fillna(0)

        feats["date"] = feats.index
        frames.append(feats.reset_index(drop=True))

    out = pd.concat(frames, ignore_index=True)
    target_cols = [c for k in horizons for c in (fwd_ret_col(k), label_col(k))]
    feature_cols = [c for c in out.columns if is_feature_column(c)]
    ordered = ["date", "ticker"] + feature_cols + target_cols
    return out[[c for c in ordered if c in out.columns]]
