from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from app.data.seams import NO_SEAMS, backward_window_invalid

# How many consecutive sessions each indicator consumes. A statistic at row R
# reads rows [R-lookback+1, R], so it spans a data seam -- and is therefore
# invalid -- for `lookback - 1` rows after that seam. Audited per indicator:
#
#   ret_1d            pct_change()          rows R-1..R
#   ret_5d/21d        pct_change(k)         rows R-k..R
#   vol_21d/63d       k returns, each of
#                     which needs one extra prior row
#   sma_20/50         rolling mean of k closes
#   bb_zscore_20      rolling mean+std of 20 closes
#   atr_14            14 true ranges, each using close.shift(1)
#   volume_zscore_21  21 volumes -- volume is ALSO discontinuous at a split
#                     seam, since pre-seam volume is on the pre-split basis
#   obv_delta_5d      5 signed volumes, each needing close.diff()
#   rsi_14 / macd*    EWM: formally infinite memory, so the value below is the
#                     point at which the seam's weight decays under 1%
#                     (alpha=1/14 -> ~62 sessions; span 26 -> ~60; the signal
#                     line adds a span-9 EWM on top, hence the larger figure)
#   day_of_week       single row, spans no transition, never invalidated
FEATURE_LOOKBACK: dict[str, int] = {
    "ret_1d": 2,
    "ret_5d": 6,
    "ret_21d": 22,
    "vol_21d": 22,
    "vol_63d": 64,
    "rsi_14": 64,
    "macd": 62,
    "macd_signal": 82,
    "macd_hist": 82,
    "sma_20": 20,
    "sma_50": 50,
    "sma_ratio_20_50": 50,
    "bb_zscore_20": 20,
    "atr_14": 15,
    "volume_zscore_21": 21,
    "obv_delta_5d": 6,
    "day_of_week": 1,
    # Scale-free replacements. Measured across the 447-ticker universe, the
    # raw forms above vary with absolute price or share count by 700x-2100x
    # between a $5 and a $1,600 stock, which makes them meaningless in a
    # pooled model. These carry the same information as ratios.
    "close_to_sma20": 20,
    "close_to_sma50": 50,
    "macd_norm": 62,
    "macd_signal_norm": 82,
    "macd_hist_norm": 82,
    "atr_pct": 15,
    "obv_delta_norm": 22,
}

# Features whose magnitude depends on absolute price or share-volume scale.
# Retained for backwards compatibility but excluded from the model matrix.
SCALE_DEPENDENT = (
    "sma_20",
    "sma_50",
    "macd",
    "macd_signal",
    "macd_hist",
    "atr_14",
    "obv_delta_5d",
)


def rsi(close: pd.Series, window: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / window, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / window, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    line = ema_fast - ema_slow
    sig = line.ewm(span=signal, adjust=False).mean()
    return pd.DataFrame({"macd": line, "macd_signal": sig, "macd_hist": line - sig})


def bollinger_zscore(close: pd.Series, window: int = 20) -> pd.Series:
    sma = close.rolling(window).mean()
    std = close.rolling(window).std()
    return (close - sma) / std.replace(0, np.nan)


def atr(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    return tr.rolling(window).mean()


def obv_delta(close: pd.Series, volume: pd.Series, window: int = 5) -> pd.Series:
    direction = np.sign(close.diff()).fillna(0)
    obv = (direction * volume).cumsum()
    return obv.diff(window)


def compute_indicators(
    ohlcv: pd.DataFrame,
    seam_dates: Sequence[pd.Timestamp] = NO_SEAMS,
) -> pd.DataFrame:
    """Backward-looking indicators for one ticker.

    `seam_dates` marks artificial discontinuities in the price series. Any
    indicator whose lookback window spans one is set to NaN rather than being
    allowed to consume the jump; prices themselves are never touched. Defaults
    to no seams, so series without a known defect are unaffected.
    """
    close, high, low, volume = ohlcv["close"], ohlcv["high"], ohlcv["low"], ohlcv["volume"]
    ret_1d = close.pct_change()
    feats = pd.DataFrame(index=ohlcv.index)
    feats["ret_1d"] = ret_1d
    feats["ret_5d"] = close.pct_change(5)
    feats["ret_21d"] = close.pct_change(21)
    feats["vol_21d"] = ret_1d.rolling(21).std()
    feats["vol_63d"] = ret_1d.rolling(63).std()
    feats["rsi_14"] = rsi(close)
    m = macd(close)
    feats[["macd", "macd_signal", "macd_hist"]] = m
    feats["sma_20"] = close.rolling(20).mean()
    feats["sma_50"] = close.rolling(50).mean()
    feats["sma_ratio_20_50"] = feats["sma_20"] / feats["sma_50"] - 1
    feats["bb_zscore_20"] = bollinger_zscore(close)
    feats["atr_14"] = atr(high, low, close)
    feats["volume_zscore_21"] = (volume - volume.rolling(21).mean()) / volume.rolling(21).std()
    feats["obv_delta_5d"] = obv_delta(close, volume)

    # --- scale-free counterparts -------------------------------------------
    # Dividing by same-bar close/volume keeps these point-in-time: both are
    # known at the close of day T, like every other feature here.
    safe_close = close.replace(0, np.nan)
    feats["close_to_sma20"] = close / feats["sma_20"].replace(0, np.nan) - 1
    feats["close_to_sma50"] = close / feats["sma_50"].replace(0, np.nan) - 1
    feats["macd_norm"] = feats["macd"] / safe_close
    feats["macd_signal_norm"] = feats["macd_signal"] / safe_close
    feats["macd_hist_norm"] = feats["macd_hist"] / safe_close
    feats["atr_pct"] = feats["atr_14"] / safe_close
    # OBV delta is a 5-day sum of signed share volume; scale it by the volume
    # actually traded over a comparable window to get a dimensionless figure.
    vol_scale = (volume.rolling(21).mean() * 5).replace(0, np.nan)
    feats["obv_delta_norm"] = feats["obv_delta_5d"] / vol_scale

    return mask_seam_contaminated(feats, seam_dates)


def mask_seam_contaminated(
    feats: pd.DataFrame,
    seam_dates: Sequence[pd.Timestamp] = NO_SEAMS,
) -> pd.DataFrame:
    """NaN every indicator value whose lookback window crosses a seam."""
    if not len(seam_dates):
        return feats
    out = feats.copy()
    for col in out.columns:
        lookback = FEATURE_LOOKBACK.get(col)
        if lookback is None or lookback < 2:
            continue
        bad = backward_window_invalid(out.index, lookback, seam_dates)
        if bad.any():
            out.iloc[bad.to_numpy(dtype=bool), out.columns.get_loc(col)] = np.nan
    return out
