from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from app.data.sources.base import REQUIRED_OHLCV

# A bar can be internally consistent (high>=low, positive price and volume)
# and still be corrupt: the FNSPID archive contains isolated bars whose OHLC
# sit ~1/1000 of the surrounding price, e.g. TT 2016-03-02 closes at 0.0343
# between neighbours near 38. Those pass every structural rule yet produce
# six-figure percentage returns.
#
# The test is distance from LOCAL CONTEXT -- a centred rolling median of close
# -- not flatness. Flat OHLC is neither necessary nor sufficient: of the 174
# bars this flags across the 447-ticker panel, only 35 are flat, and thinly
# traded sessions are legitimately flat.
#
# Threshold rationale: calibrated on the real panel, the flag count is flat
# between 5x and 20x (182 -> 170 bars, the same 3 tickers), so the choice sits
# in a wide insensitive band. The flagged bars are 100x-1300x away from local
# context while nothing legitimate approaches 10x. 10x is therefore a
# conservative pick that catches every known corruption and nothing else; it
# flags zero bars in the legacy nine-ticker panel.
LOCAL_CONTEXT_WINDOW = 21
LOCAL_CONTEXT_MIN_PERIODS = 11
MAX_LOCAL_DEVIATION = 10.0


@dataclass
class ValidationReport:
    ticker: str
    n_rows_in: int = 0
    n_null_rows_dropped: int = 0
    n_invalid_rows_dropped: int = 0
    n_duplicates_removed: int = 0
    n_implausible_rows_dropped: int = 0
    issues: list[str] = field(default_factory=list)

    @property
    def n_rows_out(self) -> int:
        return (
            self.n_rows_in
            - self.n_null_rows_dropped
            - self.n_invalid_rows_dropped
            - self.n_duplicates_removed
            - self.n_implausible_rows_dropped
        )

    def summary(self) -> str:
        base = (
            f"{self.ticker}: in={self.n_rows_in} out={self.n_rows_out} "
            f"nulls_dropped={self.n_null_rows_dropped} "
            f"invalid_dropped={self.n_invalid_rows_dropped} "
            f"dupes_removed={self.n_duplicates_removed}"
        )
        if self.issues:
            base += " | warnings: " + "; ".join(self.issues)
        return base


def implausible_vs_local_context(
    close: pd.Series,
    max_deviation: float = MAX_LOCAL_DEVIATION,
    window: int = LOCAL_CONTEXT_WINDOW,
    min_periods: int = LOCAL_CONTEXT_MIN_PERIODS,
) -> pd.Series:
    """Bars whose close sits absurdly far from its local price context.

    Deterministic: a centred rolling median over `window` sessions is the
    reference, and a bar is flagged when close/reference leaves
    [1/max_deviation, max_deviation]. The median (not mean) keeps the
    reference itself robust to the very outliers being detected, and centring
    means a one-session excursion is measured against both sides.

    A sustained move is NOT flagged: when a price genuinely steps down and
    stays there, the centred median straddles the step and the ratio lands
    near 1/2, nowhere near the threshold.

    Returns an all-False mask when there is too little context to judge.
    """
    numeric = pd.to_numeric(close, errors="coerce")
    reference = numeric.rolling(window, center=True, min_periods=min_periods).median()
    with np.errstate(invalid="ignore", divide="ignore"):
        ratio = numeric / reference.where(reference > 0)
    flagged = (ratio > max_deviation) | (ratio < 1.0 / max_deviation)
    return flagged.fillna(False)


def validate_prices(
    df: pd.DataFrame,
    ticker: str,
    min_price: float = 0.01,
    allow_zero_volume: bool = False,
    max_local_deviation: float = MAX_LOCAL_DEVIATION,
) -> tuple[pd.DataFrame, ValidationReport]:
    report = ValidationReport(ticker=ticker)

    if df.empty:
        report.issues.append("source returned no rows")
        return pd.DataFrame(columns=REQUIRED_OHLCV + ["ticker"]), report

    missing_cols = [c for c in REQUIRED_OHLCV if c not in df.columns]
    if missing_cols:
        raise ValueError(f"{ticker}: missing required columns {missing_cols}")

    out = df.loc[:, REQUIRED_OHLCV].copy()
    out.index = pd.to_datetime(out.index).tz_localize(None).normalize()
    out.index.name = "date"
    out = out.sort_index()
    report.n_rows_in = len(out)

    null_mask = out[REQUIRED_OHLCV].isna().any(axis=1)
    report.n_null_rows_dropped = int(null_mask.sum())
    if report.n_null_rows_dropped:
        report.issues.append(f"{report.n_null_rows_dropped} rows with NULL OHLCV values")
    out = out[~null_mask]

    volume_invalid = (out["volume"] <= 0) if not allow_zero_volume else (out["volume"] < 0)
    invalid_mask = (
        (out["close"] < min_price)
        | (out["open"] < min_price)
        | (out["high"] < min_price)
        | (out["low"] < min_price)
        | (out["high"] < out["low"])
        | (out["high"] < out[["open", "close"]].max(axis=1))
        | (out["low"] > out[["open", "close"]].min(axis=1))
        | volume_invalid
    )
    report.n_invalid_rows_dropped = int(invalid_mask.sum())
    if report.n_invalid_rows_dropped:
        bad_dates = out.index[invalid_mask]
        preview = ", ".join(str(d.date()) for d in bad_dates[:5])
        report.issues.append(
            f"{report.n_invalid_rows_dropped} structurally invalid rows (e.g. {preview})"
        )
    out = out[~invalid_mask]

    implausible = implausible_vs_local_context(out["close"], max_local_deviation)
    report.n_implausible_rows_dropped = int(implausible.sum())
    if report.n_implausible_rows_dropped:
        bad_dates = out.index[implausible]
        preview = ", ".join(str(d.date()) for d in bad_dates[:5])
        report.issues.append(
            f"{report.n_implausible_rows_dropped} rows implausibly far from local "
            f"price context (>{max_local_deviation:g}x the 21-session median; "
            f"e.g. {preview})"
        )
    out = out[~implausible]

    before = len(out)
    out = out[~out.index.duplicated(keep="last")]
    report.n_duplicates_removed = before - len(out)
    if report.n_duplicates_removed:
        report.issues.append(f"{report.n_duplicates_removed} duplicate dates removed")

    if not allow_zero_volume and len(out):
        n_zero = int((out["volume"] == 0).sum())
        if n_zero:
            report.issues.append(f"{n_zero} zero-volume rows remain")

    if len(out) >= 2:
        gaps = out.index.to_series().diff().dt.days.dropna()
        max_gap = int(gaps.max())
        if max_gap > 10:
            report.issues.append(f"largest date gap is {max_gap} calendar days")

    result = out.copy()
    result["ticker"] = ticker
    return result.reset_index(), report
