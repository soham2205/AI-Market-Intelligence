"""Known data seams: boundaries across which a return is not real.

A seam is an artificial discontinuity in a price series -- the archive
switching adjustment vintage between two adjacent sessions. The prices on both
sides are individually fine; it is the TRANSITION between them that is
meaningless.

The FNSPID archive stitches vintages at 2020-07-06. Validated against an
independent reference: the FNSPID/reference close ratio is piecewise constant
with exactly one step at that date, and daily returns agree to 4dp on every
other session. The step size tracks dividend yield (XOM 1.19x, KO 1.11x,
JNJ/JPM 1.10x, MSFT 1.03x, non-payers ~1.01x), so it is a dividend-adjustment
vintage difference and it touches essentially every ticker -- not only the 661
whose splits pushed it past the repair threshold.

Design
------
Nothing here modifies a price. Rows are preserved exactly as ingested; the
seam is represented as a boundary, and any statistic whose window spans that
boundary is marked invalid (NaN) instead. That keeps the artefact out of
features without inventing data.

Besides the archive-wide stitch, a handful of individual FNSPID series are
stitched at their own dates; those live in SEAMS_BY_SOURCE_TICKER and are
verified one at a time, never assumed.

Seams are passed in explicitly, so price series that have no seam (a live
yfinance pull, for instance) are entirely unaffected: the default is no seams.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

# The transition INTO this session is artificial: the previous session belongs
# to a different adjustment vintage.
FNSPID_SEAM_DATES: tuple[pd.Timestamp, ...] = (pd.Timestamp("2020-07-06"),)

NO_SEAMS: tuple[pd.Timestamp, ...] = ()

# Column carrying data provenance through curation and into feature building.
SOURCE_COLUMN = "source"

# THE single place a seam date is associated with a data source. Feature
# construction looks seams up from the provenance recorded on each row, so a
# caller cannot forget to pass them, and a source with no known defect (a live
# yfinance pull) resolves to no seams at all.
SEAMS_BY_SOURCE: dict[str, tuple[pd.Timestamp, ...]] = {
    "fnspid": FNSPID_SEAM_DATES,
    "yfinance": NO_SEAMS,
}

# Per-ticker vintage boundaries, keyed (source, ticker).
#
# The 2020-07-06 stitch above is archive-wide. These are NOT: each one is a
# boundary in how a single series was assembled, so registering it source-wide
# would mask thousands of good tickers. Each entry below was verified
# individually with the same test that established the archive-wide seam --
# the FNSPID/yfinance close ratio must be piecewise constant with exactly ONE
# step at the candidate date, and daily returns must agree everywhere else:
#
#   ticker  date        ratio step   post-step ratio cv   other >2% return
#                                                         disagreements
#   NEE     2020-04-02  0.2335x      0.000000             none
#   NDAQ    2020-04-02  0.3160x      0.000000             none
#   EW      2020-04-02  0.3333x      0.000000             none
#   ETN     2019-08-15  1.5715x      0.030065             2020-07-06 only
#   MO      2008-03-31  0.2118x      0.182353             3 minor (<0.03)
#
# Deliberately NOT registered:
#   DHR 2016-07-05 -- ratio step 0.9501x, i.e. no step. Both sources show the
#     ~1.6x move: it is the real Fortive spin-off, which adjusted series do not
#     remove. Masking it would delete a genuine market event.
#   GEN 2010-07-07 -- fails the test outright (1,584 sessions disagree by >2%,
#     ratio cv 1.09 after the candidate). That series is pervasively mismatched,
#     not stitched at one boundary, so a seam is the wrong instrument; GEN is
#     sourced from yfinance instead.
SEAMS_BY_SOURCE_TICKER: dict[tuple[str, str], tuple[pd.Timestamp, ...]] = {
    ("fnspid", "NEE"): (pd.Timestamp("2020-04-02"),),
    ("fnspid", "NDAQ"): (pd.Timestamp("2020-04-02"),),
    ("fnspid", "EW"): (pd.Timestamp("2020-04-02"),),
    ("fnspid", "ETN"): (pd.Timestamp("2019-08-15"),),
    ("fnspid", "MO"): (pd.Timestamp("2008-03-31"),),
}


def seams_for_source(
    source: str | None, ticker: str | None = None
) -> tuple[pd.Timestamp, ...]:
    """Seams for a named source, plus any registered for (source, ticker).

    Defaulting to "no seams" is deliberate: masking a series that has no
    defect would silently destroy good data, which is worse than the
    conservative failure of leaving an unrecognised source untouched.

    Passing no ticker returns only the source-wide seams, so a caller that
    does not know the ticker cannot accidentally pick up another one's.
    """
    if source is None:
        return NO_SEAMS
    key = str(source).strip().lower()
    seams = SEAMS_BY_SOURCE.get(key, NO_SEAMS)
    if ticker is not None:
        extra = SEAMS_BY_SOURCE_TICKER.get((key, str(ticker).strip().upper()), NO_SEAMS)
        if extra:
            seams = tuple(sorted(set(seams) | set(extra)))
    return seams


def seams_for_frame(df: pd.DataFrame) -> tuple[pd.Timestamp, ...]:
    """Seams implied by the provenance recorded in a price frame.

    Unions the seams of every source present, so a frame stitched from more
    than one provider is handled conservatively.
    """
    if SOURCE_COLUMN not in df.columns:
        return NO_SEAMS
    tickers: list[str | None] = [None]
    if "ticker" in df.columns:
        present = pd.unique(df["ticker"].dropna())
        if len(present):
            tickers = list(present)
    seen: list[pd.Timestamp] = []
    for source in pd.unique(df[SOURCE_COLUMN].dropna()):
        for ticker in tickers:
            for seam in seams_for_source(source, ticker):
                if seam not in seen:
                    seen.append(seam)
    return tuple(sorted(seen))


def sessions_since_seam(
    index: pd.DatetimeIndex,
    seam_dates: Sequence[pd.Timestamp] = NO_SEAMS,
) -> np.ndarray:
    """Sessions elapsed since the most recent seam, per row.

    The seam row itself is 0, the next session 1, and so on. Rows before every
    seam get a large sentinel: they sit wholly inside one vintage, so nothing
    about them is contaminated by a later stitch.

    Positional, not calendar-based -- feature windows count sessions, and
    holidays would otherwise shift the arithmetic.
    """
    n = len(index)
    out = np.full(n, np.iinfo(np.int32).max, dtype=np.int64)
    if n == 0 or not len(seam_dates):
        return out

    for seam in seam_dates:
        pos = int(np.searchsorted(index.to_numpy(), np.datetime64(pd.Timestamp(seam))))
        if pos >= n:
            continue  # seam after this series ends
        if pos == 0:
            continue  # no row before the seam, so no bad transition inside it
        distance = np.arange(n, dtype=np.int64) - pos
        here = np.where(distance >= 0, distance, np.iinfo(np.int32).max)
        out = np.minimum(out, here)
    return out


def backward_window_invalid(
    index: pd.DatetimeIndex,
    lookback: int,
    seam_dates: Sequence[pd.Timestamp] = NO_SEAMS,
) -> np.ndarray:
    """Rows whose backward window of `lookback` sessions spans a seam.

    A statistic at row R consuming rows [R-lookback+1, R] covers the
    transitions between each adjacent pair inside that window. It therefore
    contains the bad transition exactly while

        0 <= (R - seam_row) <= lookback - 2

    so it is invalid for ``lookback - 1`` rows starting at the seam. A
    single-row feature (lookback 1, e.g. day_of_week) spans no transition and
    is never invalidated.
    """
    if lookback < 2 or not len(seam_dates):
        return np.zeros(len(index), dtype=bool)
    since = sessions_since_seam(index, seam_dates)
    return since <= (lookback - 2)


def forward_window_invalid(
    index: pd.DatetimeIndex,
    horizon: int,
    seam_dates: Sequence[pd.Timestamp] = NO_SEAMS,
) -> np.ndarray:
    """Rows whose FORWARD return over `horizon` sessions spans a seam.

    A label at row T reads close[T+horizon]/close[T], covering transitions
    T->T+1 ... T+horizon-1->T+horizon. That includes the bad transition while
    the seam row sits in [T+1, T+horizon], i.e. for the `horizon` rows
    immediately BEFORE the seam.
    """
    n = len(index)
    invalid = np.zeros(n, dtype=bool)
    if horizon < 1 or not len(seam_dates):
        return invalid
    for seam in seam_dates:
        pos = int(np.searchsorted(index.to_numpy(), np.datetime64(pd.Timestamp(seam))))
        if pos <= 0 or pos > n:
            continue
        lo = max(0, pos - horizon)
        invalid[lo:pos] = True
    return invalid
