from __future__ import annotations

import datetime as dt

import pandas as pd

ET = "America/New_York"
SESSION_OPEN = dt.time(9, 30)
SESSION_CLOSE = dt.time(16, 0)


def _roll_to_session(candidate: pd.Timestamp, sessions: pd.DatetimeIndex) -> pd.Timestamp:
    """First real trading session on or after `candidate`.

    Returns NaT when `candidate` is past the end of the known calendar — the
    session it would map to has not happened yet, so there is nothing to
    attribute the article to.
    """
    pos = sessions.searchsorted(candidate, side="left")
    if pos >= len(sessions):
        return pd.NaT
    return sessions[pos]


def first_tradable_session(
    published_at: pd.Timestamp,
    sessions: pd.DatetimeIndex | None = None,
) -> pd.Timestamp:
    """Map an article's UTC publish time to its first tradable session date.

    Rules (conservative by design — staleness costs less than leakage):
      - published before 09:30 ET  -> that day, if it is a session
      - published during or after the session -> the next session
      - weekends and market holidays roll forward to the next real session

    `sessions` is the actual trading calendar (e.g. the distinct dates in the
    curated price panel). Supply it whenever it is available: pandas' BDay is
    calendar-naive and happily lands on July 4th or December 25th, which are
    not sessions. When `sessions` is None the original BDay behaviour is kept
    so existing callers do not break.
    """
    ts = pd.Timestamp(published_at)
    if ts.tz is None:
        ts = ts.tz_localize("UTC")
    local = ts.tz_convert(ET)
    day = local.normalize().tz_localize(None)

    if sessions is not None:
        sessions = pd.DatetimeIndex(sessions).normalize().sort_values()
        # At or after the open the information is only actionable next session.
        candidate = day + pd.Timedelta(days=1) if local.time() >= SESSION_OPEN else day
        return _roll_to_session(candidate, sessions)

    if local.time() >= SESSION_OPEN:
        return pd.Timestamp((day + pd.tseries.offsets.BDay(1)).normalize())
    if day.weekday() >= 5:
        return pd.Timestamp((day + pd.tseries.offsets.BDay(1)).normalize())
    return day
