"""Ticker-to-entity attribution windows for news.

A ticker is a slot, not an identity. When a symbol changes hands, headlines
tagged with it before the handover describe a DIFFERENT company, and joining
them to the current entity's prices teaches the model from the wrong firm's
news. Price series are fixed by re-sourcing the right entity (see
data/curated_sp500 provenance); news cannot be, because the archive tags each
headline with whatever ticker was current when it was published.

So each ticker whose ownership changed inside our window gets the first date
from which news under that symbol belongs to the entity now holding it.
Everything earlier is excluded with a reason, never reassigned: we do not know
that the old headlines belong to any ticker still in the panel, and guessing
would be worse than dropping them.

Only tickers verified individually appear here. The default -- absent from this
mapping -- is that all of a ticker's news belongs to it.
"""

from __future__ import annotations

import pandas as pd

# ticker -> (first date news belongs to the CURRENT entity, why)
NEWS_ENTITY_START: dict[str, tuple[pd.Timestamp, str]] = {
    "IR": (
        pd.Timestamp("2020-03-02"),
        "Ingersoll Rand Inc. (formerly Gardner Denver) took the IR ticker "
        "2020-03-02. Earlier IR headlines are Ingersoll-Rand plc, which became "
        "Trane Technologies (TT) -- verified by 0.999985 return correlation "
        "between the pre-2017 FNSPID IR price series and TT. The current entity "
        "traded as GDI until 2020-03-02 and the archive carries no GDI news.",
    ),
    "COR": (
        pd.Timestamp("2023-08-30"),
        "Cencora Inc. took the COR ticker 2023-08-30, having traded as ABC. "
        "Earlier COR headlines are CoreSite Realty Corp (IPO 2010-09-23, "
        "acquired by American Tower Dec 2021) -- verified by 0.367727 return "
        "correlation against Cencora over that span versus 1.000000 either side. "
        "The archive carries no ABC news, so Cencora has no pre-2023 news.",
    ),
    "CARR": (
        pd.Timestamp("2020-04-03"),
        "Carrier Global was distributed from United Technologies 2020-04-03 "
        "(when-issued from 2020-03-19). Its news already starts 2020-04-03, so "
        "this bound excludes nothing; it is recorded to keep the rule explicit.",
    ),
    "DOW": (
        pd.Timestamp("2019-04-01"),
        "Dow Inc. listed 2019-04; earlier DOW news is Dow Chemical (pre-DowDuPont).",
    ),
    "FOXA": (
        pd.Timestamp("2019-03-19"),
        "Fox Corp listed 2019-03; earlier FOXA news is 21st Century Fox.",
    ),
    "FOX": (
        pd.Timestamp("2019-03-19"),
        "Fox Corp listed 2019-03; earlier FOX news is 21st Century Fox.",
    ),
}


def news_entity_start(ticker: str) -> pd.Timestamp | None:
    """First date news under `ticker` belongs to the entity now holding it."""
    entry = NEWS_ENTITY_START.get(str(ticker).strip().upper())
    return entry[0] if entry else None


def news_entity_reason(ticker: str) -> str | None:
    """Why `ticker` has an attribution bound, for exclusion logs."""
    entry = NEWS_ENTITY_START.get(str(ticker).strip().upper())
    return entry[1] if entry else None


def news_is_current_entity(ticker: str, date) -> bool:
    """Does news tagged `ticker` on `date` describe the current entity?

    Tickers with no registered handover always return True.
    """
    start = news_entity_start(ticker)
    return True if start is None else pd.Timestamp(date) >= start
