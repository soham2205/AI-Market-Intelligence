"""GICS sector classification for the S&P 500 universe.

No sector data exists anywhere in this repository today -- configs/
sp500_universe.json (the existing universe snapshot) is a flat ticker list
with no sector field, and nothing else in the project carries one. Per the
explicit instruction not to invent sector assignments, this module fetches
real GICS classifications from the SAME source sp500_universe.json already
cites (Wikipedia's "List of S&P 500 companies" page), rather than guessing
or hand-coding a mapping. Free, no API key, one HTTP request, cached to disk.

IMPORTANT CAVEAT, repeated wherever this mapping is used: Wikipedia's table
reflects CURRENT sector membership only. A company's GICS sector can change
over time (reclassifications, spin-offs, M&A), and this mapping is NOT
point-in-time -- applying it to a 2008 news event uses that company's 2026
sector, not necessarily the one it was classified under in 2008. This is a
real, documented limitation of a $0-cost, no-archive approach, not a hidden
one.
"""

from __future__ import annotations

import io
import json
from pathlib import Path

import pandas as pd

WIKI_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
DEFAULT_PATH = Path("configs/sp500_sectors.json")
USER_AGENT = "Mozilla/5.0 (compatible; ai-market-intelligence-research/1.0)"


def fetch_sector_map() -> dict:
    """Fetch the current GICS sector for every S&P 500 constituent.

    Wikipedia's default anonymous fetch (what pandas.read_html does under
    the hood) returns HTTP 403 without a browser-like User-Agent header --
    this is a real request-blocking issue inspected before writing this
    function, not a hypothetical. Fetching the page with `requests` and a
    declared User-Agent, then parsing the HTML already in hand, avoids it.
    """
    import datetime as dt

    import requests

    resp = requests.get(WIKI_URL, headers={"User-Agent": USER_AGENT}, timeout=30)
    resp.raise_for_status()
    tables = pd.read_html(io.StringIO(resp.text))
    table = tables[0]
    missing = {"Symbol", "GICS Sector"} - set(table.columns)
    if missing:
        raise RuntimeError(f"Wikipedia table layout changed; missing columns {missing}")

    mapping = dict(zip(table["Symbol"], table["GICS Sector"], strict=True))
    return {
        "source": WIKI_URL,
        "snapshot_date": dt.date.today().isoformat(),
        "classification_basis": "CURRENT GICS sector membership, not point-in-time historical",
        "count": len(mapping),
        "sectors": sorted(set(mapping.values())),
        "ticker_sector": mapping,
    }


def ensure_sector_map(path: str | Path = DEFAULT_PATH, refresh: bool = False) -> dict:
    """Load the cached sector map, or fetch and cache it if absent/forced."""
    path = Path(path)
    if path.exists() and not refresh:
        return json.loads(path.read_text())
    data = fetch_sector_map()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=1))
    return data


def sector_series(tickers, path: str | Path = DEFAULT_PATH) -> pd.Series:
    """Map an iterable of tickers to their current GICS sector (NaN if unknown)."""
    data = ensure_sector_map(path)
    lookup = data["ticker_sector"]
    return pd.Series([lookup.get(t) for t in tickers], index=pd.Index(tickers, name="ticker"))
