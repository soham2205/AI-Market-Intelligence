"""Alpha Vantage NEWS_SENTIMENT adapter -- current/recent news retrieval.

Implements the existing `app.data.sources.base.NewsSource` protocol and returns
the canonical NEWS_SCHEMA, so it drops into the same pipeline as
YFinanceNewsSource and FnspidNewsSource with no downstream changes.

Why this source exists alongside the other two: yfinance returns roughly the
last two days (fine for a spot check, too short to build a current-news view),
and FNSPID is a frozen local archive ending 2024-01-08. Alpha Vantage's
NEWS_SENTIMENT endpoint serves recent articles with real publication
timestamps, which is what the current-news path needs.

CREDENTIALS: read from the ALPHAVANTAGE_API_KEY environment variable and never
written to disk, logged, printed, or echoed into any output artifact. The key
is not accepted as a CLI argument on purpose -- that would put it in shell
history and in process listings. `redacted_key_status()` exists so a run can
report that a key was found without revealing any part of it. Nothing in this
module interpolates the key into a message; the only place it appears is the
outbound request's query string.

TIMESTAMPS: Alpha Vantage returns `time_published` as "YYYYMMDDTHHMMSS" in US
Eastern. It is parsed with that explicit format and localised to US/Eastern
before being converted to UTC, because the rest of the pipeline assumes
tz-aware UTC. An article whose timestamp will not parse is DROPPED, never
back-filled with a fetch-time default: the base protocol spells out why -- a
synthesised publication time is a look-ahead bug waiting to happen, and this
feed's output flows into return-window logic keyed on event date.

RATE LIMITS: the free tier allows a small number of requests per minute, so
`fetch_many` paces requests and surfaces throttle responses as a clear error.
Alpha Vantage signals throttling with HTTP 200 plus a JSON body containing
"Note" or "Information" rather than an error status, so that case is detected
explicitly instead of being mistaken for an empty result -- an empty article
list and a refused request mean very different things, and conflating them
would silently look like "no current news".
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

import pandas as pd

from app.data.sources.base import NEWS_SCHEMA

API_URL = "https://www.alphavantage.co/query"
API_KEY_ENV = "ALPHAVANTAGE_API_KEY"
DEFAULT_TIMEOUT = 30
DEFAULT_SLEEP_SECONDS = 12.0  # free tier is ~5 requests/minute
AV_TZ = "US/Eastern"
MAX_LIMIT = 1000


class AlphaVantageConfigError(RuntimeError):
    """Raised when no API key is available to the running application."""


class AlphaVantageRateLimit(RuntimeError):
    """Raised when Alpha Vantage refuses or throttles the request."""


class AlphaVantagePremiumRequired(AlphaVantageRateLimit):
    """Raised when the account's plan does not include this endpoint.

    Kept distinct from a rate limit because the remedy is different and the
    distinction is load-bearing: a throttle clears on its own and retrying
    later works, whereas a premium gate will refuse every request on this plan
    forever. Reporting the second as the first would send someone into a retry
    loop that cannot ever succeed. Confirmed against the live API: a valid free
    key on NEWS_SENTIMENT returns HTTP 200 with an "Information" advisory
    reading "This is a premium endpoint."
    """


def api_key_available(env: dict | None = None) -> bool:
    env = os.environ if env is None else env
    return bool((env.get(API_KEY_ENV) or "").strip())


def redacted_key_status(env: dict | None = None) -> str:
    """Describe credential availability without revealing the key.

    Deliberately reports only presence and length -- no prefix, suffix, or
    masked body, since even a few characters narrow a brute-force search.
    """
    env = os.environ if env is None else env
    key = (env.get(API_KEY_ENV) or "").strip()
    if not key:
        return f"{API_KEY_ENV}: NOT SET"
    return f"{API_KEY_ENV}: set (length {len(key)}, value never logged)"


def _require_key(env: dict | None = None) -> str:
    env = os.environ if env is None else env
    key = (env.get(API_KEY_ENV) or "").strip()
    if not key:
        raise AlphaVantageConfigError(
            f"No Alpha Vantage API key found. Set {API_KEY_ENV} in your environment "
            "(or in a .env file, which this repository already git-ignores) and re-run. "
            "The key is read from the environment only -- never pass it as a CLI "
            "argument, which would record it in shell history."
        )
    return key


class AlphaVantageNewsSource:
    """NewsSource over Alpha Vantage NEWS_SENTIMENT (current/recent news)."""

    name = "alphavantage"

    def __init__(
        self,
        env: dict | None = None,
        timeout: int = DEFAULT_TIMEOUT,
        sleep_seconds: float = DEFAULT_SLEEP_SECONDS,
        limit: int = 200,
        opener=None,
    ):
        self._env = env
        self.timeout = timeout
        self.sleep_seconds = sleep_seconds
        self.limit = min(limit, MAX_LIMIT)
        # Injectable transport so tests exercise the real parsing path against
        # recorded payloads without any network access.
        self._opener = opener or self._urlopen

    # -- transport ---------------------------------------------------------

    def _urlopen(self, url: str) -> str:
        with urllib.request.urlopen(url, timeout=self.timeout) as resp:  # noqa: S310
            return resp.read().decode("utf-8")

    def available(self) -> bool:
        """Mirrors FnspidNewsSource.available(): is this source usable at all?"""
        return api_key_available(self._env)

    def _request(self, params: dict) -> dict:
        key = _require_key(self._env)
        query = {**params, "apikey": key}
        url = f"{API_URL}?{urllib.parse.urlencode(query)}"
        try:
            body = self._opener(url)
        except urllib.error.HTTPError as exc:  # pragma: no cover - network path
            raise AlphaVantageRateLimit(
                f"Alpha Vantage HTTP {exc.code} for ticker request"
            ) from exc
        except urllib.error.URLError as exc:  # pragma: no cover - network path
            raise AlphaVantageRateLimit(
                f"Alpha Vantage unreachable: {exc.reason}"
            ) from exc
        try:
            payload = json.loads(body)
        except json.JSONDecodeError as exc:
            raise AlphaVantageRateLimit(
                "Alpha Vantage returned a non-JSON body"
            ) from exc
        # Throttle/quota arrives as HTTP 200 with an advisory key. Treat it as
        # an error: silently returning zero articles would be indistinguishable
        # from "this ticker genuinely has no current news".
        for advisory in ("Note", "Information", "Error Message"):
            if advisory in payload:
                message = str(payload[advisory])
                if "premium" in message.lower():
                    raise AlphaVantagePremiumRequired(
                        "Alpha Vantage reports NEWS_SENTIMENT as a premium endpoint for "
                        "this account's plan. The API key itself authenticated fine -- "
                        "this is a plan entitlement, not a credential problem, so "
                        "retrying will not help. "
                        f"Provider message: {message}"
                    )
                raise AlphaVantageRateLimit(
                    f"Alpha Vantage refused the request ({advisory}): {message}"
                )
        return payload

    # -- parsing -----------------------------------------------------------

    @staticmethod
    def _parse_timestamp(raw: str | None) -> pd.Timestamp | None:
        """Parse 'YYYYMMDDTHHMMSS' US/Eastern -> tz-aware UTC, or None."""
        if not raw:
            return None
        ts = pd.to_datetime(raw, format="%Y%m%dT%H%M%S", errors="coerce")
        if pd.isna(ts):
            ts = pd.to_datetime(raw, errors="coerce", utc=True)
            if pd.isna(ts):
                return None
            return ts
        return ts.tz_localize(AV_TZ, ambiguous=True, nonexistent="shift_forward").tz_convert(
            "UTC"
        )

    def _rows_for_ticker(self, payload: dict, ticker: str) -> list[dict]:
        ticker = ticker.strip().upper()
        rows = []
        for item in payload.get("feed") or []:
            title = (item.get("title") or "").strip()
            published = self._parse_timestamp(item.get("time_published"))
            if not title or published is None:
                # No headline means nothing to score or classify; no parseable
                # timestamp means the event date is unknown, and guessing it
                # would corrupt every forward-return alignment downstream.
                continue
            url = (item.get("url") or "").strip()
            rows.append({
                "article_id": _article_id(url, title),
                "headline": title,
                "source": (item.get("source") or "alphavantage").strip(),
                "url": url,
                "published_at": published,
                "ticker": ticker,
                # Alpha Vantage ships its own sentiment. It is carried through
                # under av_-prefixed names for reference but is NOT used as the
                # pipeline's sentiment: FinBERT remains the scorer of record so
                # current and historical articles are scored by one model.
                "av_overall_sentiment_score": _as_float(
                    item.get("overall_sentiment_score")
                ),
                "av_overall_sentiment_label": item.get("overall_sentiment_label"),
                "av_ticker_relevance": _ticker_relevance(item, ticker),
            })
        return rows

    # -- public API --------------------------------------------------------

    def fetch(
        self,
        ticker: str,
        start: str | None = None,
        end: str | None = None,
    ) -> pd.DataFrame:
        """Return NEWS_SCHEMA (+ av_* extras) rows for one ticker.

        `start`/`end` are passed to the API as time_from/time_to AND re-applied
        locally, so the window is enforced on our side regardless of what the
        feed decides to return.
        """
        params = {
            "function": "NEWS_SENTIMENT",
            "tickers": ticker.strip().upper(),
            "limit": str(self.limit),
            "sort": "LATEST",
        }
        if start is not None:
            params["time_from"] = _to_av_time(start)
        if end is not None:
            params["time_to"] = _to_av_time(end)

        payload = self._request(params)
        rows = self._rows_for_ticker(payload, ticker)
        extra = ["av_overall_sentiment_score", "av_overall_sentiment_label",
                 "av_ticker_relevance"]
        df = pd.DataFrame(rows, columns=[*NEWS_SCHEMA, *extra])
        if len(df):
            ts = pd.to_datetime(df["published_at"], utc=True, errors="coerce")
            keep = ts.notna()
            if start is not None:
                keep &= ts >= pd.Timestamp(start, tz="UTC")
            if end is not None:
                keep &= ts <= pd.Timestamp(end, tz="UTC")
            df = df[keep]
            df = df.sort_values("published_at").reset_index(drop=True)
        return df

    def fetch_many(
        self,
        tickers,
        start: str | None = None,
        end: str | None = None,
        on_message=None,
    ) -> tuple[pd.DataFrame, list[dict]]:
        """Fetch several tickers, pacing requests for the free-tier rate limit.

        Returns (articles, per_ticker_status). A throttle or failure on one
        ticker is recorded in the status list and does not abort the run, but it
        is reported as `rate_limited`/`error` -- never as zero articles, so a
        refused request can never be read as an absence of news.
        """
        frames, status = [], []
        tickers = [t.strip().upper() for t in tickers]
        for i, ticker in enumerate(tickers):
            if i and self.sleep_seconds:
                time.sleep(self.sleep_seconds)
            try:
                df = self.fetch(ticker, start, end)
            except AlphaVantagePremiumRequired as exc:
                status.append({"ticker": ticker, "status": "premium_required",
                               "n_articles": 0, "detail": str(exc)})
                if on_message:
                    on_message(f"[ERROR] {ticker}: {exc}")
                continue
            except AlphaVantageRateLimit as exc:
                status.append({"ticker": ticker, "status": "rate_limited",
                               "n_articles": 0, "detail": str(exc)})
                if on_message:
                    on_message(f"[WARN] {ticker}: {exc}")
                continue
            except Exception as exc:  # noqa: BLE001 - one bad ticker must not kill the run
                status.append({"ticker": ticker, "status": "error",
                               "n_articles": 0, "detail": str(exc)})
                if on_message:
                    on_message(f"[WARN] {ticker}: {type(exc).__name__}: {exc}")
                continue
            status.append({"ticker": ticker, "status": "ok", "n_articles": int(len(df)),
                           "detail": ""})
            if on_message:
                on_message(f"[OK] {ticker}: {len(df)} current articles")
            if len(df):
                frames.append(df)
        articles = (
            pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=NEWS_SCHEMA)
        )
        return articles, status


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _article_id(url: str, title: str) -> str:
    """Same construction YFinanceNewsSource uses, so dedupe_news() works
    identically across sources and an article seen twice collapses."""
    import hashlib

    return hashlib.sha1(f"{url}|{title}".encode()).hexdigest()[:16]


def _as_float(value) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _ticker_relevance(item: dict, ticker: str) -> float | None:
    for ts in item.get("ticker_sentiment") or []:
        if (ts.get("ticker") or "").upper() == ticker:
            return _as_float(ts.get("relevance_score"))
    return None


def _to_av_time(value: str) -> str:
    """Our ISO-ish date -> Alpha Vantage's YYYYMMDDTHHMM."""
    ts = pd.Timestamp(value)
    return ts.strftime("%Y%m%dT%H%M")


def load_dotenv_key(path: str | Path = ".env", env: dict | None = None) -> bool:
    """Populate ALPHAVANTAGE_API_KEY from a .env file if it is not already set.

    Does NOT overwrite an existing environment value -- an already-exported key
    always wins, so running this can never silently change which credential is
    in use. Only the Alpha Vantage key is read; other entries are ignored.
    Returns True if the key is available afterwards.
    """
    env = os.environ if env is None else env
    if api_key_available(env):
        return True
    p = Path(path)
    if not p.exists():
        return False
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        name, _, value = line.partition("=")
        if name.strip() == API_KEY_ENV:
            value = value.strip().strip('"').strip("'")
            if value:
                env[API_KEY_ENV] = value
            break
    return api_key_available(env)
