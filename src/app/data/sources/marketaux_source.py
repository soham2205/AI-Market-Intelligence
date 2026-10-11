"""Marketaux news adapter -- a third provider behind the existing NewsSource protocol.

Added because Alpha Vantage's NEWS_SENTIMENT turned out to be a premium-only
endpoint on the free tier (verified against the live API: HTTP 200 with an
"Information" advisory reading "This is a premium endpoint"), and yfinance's
news endpoint currently returns zero items. Both of those adapters are kept
and unchanged; this is an addition, not a replacement, and the pipeline selects
between them with `--source`.

OFFICIAL FREE-TIER LIMITS, verified at implementation time (2026-10-11) from
marketaux.com/pricing and marketaux.com/documentation:

    Free plan:  100 requests / day,  3 ARTICLES PER NEWS REQUEST
    Basic 20 / Standard 50 / Pro 100 articles per request

THE THREE-ARTICLE CAP IS THE DOMINANT CONSTRAINT and the reason this module
reports provider metadata so loudly. `limit` is accepted by the API but
"the maximum limit is based on your plan", so a free-tier request for 50
articles silently comes back with 3. The response's `meta` object carries
`found` (total matching the query), `returned` (how many actually came back)
and `limit` (what the plan allowed), so the adapter can state exactly how much
was withheld instead of presenting 3 articles as if they were the whole story.
`fetch_with_meta` surfaces that, and `truncated` / `withheld` are computed from
it. Three articles out of four hundred is a sample, not a news feed, and
anything built on it has to say so.

CREDENTIALS: read from MARKETAUX_API_TOKEN only. This is deliberately a
SEPARATE variable from ALPHAVANTAGE_API_KEY -- neither is read by the other
adapter, neither is overwritten, and neither is logged, printed, or written to
any artifact. As with the Alpha Vantage adapter, the token is not accepted as a
CLI argument, because that would place it in shell history and process
listings. `redacted_token_status()` reports presence and length only.

SENTIMENT: Marketaux ships its own `sentiment_score` per entity. It is carried
through under `mx_`-prefixed columns for reference and is NEVER used as the
pipeline's sentiment. FinBERT remains the scorer of record so that current and
historical articles are scored by one model; swapping in a provider's score for
some articles and not others would make every cross-era comparison meaningless.

TIMESTAMPS: `published_at` arrives as ISO 8601 with microseconds and explicit
UTC ("2024-11-08T01:24:00.000000Z"). Parsed as UTC; an article whose timestamp
will not parse is DROPPED rather than back-filled, per the NewsSource contract
-- a synthesised publication time is a look-ahead bug waiting to happen.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

import pandas as pd

from app.data.sources.base import NEWS_SCHEMA

API_URL = "https://api.marketaux.com/v1/news/all"
API_TOKEN_ENV = "MARKETAUX_API_TOKEN"
DEFAULT_TIMEOUT = 30
DEFAULT_SLEEP_SECONDS = 1.0  # free tier limits per-60s bursts, not per-request
DEFAULT_LIMIT = 3

# Verified 2026-10-11 from https://www.marketaux.com/pricing
FREE_PLAN_ARTICLE_LIMIT = 3
FREE_PLAN_DAILY_REQUESTS = 100

# Extra columns this adapter adds alongside NEWS_SCHEMA. mx_* values are
# reference-only and never feed the pipeline's sentiment.
EXTRA_COLUMNS = (
    "description",
    "snippet",
    "mx_uuid",
    "mx_entity_symbol",
    "mx_entity_match_score",
    "mx_entity_sentiment_score",
    "mx_relevance_score",
)


class MarketauxError(RuntimeError):
    """Base for every Marketaux failure, so callers can catch broadly."""


class MarketauxConfigError(MarketauxError):
    """No API token available to the running application."""


class MarketauxAuthError(MarketauxError):
    """invalid_api_token (401) -- the token itself was rejected."""


class MarketauxUsageLimit(MarketauxError):
    """usage_limit_reached (402) -- the plan's daily quota is exhausted.

    Distinct from MarketauxRateLimit because the remedy differs: the daily
    quota resets on a daily boundary, whereas a 429 burst limit clears within a
    minute. Collapsing them would make "wait a minute" advice wrong for the one
    case where it cannot work.
    """


class MarketauxRateLimit(MarketauxError):
    """rate_limit_reached (429) -- too many requests in the past 60 seconds."""


class MarketauxPlanRestricted(MarketauxError):
    """endpoint_access_restricted (403) -- not available on this plan."""


class MarketauxProviderError(MarketauxError):
    """Any other provider-side failure (400/404/500/503, unparseable body)."""


_ERROR_CODE_MAP = {
    "invalid_api_token": MarketauxAuthError,
    "usage_limit_reached": MarketauxUsageLimit,
    "rate_limit_reached": MarketauxRateLimit,
    "endpoint_access_restricted": MarketauxPlanRestricted,
}


def api_token_available(env: dict | None = None) -> bool:
    env = os.environ if env is None else env
    return bool((env.get(API_TOKEN_ENV) or "").strip())


def redacted_token_status(env: dict | None = None) -> str:
    """Presence and length only -- never a prefix, suffix, or masked body,
    since even a few characters narrow a brute-force search."""
    env = os.environ if env is None else env
    token = (env.get(API_TOKEN_ENV) or "").strip()
    if not token:
        return f"{API_TOKEN_ENV}: NOT SET"
    return f"{API_TOKEN_ENV}: set (length {len(token)}, value never logged)"


def load_dotenv_token(path: str | Path = ".env", env: dict | None = None) -> bool:
    """Populate MARKETAUX_API_TOKEN from a .env file if it is not already set.

    Mirrors the Alpha Vantage loader deliberately rather than sharing code with
    it: each adapter reads ONLY its own variable, so neither can read or
    clobber the other's credential. An already-exported value always wins, so
    calling this can never silently change which token is in use.
    """
    env = os.environ if env is None else env
    if api_token_available(env):
        return True
    p = Path(path)
    if not p.exists():
        return False
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        name, _, value = line.partition("=")
        if name.strip() == API_TOKEN_ENV:
            value = value.strip().strip('"').strip("'")
            if value:
                env[API_TOKEN_ENV] = value
            break
    return api_token_available(env)


def _require_token(env: dict | None = None) -> str:
    env = os.environ if env is None else env
    token = (env.get(API_TOKEN_ENV) or "").strip()
    if not token:
        raise MarketauxConfigError(
            f"No Marketaux API token found. Set {API_TOKEN_ENV} in your environment "
            "(or in a .env file at the repository root, which is already git-ignored) "
            "and re-run:\n"
            f"  bash:        export {API_TOKEN_ENV}=<your-token>\n"
            f"  PowerShell:  $env:{API_TOKEN_ENV} = '<your-token>'\n"
            f"This is a separate variable from ALPHAVANTAGE_API_KEY; neither affects "
            "the other. The token is read from the environment only and is never logged."
        )
    return token


class MarketauxNewsSource:
    """NewsSource over Marketaux /v1/news/all (current/recent news)."""

    name = "marketaux"

    def __init__(
        self,
        env: dict | None = None,
        timeout: int = DEFAULT_TIMEOUT,
        sleep_seconds: float = DEFAULT_SLEEP_SECONDS,
        limit: int = DEFAULT_LIMIT,
        language: str | None = "en",
        filter_entities: bool = True,
        opener=None,
    ):
        self._env = env
        self.timeout = timeout
        self.sleep_seconds = sleep_seconds
        self.limit = max(1, int(limit))
        self.language = language
        self.filter_entities = filter_entities
        # Injectable transport so tests exercise the real parsing path with
        # recorded payloads and no network access.
        self._opener = opener or self._urlopen
        self.last_meta: dict = {}

    # -- transport ---------------------------------------------------------

    def _urlopen(self, url: str) -> str:
        with urllib.request.urlopen(url, timeout=self.timeout) as resp:  # noqa: S310
            return resp.read().decode("utf-8")

    def available(self) -> bool:
        """Mirrors the other adapters' available(): is this source usable?"""
        return api_token_available(self._env)

    def _request(self, params: dict) -> dict:
        token = _require_token(self._env)
        url = f"{API_URL}?{urllib.parse.urlencode({**params, 'api_token': token})}"
        body: str | None = None
        try:
            body = self._opener(url)
        except urllib.error.HTTPError as exc:
            # Marketaux returns its documented error envelope in the BODY of a
            # 4xx/5xx, so the body is read before falling back to the status.
            try:
                body = exc.read().decode("utf-8")
            except Exception:  # noqa: BLE001 - body may be absent entirely
                raise MarketauxProviderError(
                    f"Marketaux HTTP {exc.code} with no readable body"
                ) from exc
        except urllib.error.URLError as exc:
            raise MarketauxProviderError(
                f"Marketaux unreachable: {exc.reason}"
            ) from exc

        try:
            payload = json.loads(body)
        except json.JSONDecodeError as exc:
            raise MarketauxProviderError(
                "Marketaux returned a non-JSON body"
            ) from exc

        if isinstance(payload, dict) and "error" in payload:
            err = payload["error"] or {}
            code = str(err.get("code") or "unknown")
            message = str(err.get("message") or "")
            raise _ERROR_CODE_MAP.get(code, MarketauxProviderError)(
                f"Marketaux error [{code}]: {message}"
            )
        return payload

    # -- parsing -----------------------------------------------------------

    @staticmethod
    def _parse_timestamp(raw) -> pd.Timestamp | None:
        if not raw:
            return None
        ts = pd.to_datetime(raw, utc=True, errors="coerce")
        return None if pd.isna(ts) else ts

    @staticmethod
    def _matched_entity(item: dict, ticker: str) -> dict:
        """The entity block for the requested ticker, if the article carries it.

        Articles routinely mention several companies, so attribution uses the
        symbol we asked for rather than whichever entity happens to be first.
        """
        for ent in item.get("entities") or []:
            if (ent.get("symbol") or "").upper() == ticker:
                return ent
        return {}

    def _rows(self, payload: dict, ticker: str) -> list[dict]:
        ticker = ticker.strip().upper()
        rows = []
        for item in payload.get("data") or []:
            title = (item.get("title") or "").strip()
            published = self._parse_timestamp(item.get("published_at"))
            if not title or published is None:
                # No headline means nothing to score or classify; no parseable
                # timestamp means the event date is unknown, and guessing it
                # would corrupt the forward-return alignment downstream.
                continue
            url = (item.get("url") or "").strip()
            ent = self._matched_entity(item, ticker)
            rows.append({
                # Same construction as the yfinance and Alpha Vantage adapters,
                # so dedupe_news() collapses the same article across providers.
                "article_id": hashlib.sha1(f"{url}|{title}".encode()).hexdigest()[:16],
                "headline": title,
                "source": (item.get("source") or "marketaux").strip(),
                "url": url,
                "published_at": published,
                "ticker": ticker,
                "description": (item.get("description") or "").strip(),
                "snippet": (item.get("snippet") or "").strip(),
                "mx_uuid": item.get("uuid"),
                "mx_entity_symbol": ent.get("symbol"),
                "mx_entity_match_score": _as_float(ent.get("match_score")),
                # Reference only. NEVER used as the pipeline's sentiment.
                "mx_entity_sentiment_score": _as_float(ent.get("sentiment_score")),
                "mx_relevance_score": _as_float(item.get("relevance_score")),
            })
        return rows

    # -- public API --------------------------------------------------------

    def fetch_with_meta(
        self, ticker: str, start: str | None = None, end: str | None = None
    ) -> tuple[pd.DataFrame, dict]:
        """Articles plus the provider metadata describing what was withheld.

        The metadata is the point: on the free plan `found` is routinely in the
        hundreds while `returned` is 3, and a caller that sees only the frame
        would have no way to know it holds a 3-article sample of a much larger
        result set.
        """
        params: dict[str, str] = {
            "symbols": ticker.strip().upper(),
            "limit": str(self.limit),
        }
        if self.language:
            params["language"] = self.language
        if self.filter_entities:
            params["filter_entities"] = "true"
        if start is not None:
            params["published_after"] = _to_mx_time(start)
        if end is not None:
            params["published_before"] = _to_mx_time(end)

        payload = self._request(params)
        raw_meta = payload.get("meta") or {}
        rows = self._rows(payload, ticker)

        df = pd.DataFrame(rows, columns=[*NEWS_SCHEMA, *EXTRA_COLUMNS])
        if len(df):
            ts = pd.to_datetime(df["published_at"], utc=True, errors="coerce")
            keep = ts.notna()
            if start is not None:
                keep &= ts >= pd.Timestamp(start, tz="UTC")
            if end is not None:
                keep &= ts <= pd.Timestamp(end, tz="UTC")
            df = df[keep].sort_values("published_at").reset_index(drop=True)

        found = _as_int(raw_meta.get("found"))
        returned = _as_int(raw_meta.get("returned"))
        plan_limit = _as_int(raw_meta.get("limit"))
        meta = {
            "ticker": ticker.strip().upper(),
            "requested_limit": self.limit,
            "provider_found": found,
            "provider_returned": returned,
            "provider_limit": plan_limit,
            "rows_after_normalisation": int(len(df)),
            "free_plan_article_limit": FREE_PLAN_ARTICLE_LIMIT,
            "free_plan_daily_requests": FREE_PLAN_DAILY_REQUESTS,
            "retrieved_at": pd.Timestamp.now(tz="UTC").isoformat(),
        }
        meta["plan_capped_the_request"] = bool(
            plan_limit is not None and plan_limit < self.limit
        )
        meta["withheld"] = (
            max(found - returned, 0)
            if (found is not None and returned is not None) else None
        )
        meta["truncated"] = bool(meta["withheld"])
        meta["dropped_in_normalisation"] = (
            max((returned or 0) - int(len(df)), 0) if returned is not None else None
        )
        self.last_meta = meta
        return df, meta

    def fetch(
        self, ticker: str, start: str | None = None, end: str | None = None
    ) -> pd.DataFrame:
        """NewsSource protocol entry point. Metadata stays on `last_meta`."""
        df, _ = self.fetch_with_meta(ticker, start, end)
        return df

    def fetch_many(
        self,
        tickers,
        start: str | None = None,
        end: str | None = None,
        on_message=None,
    ) -> tuple[pd.DataFrame, list[dict]]:
        """Fetch several tickers, pacing requests for the burst limit.

        Returns (articles, per_ticker_status). Every documented failure mode
        becomes its own status value -- `auth_error`, `usage_limit`,
        `rate_limited`, `plan_restricted`, `provider_error` -- and never an
        empty article list, because "the provider refused" and "this ticker has
        no news" are different facts and must not be conflated.
        """
        frames, status = [], []
        for i, raw_ticker in enumerate([t.strip().upper() for t in tickers]):
            if i and self.sleep_seconds:
                time.sleep(self.sleep_seconds)
            try:
                df, meta = self.fetch_with_meta(raw_ticker, start, end)
            except MarketauxError as exc:
                kind = {
                    MarketauxAuthError: "auth_error",
                    MarketauxUsageLimit: "usage_limit",
                    MarketauxRateLimit: "rate_limited",
                    MarketauxPlanRestricted: "plan_restricted",
                    MarketauxConfigError: "not_configured",
                }.get(type(exc), "provider_error")
                status.append({"ticker": raw_ticker, "status": kind, "n_articles": 0,
                               "detail": str(exc), "meta": {}})
                if on_message:
                    on_message(f"[ERROR] {raw_ticker}: {exc}")
                continue

            status.append({
                "ticker": raw_ticker,
                "status": "ok" if len(df) else "empty",
                "n_articles": int(len(df)),
                "detail": (
                    "" if len(df) else
                    "Provider returned no articles for this ticker and window."
                ),
                "meta": meta,
            })
            if on_message:
                note = ""
                if meta.get("truncated"):
                    note = (
                        f" (provider found {meta['provider_found']}, returned "
                        f"{meta['provider_returned']} -- free plan caps at "
                        f"{FREE_PLAN_ARTICLE_LIMIT}/request)"
                    )
                on_message(f"[OK] {raw_ticker}: {len(df)} articles{note}")
            if len(df):
                frames.append(df)

        articles = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        return articles, status


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _as_float(value) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _as_int(value) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _to_mx_time(value: str) -> str:
    """Our ISO-ish date -> Marketaux's `Y-m-d\\TH:i:s`."""
    return pd.Timestamp(value).strftime("%Y-%m-%dT%H:%M:%S")
