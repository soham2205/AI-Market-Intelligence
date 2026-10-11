"""Marketaux adapter tests -- fully mocked, no network, no real token.

Free-tier limits asserted here were verified 2026-10-11 against
marketaux.com/pricing: 100 requests/day and 3 ARTICLES PER NEWS REQUEST.
The three-article cap is the behaviour most of these tests exist to pin,
because a silently truncated result is the failure mode that would quietly
misrepresent a ticker's news.
"""

from __future__ import annotations

import json
import urllib.error
from pathlib import Path

import pandas as pd
import pytest

from app.data.sources.base import NEWS_SCHEMA
from app.data.sources.marketaux_source import (
    API_TOKEN_ENV,
    FREE_PLAN_ARTICLE_LIMIT,
    FREE_PLAN_DAILY_REQUESTS,
    MarketauxAuthError,
    MarketauxConfigError,
    MarketauxError,
    MarketauxNewsSource,
    MarketauxPlanRestricted,
    MarketauxProviderError,
    MarketauxRateLimit,
    MarketauxUsageLimit,
    api_token_available,
    load_dotenv_token,
    redacted_token_status,
)
from app.events.current_news_signal import run
from app.nlp.news_categories import classify_headline

FAKE_TOKEN = "mx-test-token-abcdef123456"
AV_FAKE_KEY = "AVTESTKEY1234567"
SESSIONS = pd.DatetimeIndex(pd.bdate_range("2024-01-01", periods=60))

PHASE1_EVENTS = Path("data/curated/phase1_events_v1.parquet")
PHASE2_SUMMARY = Path("data/curated/phase2_stock_category_horizon_summary_v1.parquet")
PHASE3_REPORT = Path("data/curated/phase3_report_v1.md")


def mx_article(title, published="2024-01-05T14:24:00.000000Z", symbol="AAPL", **kw):
    """One /v1/news/all data[] element in the provider's documented shape."""
    return {
        "uuid": kw.get("uuid", f"uuid-{abs(hash(title)) % 10**8}"),
        "title": title,
        "description": kw.get("description", f"Description for: {title}"),
        "snippet": kw.get("snippet", f"Snippet for: {title}"),
        "keywords": kw.get("keywords", ""),
        "url": kw.get("url", f"https://news.test/{abs(hash(title)) % 10**8}"),
        "image_url": "https://news.test/img.jpg",
        "language": "en",
        "published_at": published,
        "source": kw.get("source", "reuters.com"),
        "relevance_score": kw.get("relevance_score", None),
        "entities": kw.get("entities", [{
            "symbol": symbol, "name": "Test Co", "exchange": None,
            "exchange_long": None, "country": "us", "type": "equity",
            "industry": "Technology", "match_score": 35.4,
            # Provider sentiment: carried for reference, never used as ours.
            "sentiment_score": kw.get("sentiment_score", 0.8),
            "highlights": [],
        }]),
        "similar": [],
    }


def mx_payload(articles, found=None, limit=FREE_PLAN_ARTICLE_LIMIT, page=1):
    returned = len(articles)
    return json.dumps({
        "meta": {
            "found": found if found is not None else returned,
            "returned": returned,
            "limit": limit,
            "page": page,
        },
        "data": articles,
    })


def source_with(body, **kw):
    """Adapter wired to a recorded payload instead of the network."""
    opener = body if callable(body) else (lambda url: body)
    kw.setdefault("sleep_seconds", 0)
    return MarketauxNewsSource(env={API_TOKEN_ENV: FAKE_TOKEN}, opener=opener, **kw)


# --------------------------------------------------------------------------
# Verified free-tier limits
# --------------------------------------------------------------------------


def test_documented_free_tier_limits_are_encoded_as_constants():
    """Verified 2026-10-11 from marketaux.com/pricing. If these change, the
    reporting that depends on them must be revisited deliberately."""
    assert FREE_PLAN_ARTICLE_LIMIT == 3
    assert FREE_PLAN_DAILY_REQUESTS == 100


# --------------------------------------------------------------------------
# Credentials: separate variable, redaction, no cross-contamination
# --------------------------------------------------------------------------


def test_token_presence_detected_from_injected_env():
    assert api_token_available({API_TOKEN_ENV: FAKE_TOKEN})
    assert not api_token_available({})
    assert not api_token_available({API_TOKEN_ENV: "   "})


def test_redacted_status_never_reveals_any_part_of_the_token():
    status = redacted_token_status({API_TOKEN_ENV: FAKE_TOKEN})
    assert FAKE_TOKEN not in status
    assert FAKE_TOKEN[:6] not in status
    assert FAKE_TOKEN[-6:] not in status
    assert "set" in status and str(len(FAKE_TOKEN)) in status
    assert "NOT SET" in redacted_token_status({})


def test_marketaux_never_reads_or_overwrites_the_alphavantage_key():
    """The two credentials must stay fully independent."""
    env = {"ALPHAVANTAGE_API_KEY": AV_FAKE_KEY}
    assert not api_token_available(env)  # AV key must not satisfy Marketaux
    src = MarketauxNewsSource(env=env, sleep_seconds=0, opener=lambda url: "{}")
    assert not src.available()
    with pytest.raises(MarketauxConfigError):
        src.fetch("AAPL")
    # AV key untouched, and no Marketaux key invented
    assert env["ALPHAVANTAGE_API_KEY"] == AV_FAKE_KEY
    assert API_TOKEN_ENV not in env


def test_alphavantage_adapter_still_ignores_the_marketaux_token():
    from app.data.sources.alphavantage_source import api_key_available

    assert not api_key_available({API_TOKEN_ENV: FAKE_TOKEN})


def test_missing_token_error_names_the_variable_and_both_places():
    src = MarketauxNewsSource(env={}, sleep_seconds=0, opener=lambda url: "{}")
    with pytest.raises(MarketauxConfigError) as exc:
        src.fetch("AAPL")
    msg = str(exc.value)
    assert API_TOKEN_ENV in msg and ".env" in msg
    assert "ALPHAVANTAGE_API_KEY" in msg  # states independence
    assert "purchase" not in msg.lower() and "upgrade" not in msg.lower()


def test_dotenv_never_overwrites_an_exported_token(tmp_path):
    env = {API_TOKEN_ENV: "EXPORTED_WINS"}
    (tmp_path / ".env").write_text(f"{API_TOKEN_ENV}=FROM_DOTENV\n", encoding="utf-8")
    assert load_dotenv_token(tmp_path / ".env", env)
    assert env[API_TOKEN_ENV] == "EXPORTED_WINS"


def test_dotenv_reads_only_its_own_variable(tmp_path):
    env = {}
    (tmp_path / ".env").write_text(
        f"ALPHAVANTAGE_API_KEY=should-be-ignored\n{API_TOKEN_ENV}='mx-quoted'\n",
        encoding="utf-8",
    )
    assert load_dotenv_token(tmp_path / ".env", env)
    assert env[API_TOKEN_ENV] == "mx-quoted"
    assert "ALPHAVANTAGE_API_KEY" not in env


def test_token_is_sent_as_api_token_param_and_not_logged(capsys):
    seen = {}

    def opener(url):
        seen["url"] = url
        return mx_payload([mx_article("Acme Q4 earnings beat estimates")])

    source_with(opener).fetch("AAPL")
    assert f"api_token={FAKE_TOKEN}" in seen["url"]
    assert "api.marketaux.com/v1/news/all" in seen["url"]
    assert FAKE_TOKEN not in capsys.readouterr().out


# --------------------------------------------------------------------------
# Request construction
# --------------------------------------------------------------------------


def test_request_sends_symbols_limit_language_and_entity_filter():
    seen = {}

    def opener(url):
        seen["url"] = url
        return mx_payload([])

    source_with(opener, limit=3).fetch("aapl")
    assert "symbols=AAPL" in seen["url"]
    assert "limit=3" in seen["url"]
    assert "language=en" in seen["url"]
    assert "filter_entities=true" in seen["url"]


def test_request_passes_published_after_and_before_in_provider_format():
    seen = {}

    def opener(url):
        seen["url"] = url
        return mx_payload([])

    source_with(opener).fetch("AAPL", start="2024-01-01", end="2024-01-31")
    assert "published_after=2024-01-01T00%3A00%3A00" in seen["url"]
    assert "published_before=2024-01-31T00%3A00%3A00" in seen["url"]


# --------------------------------------------------------------------------
# Normalisation into the existing schema
# --------------------------------------------------------------------------


def test_fetch_returns_canonical_schema_columns_first():
    df = source_with(mx_payload([mx_article("Acme Q4 earnings beat")])).fetch("AAPL")
    assert list(df.columns)[: len(NEWS_SCHEMA)] == NEWS_SCHEMA


def test_all_required_fields_are_normalised():
    art = mx_article(
        "Acme unveils new platform",
        published="2024-01-05T14:24:00.000000Z",
        url="https://news.test/story-1",
        source="bloomberg.com",
        description="A longer description of the launch.",
        snippet="A snippet of the launch.",
    )
    row = source_with(mx_payload([art])).fetch("AAPL").iloc[0]
    assert row["headline"] == "Acme unveils new platform"
    assert row["description"] == "A longer description of the launch."
    assert row["snippet"] == "A snippet of the launch."
    assert row["url"] == "https://news.test/story-1"
    assert row["source"] == "bloomberg.com"
    assert row["ticker"] == "AAPL"
    assert str(pd.Timestamp(row["published_at"]).tz) == "UTC"
    assert pd.Timestamp(row["published_at"]).strftime("%Y-%m-%d %H:%M") == (
        "2024-01-05 14:24"
    )


def test_microsecond_utc_timestamp_format_parses():
    df = source_with(
        mx_payload([mx_article("x", published="2024-11-08T01:24:00.000000Z")])
    ).fetch("AAPL")
    assert pd.Timestamp(df.iloc[0]["published_at"]).strftime("%H:%M") == "01:24"


def test_entity_matching_uses_the_requested_symbol_not_the_first_entity():
    """Articles routinely mention several companies; attribution must follow
    the symbol we asked for."""
    art = mx_article("Multi-company story", entities=[
        {"symbol": "MSFT", "match_score": 10.0, "sentiment_score": 0.1},
        {"symbol": "AAPL", "match_score": 88.8, "sentiment_score": -0.4},
    ])
    row = source_with(mx_payload([art])).fetch("AAPL").iloc[0]
    assert row["mx_entity_symbol"] == "AAPL"
    assert row["mx_entity_match_score"] == pytest.approx(88.8)


def test_article_without_the_requested_entity_still_normalises():
    art = mx_article("Story", entities=[{"symbol": "TSLA", "match_score": 5.0}])
    row = source_with(mx_payload([art])).fetch("AAPL").iloc[0]
    assert row["ticker"] == "AAPL"
    assert row["mx_entity_symbol"] is None


def test_unparseable_or_missing_timestamp_is_dropped_not_backfilled():
    arts = [
        mx_article("Good", published="2024-01-05T14:24:00.000000Z"),
        mx_article("Bad date", published="not-a-date"),
        mx_article("No date", published=None),
    ]
    df = source_with(mx_payload(arts)).fetch("AAPL")
    assert df["headline"].tolist() == ["Good"]


def test_article_without_a_title_is_dropped():
    df = source_with(mx_payload([mx_article(""), mx_article("Real")])).fetch("AAPL")
    assert df["headline"].tolist() == ["Real"]


def test_article_id_matches_other_adapters_so_cross_provider_dedupe_works():
    from app.data.sources.news_source import dedupe_news

    art = mx_article("Shared headline", url="https://news.test/same")
    df = source_with(mx_payload([art, dict(art)])).fetch("AAPL")
    assert len(dedupe_news(df)) == 1


def test_local_window_filter_is_applied_regardless_of_provider_behaviour():
    arts = [
        mx_article("Too old", published="2024-01-01T12:00:00.000000Z"),
        mx_article("In window", published="2024-01-15T12:00:00.000000Z"),
        mx_article("Too new", published="2024-01-30T12:00:00.000000Z"),
    ]
    df = source_with(mx_payload(arts)).fetch("AAPL", start="2024-01-10", end="2024-01-20")
    assert df["headline"].tolist() == ["In window"]


# --------------------------------------------------------------------------
# The three-article free-tier cap: reporting what was withheld
# --------------------------------------------------------------------------


def test_meta_reports_the_gap_between_requested_and_returned():
    """The core free-tier behaviour: ask for 50, get 3 out of 412 matches."""
    arts = [mx_article(f"Story {i}") for i in range(3)]
    src = source_with(mx_payload(arts, found=412, limit=3), limit=50)
    df, meta = src.fetch_with_meta("AAPL")
    assert len(df) == 3
    assert meta["requested_limit"] == 50
    assert meta["provider_found"] == 412
    assert meta["provider_returned"] == 3
    assert meta["provider_limit"] == 3
    assert meta["withheld"] == 409
    assert meta["truncated"] is True
    assert meta["plan_capped_the_request"] is True


def test_untruncated_result_is_not_flagged_as_truncated():
    arts = [mx_article(f"Story {i}") for i in range(2)]
    src = source_with(mx_payload(arts, found=2, limit=3), limit=3)
    _, meta = src.fetch_with_meta("AAPL")
    assert meta["withheld"] == 0
    assert meta["truncated"] is False
    assert meta["plan_capped_the_request"] is False


def test_meta_records_rows_lost_in_normalisation_separately_from_plan_cap():
    """Dropping an undated article is OUR decision; the plan cap is the
    provider's. Conflating them would hide one behind the other."""
    arts = [mx_article("Good"), mx_article("Undated", published="nope")]
    src = source_with(mx_payload(arts, found=99, limit=3), limit=3)
    df, meta = src.fetch_with_meta("AAPL")
    assert meta["provider_returned"] == 2
    assert meta["rows_after_normalisation"] == 1
    assert meta["dropped_in_normalisation"] == 1
    assert len(df) == 1


def test_meta_carries_a_retrieval_timestamp():
    _, meta = source_with(mx_payload([mx_article("x")])).fetch_with_meta("AAPL")
    assert pd.Timestamp(meta["retrieved_at"]).tz is not None


def test_fetch_exposes_meta_via_last_meta_for_protocol_callers():
    src = source_with(mx_payload([mx_article("x")], found=50))
    src.fetch("AAPL")
    assert src.last_meta["provider_found"] == 50


def test_empty_result_is_an_empty_frame_with_schema_not_an_error():
    df, meta = source_with(mx_payload([], found=0)).fetch_with_meta("AAPL")
    assert df.empty
    assert list(df.columns)[: len(NEWS_SCHEMA)] == NEWS_SCHEMA
    assert meta["provider_found"] == 0
    assert meta["truncated"] is False


# --------------------------------------------------------------------------
# Documented error codes
# --------------------------------------------------------------------------


def error_body(code, message="boom"):
    return json.dumps({"error": {"code": code, "message": message}})


@pytest.mark.parametrize(
    "code,expected",
    [
        ("invalid_api_token", MarketauxAuthError),
        ("usage_limit_reached", MarketauxUsageLimit),
        ("rate_limit_reached", MarketauxRateLimit),
        ("endpoint_access_restricted", MarketauxPlanRestricted),
        ("malformed_parameters", MarketauxProviderError),
        ("server_error", MarketauxProviderError),
        ("maintenance_mode", MarketauxProviderError),
        ("resource_not_found", MarketauxProviderError),
        ("something_new", MarketauxProviderError),
    ],
)
def test_documented_error_codes_map_to_distinct_exceptions(code, expected):
    src = source_with(error_body(code))
    with pytest.raises(expected):
        src.fetch("AAPL")


def test_every_error_type_is_catchable_as_the_base_error():
    with pytest.raises(MarketauxError):
        source_with(error_body("usage_limit_reached")).fetch("AAPL")


def test_daily_quota_and_per_minute_burst_limits_are_different_exceptions():
    """The remedies differ: a daily quota cannot be waited out in a minute."""
    assert MarketauxUsageLimit is not MarketauxRateLimit
    assert not issubclass(MarketauxUsageLimit, MarketauxRateLimit)
    assert not issubclass(MarketauxRateLimit, MarketauxUsageLimit)


def test_error_envelope_in_an_http_error_body_is_still_parsed():
    """Marketaux returns its error envelope in the body of a 4xx, so the body
    must be read rather than only the status code."""

    def opener(url):
        raise urllib.error.HTTPError(
            url, 402, "Payment Required", {},
            __import__("io").BytesIO(error_body("usage_limit_reached").encode()),
        )

    with pytest.raises(MarketauxUsageLimit):
        source_with(opener).fetch("AAPL")


def test_non_json_body_raises_provider_error():
    with pytest.raises(MarketauxProviderError):
        source_with("<html>502 Bad Gateway</html>").fetch("AAPL")


def test_unreachable_host_raises_provider_error():
    def opener(url):
        raise urllib.error.URLError("dns failure")

    with pytest.raises(MarketauxProviderError):
        source_with(opener).fetch("AAPL")


# --------------------------------------------------------------------------
# fetch_many status reporting
# --------------------------------------------------------------------------


def test_fetch_many_distinguishes_empty_from_refused():
    """The whole point: 'no news' and 'the provider said no' are different
    facts and must never collapse into the same zero."""

    def opener(url):
        if "EMPTY" in url:
            return mx_payload([], found=0)
        if "QUOTA" in url:
            return error_body("usage_limit_reached")
        if "BURST" in url:
            return error_body("rate_limit_reached")
        if "BADTOK" in url:
            return error_body("invalid_api_token")
        return mx_payload([mx_article("Acme Q4 earnings beat")], found=1)

    articles, status = source_with(opener).fetch_many(
        ["AAPL", "EMPTY", "QUOTA", "BURST", "BADTOK"]
    )
    by = {s["ticker"]: s["status"] for s in status}
    assert by == {
        "AAPL": "ok", "EMPTY": "empty", "QUOTA": "usage_limit",
        "BURST": "rate_limited", "BADTOK": "auth_error",
    }
    assert len(articles) == 1
    # every non-ok row explains itself
    assert all(s["detail"] for s in status if s["status"] != "ok")


def test_fetch_many_keeps_per_ticker_meta_for_reporting():
    src = source_with(mx_payload([mx_article("x")], found=77, limit=3), limit=3)
    _, status = src.fetch_many(["AAPL"])
    assert status[0]["meta"]["provider_found"] == 77
    assert status[0]["meta"]["withheld"] == 76


def test_one_failing_ticker_does_not_abort_the_run():
    def opener(url):
        if "BAD" in url:
            return error_body("server_error")
        return mx_payload([mx_article("Acme Q4 earnings beat")], found=1)

    articles, status = source_with(opener).fetch_many(["BAD", "AAPL"])
    assert len(articles) == 1
    assert [s["status"] for s in status] == ["provider_error", "ok"]


# --------------------------------------------------------------------------
# Provider sentiment must never substitute for FinBERT
# --------------------------------------------------------------------------


def test_provider_sentiment_is_carried_but_is_not_the_pipeline_score():
    art = mx_article("Acme Q4 earnings beat", sentiment_score=0.95)
    df = source_with(mx_payload([art])).fetch("AAPL")
    assert df.iloc[0]["mx_entity_sentiment_score"] == pytest.approx(0.95)
    # the adapter must not emit anything that looks like our own score
    assert "sentiment_score" not in df.columns
    assert "sentiment_label" not in df.columns
    assert "score" not in df.columns


class StubScorer:
    """Deliberately disagrees with the provider's sentiment, so a substitution
    would be visible in the assertion below."""

    name = "stub-finbert"

    def score(self, text):
        return -1.0


def test_pipeline_scores_with_our_scorer_not_the_provider_value(tmp_path):
    from app.events.current_news_signal import score_and_classify

    art = mx_article("Acme Q4 earnings beat", sentiment_score=0.95)
    df = source_with(mx_payload([art])).fetch("AAPL")
    scored = score_and_classify(df, StubScorer(), SESSIONS)
    assert scored.iloc[0]["sentiment_score"] == pytest.approx(-1.0)
    assert scored.iloc[0]["scorer"] == "stub-finbert"
    # provider value preserved alongside, untouched
    assert scored.iloc[0]["mx_entity_sentiment_score"] == pytest.approx(0.95)


def test_marketaux_articles_flow_through_the_seven_category_classifier():
    arts = [
        mx_article("Acme Q4 Earnings & Revenues Beat Estimates"),
        mx_article("Citi downgrades Acme to neutral"),
        mx_article("Acme agrees to buy Beta Corp"),
        mx_article("Acme names new chief financial officer"),
        mx_article("Acme sued by DOJ in antitrust case"),
        mx_article("Acme unveils new AI platform"),
        mx_article("Why I still like this stock"),
    ]
    df = source_with(mx_payload(arts, found=7)).fetch("AAPL")
    cats = [classify_headline(h) for h in df["headline"]]
    assert set(cats) == {
        "EARNINGS_RESULTS", "ANALYST_ACTION", "M_AND_A", "MANAGEMENT",
        "REGULATORY_LEGAL", "MAJOR_COMPANY_NEWS", "OTHER",
    }


# --------------------------------------------------------------------------
# End to end through the pipeline
# --------------------------------------------------------------------------


def _write_panel(tmp_path):
    from app.data.curate import save_curated

    panel_dir = tmp_path / "panel"
    for t in ("AAA", "AAPL"):
        save_curated(
            pd.DataFrame({
                "date": SESSIONS, "ticker": t, "open": 100.0, "high": 101.0,
                "low": 99.0, "close": 100.0, "volume": 1e6,
            }),
            panel_dir, t, source="yfinance",
        )
    return panel_dir


def _write_phase2(tmp_path):
    summary = pd.DataFrame([
        {"ticker": "AAPL", "category": "EARNINGS_RESULTS", "horizon": 1,
         "n_events_valid": 300, "evidence_tier": "adequately_supported",
         "mean_abnormal_return": 0.002, "nw_mean": 0.002, "nw_ci95_lo": 0.0005,
         "nw_ci95_hi": 0.004, "shrunk_mean": 0.0018, "sector": "Technology",
         "sector_nw_mean": 0.001},
        {"ticker": "AAPL", "category": "MANAGEMENT", "horizon": 1,
         "n_events_valid": 9, "evidence_tier": "exploratory",
         "mean_abnormal_return": -0.01, "nw_mean": -0.01, "nw_ci95_lo": -0.04,
         "nw_ci95_hi": 0.02, "shrunk_mean": -0.002, "sector": "Technology",
         "sector_nw_mean": 0.0},
    ])
    vs_other = pd.DataFrame([
        {"ticker": "AAPL", "category": "EARNINGS_RESULTS", "horizon": 1,
         "n_events_category": 300, "diff_mean": 0.0015, "se": 0.0006,
         "p_value": 0.012, "ci95_lo": 0.0003, "ci95_hi": 0.0027,
         "significant_fdr": False, "sufficient_sample": True},
    ])
    sp, vp = tmp_path / "p2s.parquet", tmp_path / "p2v.parquet"
    summary.to_parquet(sp, index=False)
    vs_other.to_parquet(vp, index=False)
    return sp, vp


def _run_pipeline(tmp_path, src, out="out", **kw):
    sp, vp = _write_phase2(tmp_path)
    return run(
        tickers=kw.pop("tickers", ["AAPL"]),
        panel_dir=_write_panel(tmp_path),
        phase2_summary_path=sp, phase2_vs_other_path=vp,
        out_dir=tmp_path / out, version="t", horizons=(1,),
        scorer_name="lexicon", source=src, check_agreement=False,
        on_message=lambda m: None, **kw,
    )


def test_end_to_end_marketaux_through_finbert_classifier_and_phase2(tmp_path):
    arts = [
        mx_article("AAPL Q1 Earnings & Revenues Beat Estimates"),
        mx_article("AAPL names new chief financial officer"),
        mx_article("Why I think this stock is a long-term hold"),
    ]
    summary = _run_pipeline(
        tmp_path, source_with(mx_payload(arts, found=250, limit=3), limit=3)
    )
    assert summary["n_articles_retrieved"] == 3
    assert summary["source"] == "marketaux"
    assert summary["artifacts_written"] is True
    assert set(summary["category_counts"]) == {
        "EARNINGS_RESULTS", "MANAGEMENT", "OTHER"
    }
    # evidence tiers came from Phase 2 and include the thin ones
    assert "exploratory_history" in summary["evidence_status_counts"]
    report = Path(summary["report_path"]).read_text(encoding="utf-8")
    assert "3 articles per news request" in report
    assert "247" in report  # withheld = 250 - 3, reported not hidden


def test_pipeline_reports_the_plan_cap_and_withheld_count(tmp_path):
    arts = [mx_article(f"Story {i} earnings beat") for i in range(3)]
    summary = _run_pipeline(
        tmp_path, source_with(mx_payload(arts, found=412, limit=3), limit=50)
    )
    limits = summary["provider_limits"]
    assert limits["free_plan_article_limit_per_request"] == 3
    assert limits["total_matching_at_provider"] == 412
    assert limits["total_returned_by_provider"] == 3
    assert limits["total_withheld_by_plan"] == 409
    assert limits["any_request_capped_by_plan"] is True
    report = Path(summary["report_path"]).read_text(encoding="utf-8")
    assert "sample, not the full feed" in report


def test_empty_result_writes_no_misleading_artifacts(tmp_path):
    summary = _run_pipeline(tmp_path, source_with(mx_payload([], found=0)))
    assert summary["n_articles_retrieved"] == 0
    assert summary["artifacts_written"] is False
    written = sorted(p.name for p in (tmp_path / "out").iterdir())
    assert "current_news_articles_t.parquet" not in written
    assert "current_news_events_t.parquet" not in written
    assert "current_news_evidence_t.parquet" not in written
    # the run record and report ARE written, and explain the emptiness
    assert "current_news_run_t.json" in written
    report = Path(summary["report_path"]).read_text(encoding="utf-8")
    assert "No articles retrieved" in report
    assert "nothing to analyse" in report
    # no advertised path may point at a file this run chose not to write
    assert all(v is None for v in summary["outputs"].values()), summary["outputs"]


def test_empty_run_removes_a_previous_runs_artifacts(tmp_path):
    """Otherwise a stale parquet sits beside a report saying nothing was
    retrieved -- exactly how cached data gets read as current."""
    arts = [mx_article("AAPL Q1 Earnings Beat Estimates")]
    first = _run_pipeline(tmp_path, source_with(mx_payload(arts, found=1)))
    assert Path(first["outputs"]["articles"]).exists()
    second = _run_pipeline(tmp_path, source_with(mx_payload([], found=0)))
    assert second["artifacts_written"] is False
    assert not Path(first["outputs"]["articles"]).exists()


def test_refusal_is_reported_and_writes_no_artifacts(tmp_path):
    summary = _run_pipeline(tmp_path, source_with(error_body("usage_limit_reached")))
    assert summary["n_articles_retrieved"] == 0
    assert summary["artifacts_written"] is False
    assert summary["fetch_status"][0]["status"] == "usage_limit"
    report = Path(summary["report_path"]).read_text(encoding="utf-8")
    assert "not evidence of no news" in report
    assert "daily" in report.lower()


def test_report_states_the_price_panel_limitation_and_does_not_call_phase3_current(
    tmp_path,
):
    arts = [mx_article("AAPL Q1 Earnings Beat Estimates")]
    summary = _run_pipeline(tmp_path, source_with(mx_payload(arts, found=1)))
    report = Path(summary["report_path"]).read_text(encoding="utf-8")
    assert "price panel ends January 2024" in report
    assert "must not be presented as current" in report
    assert summary["forward_returns_computable"] is False


def test_run_never_writes_either_credential_into_any_output(tmp_path, monkeypatch):
    monkeypatch.setenv(API_TOKEN_ENV, FAKE_TOKEN)
    monkeypatch.setenv("ALPHAVANTAGE_API_KEY", AV_FAKE_KEY)
    arts = [mx_article("AAPL Q1 Earnings Beat Estimates")]
    summary = _run_pipeline(tmp_path, source_with(mx_payload(arts, found=1)))
    for path in (tmp_path / "out").iterdir():
        body = path.read_bytes().decode("utf-8", errors="ignore")
        assert FAKE_TOKEN not in body, path
        assert AV_FAKE_KEY not in body, path
    assert FAKE_TOKEN not in json.dumps(summary, default=str)
    assert AV_FAKE_KEY not in json.dumps(summary, default=str)


def test_run_is_reproducible_for_the_same_payload(tmp_path):
    arts = [mx_article("AAPL Q1 Earnings Beat Estimates")]
    body = mx_payload(arts, found=5, limit=3)
    s1 = _run_pipeline(tmp_path, source_with(body), out="o1")
    s2 = _run_pipeline(tmp_path, source_with(body), out="o2")
    pd.testing.assert_frame_equal(
        pd.read_parquet(s1["outputs"]["evidence"]),
        pd.read_parquet(s2["outputs"]["evidence"]),
    )


def test_missing_token_without_injected_source_explains_where_to_configure(
    tmp_path, monkeypatch
):
    monkeypatch.delenv(API_TOKEN_ENV, raising=False)
    monkeypatch.chdir(tmp_path)
    sp, vp = _write_phase2(tmp_path)
    with pytest.raises(RuntimeError) as exc:
        run(
            tickers=["AAPL"], panel_dir=_write_panel(tmp_path),
            phase2_summary_path=sp, phase2_vs_other_path=vp,
            out_dir=tmp_path / "out", version="t", horizons=(1,),
            scorer_name="lexicon", source_name="marketaux",
            check_agreement=False, on_message=lambda m: None,
        )
    msg = str(exc.value)
    assert API_TOKEN_ENV in msg and ".env" in msg


# --------------------------------------------------------------------------
# The other adapters still work, and earlier phases are untouched
# --------------------------------------------------------------------------


def test_all_three_adapters_remain_importable_and_protocol_shaped():
    from app.data.sources.alphavantage_source import AlphaVantageNewsSource
    from app.data.sources.marketaux_source import MarketauxNewsSource as MX
    from app.data.sources.news_source import YFinanceNewsSource

    for cls, name in (
        (YFinanceNewsSource, "yfinance"),
        (AlphaVantageNewsSource, "alphavantage"),
        (MX, "marketaux"),
    ):
        assert cls.name == name
        assert callable(cls.fetch)


def test_alphavantage_adapter_behaviour_is_unchanged():
    """Regression guard: adding Marketaux must not alter the AV adapter."""
    from app.data.sources.alphavantage_source import (
        API_KEY_ENV,
        AlphaVantageNewsSource,
    )

    payload = json.dumps({"feed": [{
        "title": "Acme Q4 earnings beat", "url": "https://x.test/1",
        "time_published": "20240115T143000", "source": "Wire",
        "overall_sentiment_score": 0.2, "overall_sentiment_label": "Bullish",
        "ticker_sentiment": [{"ticker": "AAPL", "relevance_score": "0.4"}],
    }]})
    src = AlphaVantageNewsSource(
        env={API_KEY_ENV: AV_FAKE_KEY}, sleep_seconds=0, opener=lambda url: payload
    )
    df = src.fetch("AAPL")
    assert len(df) == 1
    assert pd.Timestamp(df.iloc[0]["published_at"]).strftime("%H:%M") == "19:30"


@pytest.mark.parametrize("path", [PHASE1_EVENTS, PHASE2_SUMMARY, PHASE3_REPORT])
def test_marketaux_pipeline_does_not_modify_earlier_phase_outputs(path, tmp_path):
    if not path.exists():
        pytest.skip(f"{path} absent")
    before = path.stat().st_mtime, path.stat().st_size
    arts = [mx_article("AAPL Q1 Earnings Beat Estimates")]
    _run_pipeline(tmp_path, source_with(mx_payload(arts, found=1)))
    assert (path.stat().st_mtime, path.stat().st_size) == before
