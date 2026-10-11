"""Alpha Vantage adapter, 7-category classifier, and current-news pipeline.

Every test here is fully mocked: the transport is injected, so no HTTP request
is ever made and no real API key is read. Credential tests use a fake env dict
and assert on redaction behaviour, never on a key value.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from app.data.sources.alphavantage_source import (
    API_KEY_ENV,
    AlphaVantageConfigError,
    AlphaVantageNewsSource,
    AlphaVantageRateLimit,
    api_key_available,
    load_dotenv_key,
    redacted_key_status,
)
from app.data.sources.base import NEWS_SCHEMA
from app.events.current_news_signal import (
    _evidence_status,
    attach_phase2_evidence,
    collapse_to_events,
    render_report,
    run,
    score_and_classify,
)
from app.nlp.news_categories import (
    CATEGORIES,
    agreement_report,
    classify_frame,
    classify_headline,
    rule_description,
)

FAKE_KEY = "TESTKEY1234567890"
SESSIONS = pd.DatetimeIndex(pd.bdate_range("2024-01-01", periods=60))

PHASE1_CLASSIFIED = Path(
    "data/diagnostics_EXPERIMENTAL/news_category_vs_return/"
    "classified_articles_EXPERIMENTAL.parquet"
)
PHASE2_SUMMARY = Path("data/curated/phase2_stock_category_horizon_summary_v1.parquet")
PHASE2_VS_OTHER = Path("data/curated/phase2_stock_vs_other_v1.parquet")
PHASE3_REPORT = Path("data/curated/phase3_report_v1.md")


def av_payload(items):
    return json.dumps({"items": str(len(items)), "feed": items})


def av_item(title, published="20240115T143000", ticker="AAPL", **kw):
    item = {
        "title": title,
        "url": kw.get("url", f"https://example.test/{abs(hash(title)) % 10**8}"),
        "time_published": published,
        "source": kw.get("source", "Test Wire"),
        "overall_sentiment_score": kw.get("sentiment", 0.21),
        "overall_sentiment_label": kw.get("label", "Somewhat-Bullish"),
        "ticker_sentiment": [
            {"ticker": ticker, "relevance_score": kw.get("relevance", "0.45")}
        ],
    }
    return item


def source_with(items, **kw):
    """Adapter wired to a recorded payload instead of the network."""
    return AlphaVantageNewsSource(
        env={API_KEY_ENV: FAKE_KEY},
        sleep_seconds=0,
        opener=lambda url: av_payload(items),
        **kw,
    )


# --------------------------------------------------------------------------
# Credentials: presence, redaction, precedence
# --------------------------------------------------------------------------


def test_api_key_presence_is_detected_from_injected_env():
    assert api_key_available({API_KEY_ENV: FAKE_KEY})
    assert not api_key_available({})
    assert not api_key_available({API_KEY_ENV: "   "})


def test_redacted_status_never_reveals_any_part_of_the_key():
    status = redacted_key_status({API_KEY_ENV: FAKE_KEY})
    assert FAKE_KEY not in status
    # not even a prefix/suffix fragment
    assert FAKE_KEY[:4] not in status
    assert FAKE_KEY[-4:] not in status
    assert "set" in status and str(len(FAKE_KEY)) in status
    assert "NOT SET" in redacted_key_status({})


def test_missing_key_raises_a_configuration_error_naming_where_to_put_it():
    src = AlphaVantageNewsSource(env={}, sleep_seconds=0, opener=lambda url: "{}")
    assert not src.available()
    with pytest.raises(AlphaVantageConfigError) as exc:
        src.fetch("AAPL")
    msg = str(exc.value)
    assert API_KEY_ENV in msg
    # must point at configuring an existing key, never at obtaining a new one
    assert "register" not in msg.lower()
    assert "sign up" not in msg.lower()


def test_dotenv_never_overwrites_an_already_exported_key(tmp_path):
    env = {API_KEY_ENV: "EXPORTED_WINS"}
    p = tmp_path / ".env"
    p.write_text(f"{API_KEY_ENV}=FROM_DOTENV\n", encoding="utf-8")
    assert load_dotenv_key(p, env)
    assert env[API_KEY_ENV] == "EXPORTED_WINS"


def test_dotenv_is_used_only_when_env_is_empty(tmp_path):
    env = {}
    p = tmp_path / ".env"
    p.write_text(f"# comment\nOTHER=x\n{API_KEY_ENV}='quoted-key'\n", encoding="utf-8")
    assert load_dotenv_key(p, env)
    assert env[API_KEY_ENV] == "quoted-key"


def test_dotenv_absent_is_not_an_error(tmp_path):
    assert load_dotenv_key(tmp_path / "nope.env", {}) is False


def test_key_is_sent_as_a_query_param_and_not_logged(capsys):
    seen = {}

    def opener(url):
        seen["url"] = url
        return av_payload([av_item("Acme Q4 Earnings Beat Estimates")])

    src = AlphaVantageNewsSource(env={API_KEY_ENV: FAKE_KEY}, sleep_seconds=0, opener=opener)
    src.fetch("AAPL")
    assert f"apikey={FAKE_KEY}" in seen["url"]
    assert FAKE_KEY not in capsys.readouterr().out


# --------------------------------------------------------------------------
# Adapter: schema, parsing, timestamps, throttling
# --------------------------------------------------------------------------


def test_fetch_returns_canonical_news_schema_columns_first():
    df = source_with([av_item("Acme Q4 Earnings Beat Estimates")]).fetch("AAPL")
    assert list(df.columns)[: len(NEWS_SCHEMA)] == NEWS_SCHEMA
    assert len(df) == 1
    assert df.iloc[0]["ticker"] == "AAPL"


def test_timestamp_is_parsed_from_eastern_into_utc():
    """14:30 US/Eastern on 2024-01-15 (EST, UTC-5) is 19:30 UTC."""
    df = source_with([av_item("Acme Q4 Earnings Beat", published="20240115T143000")]).fetch(
        "AAPL"
    )
    ts = pd.Timestamp(df.iloc[0]["published_at"])
    assert str(ts.tz) == "UTC"
    assert ts.strftime("%Y-%m-%d %H:%M") == "2024-01-15 19:30"


def test_article_with_unparseable_timestamp_is_dropped_not_backfilled():
    """A synthesised publication time is a look-ahead bug; dropping is correct."""
    items = [
        av_item("Good headline", published="20240115T143000"),
        av_item("Undated headline", published="not-a-timestamp"),
        av_item("Missing timestamp", published=None),
    ]
    df = source_with(items).fetch("AAPL")
    assert len(df) == 1
    assert df.iloc[0]["headline"] == "Good headline"


def test_article_without_a_headline_is_dropped():
    df = source_with([av_item(""), av_item("Real headline")]).fetch("AAPL")
    assert len(df) == 1


def test_throttle_response_raises_instead_of_looking_like_zero_news():
    """HTTP 200 + 'Note' is Alpha Vantage's rate limit. Returning an empty frame
    would be indistinguishable from 'this ticker has no news'."""
    for advisory in ("Note", "Information", "Error Message"):
        src = AlphaVantageNewsSource(
            env={API_KEY_ENV: FAKE_KEY},
            sleep_seconds=0,
            opener=lambda url, a=advisory: json.dumps({a: "rate limit reached"}),
        )
        with pytest.raises(AlphaVantageRateLimit):
            src.fetch("AAPL")


def test_non_json_body_raises_rather_than_silently_returning_nothing():
    src = AlphaVantageNewsSource(
        env={API_KEY_ENV: FAKE_KEY}, sleep_seconds=0, opener=lambda url: "<html>502</html>"
    )
    with pytest.raises(AlphaVantageRateLimit):
        src.fetch("AAPL")


def test_empty_feed_returns_empty_frame_with_schema():
    df = source_with([]).fetch("AAPL")
    assert df.empty
    assert list(df.columns)[: len(NEWS_SCHEMA)] == NEWS_SCHEMA


def test_local_window_filter_is_applied_regardless_of_api_behaviour():
    items = [
        av_item("Old news", published="20240102T120000"),
        av_item("In window", published="20240115T120000"),
        av_item("Too new", published="20240130T120000"),
    ]
    df = source_with(items).fetch("AAPL", start="2024-01-10", end="2024-01-20")
    assert df["headline"].tolist() == ["In window"]


def test_fetch_many_records_rate_limit_as_status_not_as_zero_articles():
    calls = {"n": 0}

    def opener(url):
        calls["n"] += 1
        if "FAIL" in url:
            return json.dumps({"Note": "throttled"})
        return av_payload([av_item("Acme Q4 Earnings Beat Estimates")])

    src = AlphaVantageNewsSource(env={API_KEY_ENV: FAKE_KEY}, sleep_seconds=0, opener=opener)
    articles, status = src.fetch_many(["AAPL", "FAIL", "MSFT"])
    by_ticker = {s["ticker"]: s for s in status}
    assert by_ticker["FAIL"]["status"] == "rate_limited"
    assert by_ticker["AAPL"]["status"] == "ok"
    assert by_ticker["MSFT"]["status"] == "ok"
    assert len(articles) == 2  # the throttled ticker contributes nothing, loudly


def test_av_sentiment_is_carried_but_not_used_as_the_pipeline_score():
    df = source_with([av_item("Acme Q4 Earnings Beat", sentiment=0.9)]).fetch("AAPL")
    assert df.iloc[0]["av_overall_sentiment_score"] == pytest.approx(0.9)
    assert "sentiment_score" not in df.columns  # FinBERT adds that later


def test_article_id_matches_the_yfinance_construction_so_dedupe_works():
    from app.data.sources.news_source import dedupe_news

    item = av_item("Repeated headline", url="https://example.test/same")
    df = source_with([item, dict(item)]).fetch("AAPL")
    assert len(dedupe_news(df)) == 1


# --------------------------------------------------------------------------
# Seven-category classifier
# --------------------------------------------------------------------------


def test_classifier_categories_match_the_analysis_layer_exactly():
    """Phase 1-3 key their evidence on this tuple; drift would silently break
    every join between new articles and historical evidence."""
    from app.events.category_horizon_analysis import CATEGORIES as ANALYSIS_CATEGORIES

    assert set(CATEGORIES) == set(ANALYSIS_CATEGORIES)
    assert len(CATEGORIES) == 7


@pytest.mark.parametrize(
    "headline,expected",
    [
        ("Barclays Maintains Agilent (A) Underweight Recommendation", "ANALYST_ACTION"),
        ("Wolfe Research Initiates Coverage of Agilent With Outperform", "ANALYST_ACTION"),
        ("Morgan Stanley downgrades Acme on valuation", "ANALYST_ACTION"),
        ("Agilent (A) Q4 Earnings & Revenues Beat Estimates", "EARNINGS_RESULTS"),
        ("Danaher beats profit estimates on testing demand", "EARNINGS_RESULTS"),
        ("Agilent to Expand Portfolio With Avida Biomed Acquisition", "M_AND_A"),
        ("Acme agrees to buy Beta Corp in all-cash deal", "M_AND_A"),
        ("Apple sued with Visa, Mastercard in card-fee antitrust case", "REGULATORY_LEGAL"),
        ("FDA approval granted for Acme lung cancer drug", "REGULATORY_LEGAL"),
        ("Illumina names Agilent exec Jacob Thaysen as CEO", "MANAGEMENT"),
        ("Acme CFO steps down after four years", "MANAGEMENT"),
        ("Microsoft unveils unified AI assistant", "MAJOR_COMPANY_NEWS"),
        ("Roku Collaborates With Tennis Channel", "MAJOR_COMPANY_NEWS"),
        ("Acme announces layoffs across its retail division", "MAJOR_COMPANY_NEWS"),
        ("Stocks drift as traders await the Fed", "OTHER"),
        ("3 reasons I like this stock for the long haul", "OTHER"),
    ],
)
def test_classifier_assigns_expected_categories(headline, expected):
    assert classify_headline(headline) == expected


def test_classifier_is_deterministic_and_total():
    headlines = [
        "Acme Q3 earnings beat", "Beta downgraded by Citi", "Gamma acquires Delta",
        "Epsilon sued by DOJ", "Zeta names new CEO", "Eta unveils new chip",
        "random market chatter", "", None,
    ]
    first = [classify_headline(h) for h in headlines]
    second = [classify_headline(h) for h in headlines]
    assert first == second
    assert all(c in CATEGORIES for c in first)


def test_empty_or_missing_headline_is_other_not_an_error():
    assert classify_headline(None) == "OTHER"
    assert classify_headline("") == "OTHER"
    assert classify_headline(float("nan")) == "OTHER"


def test_priority_order_resolves_multi_signal_headlines():
    """A headline carrying two signals must resolve by documented priority, not
    by whichever rule happens to be checked first."""
    # acquisition + product expansion -> M_AND_A outranks MAJOR_COMPANY_NEWS
    assert classify_headline("Acme expands its portfolio with Beta acquisition") == "M_AND_A"
    # earnings + analyst rating -> ANALYST_ACTION is priority 1
    assert (
        classify_headline("Citi maintains Buy recommendation ahead of Q4 earnings")
        == "ANALYST_ACTION"
    )


def test_classifier_does_not_reproduce_the_historical_sec_substring_bug():
    """'GPC/SEC Solutions' is chromatography, not the SEC regulator. The old
    labeller called this REGULATORY_LEGAL; this classifier must not."""
    assert (
        classify_headline("Agilent (A) Boosts LSAG Segment With New GPC/SEC Solutions")
        != "REGULATORY_LEGAL"
    )


def test_classifier_biases_toward_other_rather_than_inventing_events():
    """Over-classifying commentary as an event manufactures false signal, which
    is the worse failure mode -- these generic headlines must stay OTHER."""
    for headline in (
        "Time to Buy Microsoft (MSFT) Stock?",
        "3 Dividend Stocks to Buy in October",
        "Is Apple a good long-term hold?",
        "Market wrap: stocks end mixed",
    ):
        assert classify_headline(headline) == "OTHER", headline


def test_classify_frame_handles_empty_input():
    empty = pd.DataFrame({"headline": pd.Series([], dtype=object)})
    assert classify_frame(empty).empty


def test_rule_description_exposes_priority_ordered_rules():
    rules = rule_description()
    assert [r["priority"] for r in rules] == list(range(1, len(rules) + 1))
    assert rules[0]["category"] == "ANALYST_ACTION"


def test_agreement_report_shape_on_a_small_reference():
    ref = pd.DataFrame({
        "headline": ["Acme Q4 earnings beat estimates", "Citi downgrades Beta",
                     "random chatter"],
        "category": ["EARNINGS_RESULTS", "ANALYST_ACTION", "OTHER"],
    })
    rep = agreement_report(ref)
    assert rep["n"] == 3
    assert rep["overall_agreement"] == 1.0
    assert {p["category"] for p in rep["per_category"]} == {
        "EARNINGS_RESULTS", "ANALYST_ACTION", "OTHER"
    }


def test_agreement_report_handles_empty_reference():
    rep = agreement_report(pd.DataFrame({"headline": [], "category": []}))
    assert rep["n"] == 0 and rep["overall_agreement"] is None


@pytest.mark.skipif(not PHASE1_CLASSIFIED.exists(), reason="historical labels absent")
def test_measured_agreement_against_real_historical_labels_stays_high():
    """Pins the reported comparability figure. If a rule change moves this, the
    number quoted in the report and docstring must move with it."""
    ref = pd.read_parquet(PHASE1_CLASSIFIED, columns=["headline", "category"]).head(200_000)
    rep = agreement_report(ref)
    assert rep["overall_agreement"] > 0.85
    recall = {p["category"]: p["recall"] for p in rep["per_category"]}
    assert recall["EARNINGS_RESULTS"] > 0.95
    assert recall["OTHER"] > 0.85


# --------------------------------------------------------------------------
# Scoring, bucketing, event collapse
# --------------------------------------------------------------------------


class StubScorer:
    name = "stub"

    def score(self, text):
        return 0.5 if "beat" in (text or "").lower() else -0.25


class StubBatchScorer(StubScorer):
    name = "stub-batch"

    def score_batch(self, texts, batch_size=64):
        return [
            {"score": self.score(t), "label": "positive" if self.score(t) > 0 else "negative"}
            for t in texts
        ]


def _articles(items, ticker="AAPL"):
    return source_with(items).fetch(ticker)


def test_score_and_classify_adds_sentiment_and_category():
    df = _articles([av_item("Acme Q4 Earnings Beat Estimates", published="20240110T080000")])
    out = score_and_classify(df, StubScorer(), SESSIONS)
    assert out.iloc[0]["category"] == "EARNINGS_RESULTS"
    assert out.iloc[0]["sentiment_score"] == pytest.approx(0.5)
    assert out.iloc[0]["scorer"] == "stub"


def test_batch_scorer_is_used_when_available():
    df = _articles([av_item("Acme Q4 Earnings Beat", published="20240110T080000")])
    out = score_and_classify(df, StubBatchScorer(), SESSIONS)
    assert out.iloc[0]["sentiment_label"] == "positive"


def test_article_before_the_open_maps_to_the_same_session():
    """08:00 ET is pre-open, so it belongs to that same trading day."""
    df = _articles([av_item("Acme news", published="20240110T080000")])
    out = score_and_classify(df, StubScorer(), SESSIONS)
    assert pd.Timestamp(out.iloc[0]["trade_date"]).date() == pd.Timestamp("2024-01-10").date()
    assert out.iloc[0]["trade_date_status"] == "mapped"


def test_article_after_the_open_rolls_to_the_next_session():
    df = _articles([av_item("Acme news", published="20240110T143000")])
    out = score_and_classify(df, StubScorer(), SESSIONS)
    assert pd.Timestamp(out.iloc[0]["trade_date"]) > pd.Timestamp("2024-01-10")


def test_article_past_the_price_panel_is_kept_and_flagged_not_dropped():
    """The whole point of current news: an article newer than our price data
    still matters, but no forward return can be attributed to it."""
    df = _articles([av_item("Acme news today", published="20260601T120000")])
    out = score_and_classify(df, StubScorer(), SESSIONS)
    assert len(out) == 1
    assert pd.isna(out.iloc[0]["trade_date"])
    assert out.iloc[0]["trade_date_status"] == "beyond_price_panel"


def test_collapse_counts_same_day_same_category_articles_as_one_event():
    items = [
        av_item("Acme Q4 Earnings Beat Estimates", published="20240110T080000"),
        av_item("Acme Q4 revenues top views", published="20240110T081000"),
        av_item("Citi downgrades Acme", published="20240110T082000"),
    ]
    scored = score_and_classify(_articles(items), StubScorer(), SESSIONS)
    events = collapse_to_events(scored)
    earnings = events[events["category"] == "EARNINGS_RESULTS"]
    assert len(earnings) == 1
    assert int(earnings.iloc[0]["n_articles"]) == 2
    # different categories on the same day stay separate, as in Phase 1
    assert set(events["category"]) == {"EARNINGS_RESULTS", "ANALYST_ACTION"}


def test_collapse_handles_empty_input():
    assert collapse_to_events(pd.DataFrame()).empty


# --------------------------------------------------------------------------
# Phase 2 evidence join
# --------------------------------------------------------------------------


def _phase2_fixtures():
    summary = pd.DataFrame([
        {"ticker": "AAA", "category": "EARNINGS_RESULTS", "horizon": 1,
         "n_events_valid": 220, "evidence_tier": "adequately_supported",
         "mean_abnormal_return": 0.004, "nw_mean": 0.004, "nw_ci95_lo": 0.001,
         "nw_ci95_hi": 0.007, "shrunk_mean": 0.003, "sector": "Industrials",
         "sector_nw_mean": 0.002},
        {"ticker": "AAA", "category": "MANAGEMENT", "horizon": 1,
         "n_events_valid": 12, "evidence_tier": "exploratory",
         "mean_abnormal_return": -0.02, "nw_mean": -0.02, "nw_ci95_lo": -0.05,
         "nw_ci95_hi": 0.01, "shrunk_mean": -0.005, "sector": "Industrials",
         "sector_nw_mean": 0.0},
        {"ticker": "AAA", "category": "M_AND_A", "horizon": 1,
         "n_events_valid": 3, "evidence_tier": "insufficient",
         "mean_abnormal_return": 0.09, "nw_mean": 0.09, "nw_ci95_lo": None,
         "nw_ci95_hi": None, "shrunk_mean": 0.001, "sector": "Industrials",
         "sector_nw_mean": 0.0},
    ])
    vs_other = pd.DataFrame([
        {"ticker": "AAA", "category": "EARNINGS_RESULTS", "horizon": 1,
         "n_events_category": 220, "diff_mean": 0.003, "se": 0.001, "p_value": 0.0001,
         "ci95_lo": 0.001, "ci95_hi": 0.005, "significant_fdr": True,
         "sufficient_sample": True},
        {"ticker": "AAA", "category": "MANAGEMENT", "horizon": 1,
         "n_events_category": 12, "diff_mean": -0.01, "se": 0.02, "p_value": 0.6,
         "ci95_lo": -0.05, "ci95_hi": 0.03, "significant_fdr": False,
         "sufficient_sample": False},
    ])
    return summary, vs_other


def _events_fixture():
    return pd.DataFrame([
        {"ticker": "AAA", "event_date": pd.Timestamp("2024-01-10"),
         "category": "EARNINGS_RESULTS", "n_articles": 2, "mean_sentiment": 0.5,
         "trade_date_status": "mapped", "headlines": "Acme Q4 beat"},
        {"ticker": "AAA", "event_date": pd.Timestamp("2024-01-10"),
         "category": "MANAGEMENT", "n_articles": 1, "mean_sentiment": -0.25,
         "trade_date_status": "mapped", "headlines": "CFO steps down"},
        {"ticker": "AAA", "event_date": pd.Timestamp("2024-01-10"),
         "category": "M_AND_A", "n_articles": 1, "mean_sentiment": 0.1,
         "trade_date_status": "mapped", "headlines": "buys Beta"},
        {"ticker": "ZZZ", "event_date": pd.Timestamp("2024-01-10"),
         "category": "EARNINGS_RESULTS", "n_articles": 1, "mean_sentiment": 0.2,
         "trade_date_status": "mapped", "headlines": "unknown ticker"},
    ])


def test_evidence_status_classification_is_exhaustive():
    assert _evidence_status(None, None) == "no_history"
    assert _evidence_status(float("nan"), None) == "no_history"
    assert _evidence_status("insufficient", False) == "insufficient_history"
    assert _evidence_status("exploratory", False) == "exploratory_history"
    assert _evidence_status("adequately_supported", False) == "supported_not_significant"
    assert _evidence_status("adequately_supported", True) == "supported_and_significant"


def test_missing_fdr_flag_is_never_reported_as_significant():
    """Regression: `bool(float('nan'))` is True in Python, so a missing vs-OTHER
    row was being reported as a statistically significant result. A left join
    leaves NaN whenever Phase 2 has no comparison for that cell."""
    assert _evidence_status("adequately_supported", float("nan")) == (
        "supported_no_comparison"
    )
    assert _evidence_status("adequately_supported", None) == "supported_no_comparison"


def test_other_category_events_get_no_comparison_not_significance(tmp_path):
    """OTHER is the baseline Phase 2 compares everything else against, so it has
    no vs-OTHER row by construction -- and must not inherit significance."""
    summary = pd.DataFrame([{
        "ticker": "AAA", "category": "OTHER", "horizon": 1, "n_events_valid": 900,
        "evidence_tier": "adequately_supported", "mean_abnormal_return": 0.0005,
        "nw_mean": 0.0005, "nw_ci95_lo": -0.001, "nw_ci95_hi": 0.002,
        "shrunk_mean": 0.0004, "sector": "Industrials", "sector_nw_mean": 0.0,
    }])
    vs_other = pd.DataFrame([{  # deliberately contains no OTHER row
        "ticker": "AAA", "category": "EARNINGS_RESULTS", "horizon": 1,
        "n_events_category": 220, "diff_mean": 0.003, "se": 0.001, "p_value": 0.0001,
        "ci95_lo": 0.001, "ci95_hi": 0.005, "significant_fdr": True,
        "sufficient_sample": True,
    }])
    events = pd.DataFrame([{
        "ticker": "AAA", "event_date": pd.Timestamp("2024-01-10"), "category": "OTHER",
        "n_articles": 3, "mean_sentiment": 0.1, "trade_date_status": "mapped",
        "headlines": "generic market piece",
    }])
    ev = attach_phase2_evidence(events, summary, vs_other, horizons=(1,))
    assert ev.iloc[0]["evidence_status"] == "supported_no_comparison"
    assert not ev.iloc[0]["evidence_is_actionable"]


@pytest.mark.skipif(
    not (PHASE2_SUMMARY.exists() and PHASE2_VS_OTHER.exists()),
    reason="Phase 2 outputs absent",
)
def test_actionable_rows_cannot_exceed_phase2s_own_significant_cell_count():
    """Cross-check against Phase 2's ground truth: its entire vs-OTHER table
    contains exactly 2 FDR-significant cells, so no join over a handful of
    tickers may ever surface more than that."""
    vs_other = pd.read_parquet(PHASE2_VS_OTHER)
    n_significant = int(vs_other["significant_fdr"].fillna(False).astype(bool).sum())
    summary = pd.read_parquet(PHASE2_SUMMARY)
    tickers = ["AAPL", "BEN", "BR", "MSFT"]
    events = pd.DataFrame([
        {"ticker": t, "event_date": pd.Timestamp("2024-01-05"), "category": c,
         "n_articles": 1, "mean_sentiment": 0.0, "trade_date_status": "mapped",
         "headlines": "x"}
        for t in tickers for c in CATEGORIES
    ])
    ev = attach_phase2_evidence(events, summary, vs_other)
    assert int(ev["evidence_is_actionable"].sum()) <= n_significant


def test_join_marks_each_event_with_its_evidence_strength():
    summary, vs_other = _phase2_fixtures()
    ev = attach_phase2_evidence(_events_fixture(), summary, vs_other, horizons=(1,))
    by = {(r["ticker"], r["category"]): r for _, r in ev.iterrows()}
    assert by[("AAA", "EARNINGS_RESULTS")]["evidence_status"] == "supported_and_significant"
    assert by[("AAA", "EARNINGS_RESULTS")]["evidence_is_actionable"]
    assert by[("AAA", "MANAGEMENT")]["evidence_status"] == "exploratory_history"
    assert by[("AAA", "M_AND_A")]["evidence_status"] == "insufficient_history"
    assert not by[("AAA", "M_AND_A")]["evidence_is_actionable"]


def test_ticker_absent_from_phase2_is_reported_as_no_history_not_dropped():
    """A missing row is the finding most worth surfacing; it must not vanish."""
    summary, vs_other = _phase2_fixtures()
    ev = attach_phase2_evidence(_events_fixture(), summary, vs_other, horizons=(1,))
    zzz = ev[ev["ticker"] == "ZZZ"]
    assert len(zzz) == 1
    assert zzz.iloc[0]["evidence_status"] == "no_history"
    assert not zzz.iloc[0]["evidence_is_actionable"]


def test_join_covers_every_requested_horizon():
    summary, vs_other = _phase2_fixtures()
    ev = attach_phase2_evidence(
        _events_fixture(), summary, vs_other, horizons=(1, 5, 20)
    )
    assert len(ev) == len(_events_fixture()) * 3
    # horizons Phase 2 has no row for still appear, as no_history
    assert set(ev[ev["horizon"] == 20]["evidence_status"]) == {"no_history"}


def test_join_handles_empty_events():
    summary, vs_other = _phase2_fixtures()
    assert attach_phase2_evidence(pd.DataFrame(), summary, vs_other).empty


def test_every_evidence_status_has_a_plain_language_note():
    from app.events.current_news_signal import EVIDENCE_STATUS

    summary, vs_other = _phase2_fixtures()
    ev = attach_phase2_evidence(_events_fixture(), summary, vs_other, horizons=(1,))
    assert ev["evidence_note"].notna().all()
    assert set(ev["evidence_status"]) <= set(EVIDENCE_STATUS)


# --------------------------------------------------------------------------
# End to end, fully mocked
# --------------------------------------------------------------------------


def _write_panel(tmp_path):
    from app.data.curate import save_curated

    panel_dir = tmp_path / "panel"
    for t in ("AAA", "ZZZ"):
        save_curated(
            pd.DataFrame({
                "date": SESSIONS, "ticker": t, "open": 100.0, "high": 101.0,
                "low": 99.0, "close": 100.0, "volume": 1e6,
            }),
            panel_dir, t, source="yfinance",
        )
    return panel_dir


def _write_phase2(tmp_path):
    summary, vs_other = _phase2_fixtures()
    sp = tmp_path / "p2_summary.parquet"
    vp = tmp_path / "p2_vs_other.parquet"
    summary.to_parquet(sp, index=False)
    vs_other.to_parquet(vp, index=False)
    return sp, vp


def test_run_end_to_end_writes_outputs_and_states_its_limits(tmp_path):
    items = [
        av_item("AAA Q4 Earnings Beat Estimates", published="20240110T080000"),
        av_item("AAA CFO steps down", published="20240110T090000"),
    ]
    sp, vp = _write_phase2(tmp_path)
    out_dir = tmp_path / "out"
    summary = run(
        tickers=["AAA"],
        panel_dir=_write_panel(tmp_path),
        phase2_summary_path=sp,
        phase2_vs_other_path=vp,
        out_dir=out_dir,
        version="t",
        horizons=(1,),
        scorer_name="lexicon",
        source=source_with(items),
        check_agreement=False,
        on_message=lambda m: None,
    )
    assert summary["n_articles_retrieved"] == 2
    assert summary["n_events"] == 2
    assert summary["forward_returns_computable"] is False
    for key in ("articles", "events", "evidence"):
        assert Path(summary["outputs"][key]).exists()
    assert Path(summary["run_path"]).exists()
    report = Path(summary["report_path"]).read_text(encoding="utf-8")
    assert "Not a forecast" in report
    assert "No forward return is computed" in report


def test_run_writes_only_current_news_artifacts(tmp_path):
    sp, vp = _write_phase2(tmp_path)
    out_dir = tmp_path / "out"
    run(
        tickers=["AAA"],
        panel_dir=_write_panel(tmp_path),
        phase2_summary_path=sp, phase2_vs_other_path=vp,
        out_dir=out_dir, version="t", horizons=(1,), scorer_name="lexicon",
        source=source_with([av_item("AAA Q4 Earnings Beat", published="20240110T080000")]),
        check_agreement=False, on_message=lambda m: None,
    )
    written = sorted(p.name for p in out_dir.iterdir())
    assert written and all(n.startswith("current_news_") for n in written), written


def test_run_never_writes_the_key_into_any_output(tmp_path, monkeypatch):
    monkeypatch.setenv(API_KEY_ENV, FAKE_KEY)
    sp, vp = _write_phase2(tmp_path)
    out_dir = tmp_path / "out"
    summary = run(
        tickers=["AAA"],
        panel_dir=_write_panel(tmp_path),
        phase2_summary_path=sp, phase2_vs_other_path=vp,
        out_dir=out_dir, version="t", horizons=(1,), scorer_name="lexicon",
        source=source_with([av_item("AAA Q4 Earnings Beat", published="20240110T080000")]),
        check_agreement=False, on_message=lambda m: None,
    )
    for path in out_dir.iterdir():
        assert FAKE_KEY not in path.read_bytes().decode("utf-8", errors="ignore"), path
    assert FAKE_KEY not in json.dumps(summary, default=str)


def test_run_is_reproducible_for_the_same_payload(tmp_path):
    sp, vp = _write_phase2(tmp_path)
    items = [av_item("AAA Q4 Earnings Beat Estimates", published="20240110T080000")]
    kw = dict(
        tickers=["AAA"], panel_dir=_write_panel(tmp_path),
        phase2_summary_path=sp, phase2_vs_other_path=vp,
        version="t", horizons=(1,), scorer_name="lexicon",
        check_agreement=False, on_message=lambda m: None,
    )
    s1 = run(out_dir=tmp_path / "o1", source=source_with(items), **kw)
    s2 = run(out_dir=tmp_path / "o2", source=source_with(items), **kw)
    pd.testing.assert_frame_equal(
        pd.read_parquet(s1["outputs"]["evidence"]), pd.read_parquet(s2["outputs"]["evidence"])
    )

    # The report embeds `retrieved_at`, which SHOULD differ between two
    # separate retrievals -- that line is how a reader tells a fresh run from
    # a cached one, so pinning it byte-equal would defeat its purpose. The
    # analysis must be identical, so everything else is compared.
    def without_retrieval_time(path):
        return [
            ln for ln in Path(path).read_text(encoding="utf-8").splitlines()
            if "Retrieved at:" not in ln
        ]

    assert without_retrieval_time(s1["report_path"]) == without_retrieval_time(
        s2["report_path"]
    )


def test_run_without_credentials_or_source_explains_where_to_configure(tmp_path, monkeypatch):
    monkeypatch.delenv(API_KEY_ENV, raising=False)
    monkeypatch.chdir(tmp_path)  # so no repo .env is picked up
    sp, vp = _write_phase2(tmp_path)
    with pytest.raises(RuntimeError) as exc:
        run(
            tickers=["AAA"], panel_dir=_write_panel(tmp_path),
            phase2_summary_path=sp, phase2_vs_other_path=vp,
            out_dir=tmp_path / "out", version="t", horizons=(1,),
            scorer_name="lexicon", check_agreement=False, on_message=lambda m: None,
        )
    msg = str(exc.value)
    assert API_KEY_ENV in msg and ".env" in msg
    assert "register" not in msg.lower() and "sign up" not in msg.lower()


def test_report_says_none_when_no_event_has_strong_history(tmp_path):
    """The expected real-world outcome, and it must be stated plainly rather
    than left as an empty table."""
    summary_df, vs_other = _phase2_fixtures()
    events = _events_fixture()
    events = events[events["category"] != "EARNINGS_RESULTS"]  # drop the strong one
    ev = attach_phase2_evidence(events, summary_df, vs_other, horizons=(1,))
    report = render_report(
        pd.DataFrame(), events, ev,
        {"purpose": "p", "forward_returns_note": "n", "source": "s", "scorer": "x",
         "credentials": {"status": "set"}, "tickers_requested": ["AAA"],
         "n_articles_retrieved": 0, "n_events": len(events), "fetch_status": [],
         "evidence_status_counts": ev.groupby("evidence_status").size().to_dict(),
         "classifier_agreement_vs_historical": None, "caveats": ["c"]},
    )
    assert "**None.**" in report
    assert "not evidence of no effect" in report


# --------------------------------------------------------------------------
# Non-interference with Phase 0-3
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "path", [PHASE1_CLASSIFIED, PHASE2_SUMMARY, PHASE2_VS_OTHER, PHASE3_REPORT]
)
def test_current_news_pipeline_does_not_modify_earlier_phase_outputs(path, tmp_path):
    if not path.exists():
        pytest.skip(f"{path} absent")
    before = path.stat().st_mtime, path.stat().st_size
    sp, vp = _write_phase2(tmp_path)
    run(
        tickers=["AAA"], panel_dir=_write_panel(tmp_path),
        phase2_summary_path=sp, phase2_vs_other_path=vp,
        out_dir=tmp_path / "out", version="t", horizons=(1,), scorer_name="lexicon",
        source=source_with([av_item("AAA Q4 Earnings Beat", published="20240110T080000")]),
        check_agreement=False, on_message=lambda m: None,
    )
    assert (path.stat().st_mtime, path.stat().st_size) == before
