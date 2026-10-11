"""Current-news pipeline: Alpha Vantage -> FinBERT -> 7 categories -> Phase 2 evidence.

    AlphaVantageNewsSource.fetch_many()          (current/recent articles)
            |
            |  dedupe_news()                     <- REUSED, unchanged
            |  first_tradable_session()           <- REUSED, unchanged
            |  get_scorer("finbert").score_batch  <- REUSED, unchanged
            |  classify_frame()                   <- the new reproducible
            |                                        7-category classifier
            v
    scored + classified articles
            |
            |  collapse to one row per (ticker, trade_date, category),
            |  n_articles = how many collapsed -- the SAME counting unit
            |  Phase 1 defined, so an event here means what it means there
            v
    current events
            |
            |  left-join Phase 2 stock-sensitivity outputs on
            |  (ticker, category, horizon), READ-ONLY
            v
    current_news_articles_v1.parquet
    current_news_events_v1.parquet
    current_news_evidence_v1.parquet
    current_news_run_v1.json
    current_news_report_v1.md

WHAT THIS DOES AND DOES NOT TELL YOU. It answers "a story of category C just
landed on ticker T -- what has historically followed that combination, and how
good is that evidence?" It does NOT forecast this article's return, and it
cannot: the answer is Phase 2's historical association, carried over to a new
article only on the assumption that the relationship still holds. Phase 3
already showed that assumption is often unverifiable. Where Phase 2's evidence
is thin, this pipeline says so in the loudest terms available to it rather than
printing a number that looks actionable.

TWO STRUCTURAL LIMITS, SURFACED NOT HIDDEN:

1. NO FORWARD RETURN IS COMPUTABLE FOR CURRENT NEWS. The curated price panel
   ends 2024-01-08. `first_tradable_session` correctly returns NaT for anything
   past the end of the known calendar, so a 2026 article has no session to
   attribute to and no future prices to measure. Phase 1's loader DROPS such
   rows; this pipeline keeps them and labels them
   `trade_date_status = beyond_price_panel`, because an article we cannot score
   for outcome is still the article the user is asking about. Every such row is
   excluded from anything resembling an outcome measurement.

2. THE CATEGORY LABELS ARE ONLY ~91% COMPARABLE WITH PHASE 2's. Phase 2's
   evidence is keyed on the historical category labels, which had no generating
   code. The new classifier agrees with them 91.0% of the time overall, and as
   low as 0.49 recall on MAJOR_COMPANY_NEWS (see app.nlp.news_categories). So
   joining a newly-labelled article to Phase 2 evidence is approximate, and the
   run report states the measured rate rather than implying an exact match.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from app.data.curate import load_curated
from app.data.sources.alphavantage_source import (
    AlphaVantageNewsSource,
    api_key_available,
    load_dotenv_key,
    redacted_key_status,
)
from app.data.sources.news_source import dedupe_news
from app.events.category_horizon_analysis import HORIZONS
from app.nlp.bucketing import first_tradable_session
from app.nlp.finbert_scorer import get_scorer
from app.nlp.news_categories import CATEGORIES, classify_frame

DEFAULT_PANEL_DIR = "data/curated_sp500"
DEFAULT_PHASE2_SUMMARY = Path("data/curated/phase2_stock_category_horizon_summary_v1.parquet")
DEFAULT_PHASE2_VS_OTHER = Path("data/curated/phase2_stock_vs_other_v1.parquet")
DEFAULT_OUT_DIR = Path("data/curated")
OUTPUT_VERSION = "v1"

# Phase 2's own evidence tiers, mapped to what a reader should do about them.
EVIDENCE_STATUS = {
    "no_history": "No Phase 2 row for this ticker+category: no historical evidence at all.",
    "insufficient_history": "Phase 2 judged this ticker+category INSUFFICIENTLY sampled.",
    "exploratory_history": "Phase 2 tier 'exploratory': too few events to rely on.",
    "supported_no_comparison": (
        "Adequately sampled, but no vs-OTHER test exists -- this IS the OTHER "
        "baseline category, so there is nothing to compare it against."
    ),
    "supported_not_significant": (
        "Adequately sampled, but NOT distinguishable from this stock's other news "
        "after multiple-testing correction."
    ),
    "supported_and_significant": (
        "Adequately sampled AND FDR-significant versus this stock's own OTHER news."
    ),
}


def resolve_credentials(
    provider: str = "alphavantage", dotenv_path: str | Path = ".env"
) -> dict:
    """Find the credential for ONE provider without revealing it.

    Order: existing environment first, then a git-ignored .env as a fallback.
    An exported value is never overwritten, so this cannot silently swap which
    credential is in use. Each provider reads only its own variable --
    ALPHAVANTAGE_API_KEY and MARKETAUX_API_TOKEN are independent and neither
    call touches the other.
    """
    if provider == "marketaux":
        from app.data.sources.marketaux_source import (
            API_TOKEN_ENV,
            api_token_available,
            load_dotenv_token,
        )

        pre_existing = api_token_available()
        found = load_dotenv_token(dotenv_path)
        status, env_var = redacted_token_status_for_marketaux(), API_TOKEN_ENV
    elif provider == "alphavantage":
        pre_existing = api_key_available()
        found = load_dotenv_key(dotenv_path)
        status, env_var = redacted_key_status(), "ALPHAVANTAGE_API_KEY"
    else:
        # Keyless providers (yfinance) and injected test sources.
        return {"available": True, "source": "not required", "env_var": None,
                "status": f"{provider}: no credential required"}
    return {
        "available": bool(found),
        "source": "environment" if pre_existing else (".env" if found else "none"),
        "status": status,
        "env_var": env_var,
    }


def redacted_token_status_for_marketaux() -> str:
    from app.data.sources.marketaux_source import redacted_token_status

    return redacted_token_status()


def fetch_all(src, tickers, start=None, end=None, on_message=None):
    """Retrieve articles from ANY NewsSource, preserving per-ticker status.

    Uses the source's own `fetch_many` when it has one (Alpha Vantage pages and
    paces requests for its rate limit), and otherwise loops `fetch`, so the
    keyless YFinanceNewsSource and the frozen FNSPID archive work through the
    identical downstream path. A per-ticker failure is recorded as a status, not
    as an absence of news.
    """
    if hasattr(src, "fetch_many"):
        return src.fetch_many(tickers, start, end, on_message=on_message)
    frames, status = [], []
    for ticker in tickers:
        try:
            df = src.fetch(ticker, start, end)
        except Exception as exc:  # noqa: BLE001 - one bad ticker must not kill the run
            status.append({"ticker": ticker, "status": "error", "n_articles": 0,
                           "detail": f"{type(exc).__name__}: {exc}"})
            if on_message:
                on_message(f"[WARN] {ticker}: {type(exc).__name__}: {exc}")
            continue
        status.append({"ticker": ticker, "status": "ok", "n_articles": int(len(df)),
                       "detail": ""})
        if on_message:
            on_message(f"[OK] {ticker}: {len(df)} current articles")
        if len(df):
            frames.append(df)
    articles = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    return articles, status


def score_and_classify(articles: pd.DataFrame, scorer, sessions: pd.DatetimeIndex):
    """Attach session, FinBERT sentiment, and the 7-category label.

    Batched through `score_batch` when the scorer offers it (FinBERT does), which
    is one padded forward pass per batch instead of one per headline. Same model
    and same numbers either way -- only the grouping differs.
    """
    out = articles.copy()
    if out.empty:
        for col in ("trade_date", "trade_date_status", "sentiment_score",
                    "sentiment_label", "category"):
            out[col] = pd.Series(dtype=object)
        return out

    out["published_at"] = pd.to_datetime(out["published_at"], utc=True)
    out["trade_date"] = out["published_at"].map(
        lambda ts: first_tradable_session(ts, sessions)
    )
    out["trade_date_status"] = [
        "mapped" if pd.notna(td) else "beyond_price_panel" for td in out["trade_date"]
    ]

    headlines = out["headline"].fillna("").tolist()
    if hasattr(scorer, "score_batch"):
        details = scorer.score_batch(headlines)
        out["sentiment_score"] = [d["score"] for d in details]
        out["sentiment_label"] = [d["label"] for d in details]
    else:
        out["sentiment_score"] = [scorer.score(h) for h in headlines]
        out["sentiment_label"] = [
            "positive" if s > 0 else ("negative" if s < 0 else "neutral")
            for s in out["sentiment_score"]
        ]
    out["sentiment_score"] = out["sentiment_score"].clip(-1, 1)
    out["scorer"] = scorer.name
    out["category"] = classify_frame(out, "headline")
    return out


def collapse_to_events(scored: pd.DataFrame) -> pd.DataFrame:
    """One row per (ticker, event_date, category), n_articles = how many
    collapsed -- Phase 1's counting unit, reused verbatim so that an "event"
    here is the same object Phase 2's evidence was built from.

    Articles beyond the price panel have no session, so they are grouped by
    their calendar publication date instead and flagged; they are still one
    event per ticker-date-category, just not an attributable trading session.
    """
    if scored.empty:
        return pd.DataFrame(columns=[
            "ticker", "event_date", "category", "n_articles", "mean_sentiment",
            "trade_date_status", "first_published_at", "last_published_at", "headlines",
        ])
    df = scored.copy()
    df["event_date"] = [
        td if pd.notna(td) else pub.tz_convert(None).normalize()
        for td, pub in zip(df["trade_date"], df["published_at"], strict=True)
    ]
    grouped = (
        df.groupby(["ticker", "event_date", "category"], sort=True)
        .agg(
            n_articles=("article_id", "size"),
            mean_sentiment=("sentiment_score", "mean"),
            trade_date_status=("trade_date_status", "first"),
            first_published_at=("published_at", "min"),
            last_published_at=("published_at", "max"),
            headlines=("headline", lambda s: " | ".join(s.head(3))),
        )
        .reset_index()
    )
    return grouped


def attach_phase2_evidence(
    events: pd.DataFrame,
    phase2_summary: pd.DataFrame,
    phase2_vs_other: pd.DataFrame,
    horizons=HORIZONS,
) -> pd.DataFrame:
    """Cross every current event with Phase 2's evidence at each horizon.

    Left join on purpose: a (ticker, category) Phase 2 never had enough data to
    evaluate must appear as `no_history`, not vanish. A missing row is itself
    the finding most worth reporting.
    """
    if events.empty:
        return pd.DataFrame()
    summary_cols = [
        "ticker", "category", "horizon", "n_events_valid", "evidence_tier",
        "mean_abnormal_return", "nw_mean", "nw_ci95_lo", "nw_ci95_hi", "shrunk_mean",
        "sector", "sector_nw_mean",
    ]
    vs_cols = [
        "ticker", "category", "horizon", "n_events_category", "diff_mean", "se",
        "p_value", "ci95_lo", "ci95_hi", "significant_fdr", "sufficient_sample",
    ]
    s = phase2_summary[[c for c in summary_cols if c in phase2_summary.columns]].copy()
    v = phase2_vs_other[[c for c in vs_cols if c in phase2_vs_other.columns]].copy()
    v = v.rename(columns={
        "diff_mean": "vs_other_diff_mean", "se": "vs_other_se",
        "p_value": "vs_other_p_value", "ci95_lo": "vs_other_ci95_lo",
        "ci95_hi": "vs_other_ci95_hi", "sufficient_sample": "vs_other_sufficient",
    })

    base = events[["ticker", "category", "event_date", "n_articles", "mean_sentiment",
                   "trade_date_status", "headlines"]].copy()
    grid = base.merge(pd.DataFrame({"horizon": list(horizons)}), how="cross")
    out = grid.merge(s, on=["ticker", "category", "horizon"], how="left")
    out = out.merge(v, on=["ticker", "category", "horizon"], how="left")

    out["evidence_status"] = [
        _evidence_status(tier, sig) for tier, sig in
        zip(out.get("evidence_tier"), out.get("significant_fdr"), strict=True)
    ]
    out["evidence_is_actionable"] = out["evidence_status"] == "supported_and_significant"
    out["evidence_note"] = out["evidence_status"].map(EVIDENCE_STATUS)
    return out.sort_values(
        ["ticker", "event_date", "category", "horizon"]
    ).reset_index(drop=True)


def _evidence_status(tier, significant) -> str:
    """Map Phase 2's tier + FDR flag to a status.

    The NaN guard on `significant` is load-bearing, not defensive padding.
    After the left join, a missing vs-OTHER row leaves NaN -- and `bool(nan)`
    is True in Python, so an earlier version silently reported every such row
    as `supported_and_significant`. That produced 13 "statistically
    significant" rows in a validation run where Phase 2's entire output
    contains exactly 2, all of the spurious ones being OTHER-category events.
    OTHER is the comparison baseline, so it has no vs-OTHER test by
    construction; that absence must read as "no comparison", never as a
    positive result.
    """
    if tier is None or pd.isna(tier):
        return "no_history"
    if tier == "insufficient":
        return "insufficient_history"
    if tier == "exploratory":
        return "exploratory_history"
    if significant is None or pd.isna(significant):
        return "supported_no_comparison"
    return (
        "supported_and_significant" if bool(significant) else "supported_not_significant"
    )


def classifier_agreement(reference_path: str | Path) -> dict | None:
    """Measured comparability of the new labels with Phase 2's historical ones.

    Returned for the run record so every output carries the rate at which its
    category labels match the convention the evidence was built on.
    """
    from app.nlp.news_categories import agreement_report

    p = Path(reference_path)
    if not p.exists():
        return None
    ref = pd.read_parquet(p, columns=["headline", "category"])
    return agreement_report(ref)


def run(
    tickers: list[str] | None = None,
    panel_dir: str | Path = DEFAULT_PANEL_DIR,
    phase2_summary_path: str | Path = DEFAULT_PHASE2_SUMMARY,
    phase2_vs_other_path: str | Path = DEFAULT_PHASE2_VS_OTHER,
    out_dir: str | Path = DEFAULT_OUT_DIR,
    version: str = OUTPUT_VERSION,
    horizons: tuple[int, ...] = HORIZONS,
    scorer_name: str = "finbert",
    source=None,
    source_name: str = "alphavantage",
    provider_limit: int | None = None,
    start: str | None = None,
    end: str | None = None,
    reference_path: str | Path = (
        "data/diagnostics_EXPERIMENTAL/news_category_vs_return/"
        "classified_articles_EXPERIMENTAL.parquet"
    ),
    check_agreement: bool = True,
    on_message=print,
) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tickers = [t.strip().upper() for t in (tickers or [])]

    needs_credential = source is None and source_name in ("alphavantage", "marketaux")
    creds = resolve_credentials(source_name if source is None else "injected")
    if needs_credential and not creds["available"]:
        env_var = creds["env_var"]
        raise RuntimeError(
            f"No {source_name} credential is visible to the application.\n"
            f"Configure your EXISTING credential in either place:\n"
            f"  1. environment:  export {env_var}=<your-value>\n"
            f"     (PowerShell:  $env:{env_var} = '<your-value>')\n"
            f"  2. a .env file at the repository root containing:\n"
            f"     {env_var}=<your-value>\n"
            "     (.env is already git-ignored by this repository)\n"
            f"{env_var} is separate from the other providers' credentials; setting it "
            "does not affect them. It is read from the environment only, never logged."
        )
    if needs_credential:
        on_message(f"[INFO] credentials: {creds['status']} (source: {creds['source']})")

    if source is not None:
        src = source
    elif source_name == "yfinance":
        from app.data.sources.news_source import YFinanceNewsSource

        src = YFinanceNewsSource()
    elif source_name == "alphavantage":
        src = AlphaVantageNewsSource()
    elif source_name == "marketaux":
        from app.data.sources.marketaux_source import MarketauxNewsSource

        src = MarketauxNewsSource(limit=provider_limit or 3)
    else:
        raise ValueError(f"unknown news source '{source_name}'")
    scorer = get_scorer(scorer_name)
    on_message(f"[INFO] sentiment backend: {scorer.name} (requested: {scorer_name})")

    raw, fetch_status = fetch_all(src, tickers, start, end, on_message=on_message)
    raw = dedupe_news(raw) if len(raw) else raw

    panel = load_curated(panel_dir)
    sessions = pd.DatetimeIndex(
        pd.to_datetime(panel["date"]).dt.normalize().unique()
    ).sort_values()

    scored = score_and_classify(raw, scorer, sessions)
    events = collapse_to_events(scored)

    phase2_summary = pd.read_parquet(phase2_summary_path)
    phase2_vs_other = pd.read_parquet(phase2_vs_other_path)
    evidence = attach_phase2_evidence(events, phase2_summary, phase2_vs_other, horizons)

    agreement = classifier_agreement(reference_path) if check_agreement else None

    # Requirement: never create misleading empty current-news artifacts. With
    # zero articles there is nothing to analyse, and writing empty parquets
    # would leave files that LOOK like a completed retrieval -- and that a
    # later run, or a reader, could mistake for "we checked and there was no
    # news" when the truth may be "the provider refused the request". On an
    # empty result only the run record and report are written, and both say
    # why it was empty.
    articles_path = out_dir / f"current_news_articles_{version}.parquet"
    events_path = out_dir / f"current_news_events_{version}.parquet"
    evidence_path = out_dir / f"current_news_evidence_{version}.parquet"
    wrote_artifacts = bool(len(scored))
    if wrote_artifacts:
        scored.to_parquet(articles_path, index=False)
        events.to_parquet(events_path, index=False)
        if len(evidence):
            evidence.to_parquet(evidence_path, index=False)
    else:
        for stale in (articles_path, events_path, evidence_path):
            # An earlier run's artifacts must not survive alongside a report
            # saying this run retrieved nothing; that is exactly how cached
            # data gets read as fresh.
            if stale.exists():
                stale.unlink()

    status_counts = (
        evidence.groupby("evidence_status").size().to_dict() if len(evidence) else {}
    )
    summary = {
        "purpose": (
            "Current news retrieved from Alpha Vantage, scored with FinBERT, labelled "
            "with the reproducible 7-category classifier, and matched against Phase 2 "
            "stock-sensitivity evidence. Historical association only -- not a forecast."
        ),
        "credentials": {k: v for k, v in creds.items() if k != "status"} | {
            "status": creds["status"]
        },
        "source": getattr(src, "name", "injected"),
        "scorer": scorer.name,
        "retrieved_at": pd.Timestamp.now(tz="UTC").isoformat(),
        "freshness_note": (
            "`retrieved_at` is when this run queried the news provider. It describes "
            "the news retrieval ONLY -- it says nothing about the price panel, which "
            "is a separate and much older dataset. No cached artifact is relabelled "
            "as fresh, and on an empty result no article artifacts are written."
        ),
        "artifacts_written": wrote_artifacts,
        "tickers_requested": tickers,
        "fetch_status": fetch_status,
        "provider_limits": _provider_limit_summary(src, fetch_status),
        "n_articles_retrieved": int(len(scored)),
        "n_events": int(len(events)),
        "n_evidence_rows": int(len(evidence)),
        "category_counts": (
            scored["category"].value_counts().to_dict() if len(scored) else {}
        ),
        "trade_date_status_counts": (
            scored["trade_date_status"].value_counts().to_dict() if len(scored) else {}
        ),
        "last_known_session": (
            str(sessions.max().date()) if len(sessions) else None
        ),
        "forward_returns_computable": False,
        "forward_returns_note": (
            "No forward return is computed for current news. The curated price panel "
            f"ends {sessions.max().date() if len(sessions) else 'n/a'}, so articles "
            "published after that have no subsequent prices to measure and no trading "
            "session to attribute to. Outcome measurement requires refreshed price data."
        ),
        "evidence_status_counts": status_counts,
        "classifier_agreement_vs_historical": agreement,
        "horizons": list(horizons),
        "caveats": [
            "Phase 2 evidence is a HISTORICAL association, applied to a new article "
            "only under the assumption the relationship still holds -- Phase 3 found "
            "that assumption frequently unverifiable.",
            "Category labels agree with the historical labelling convention ~91% of "
            "the time, so matching new articles to Phase 2 evidence is approximate.",
            "Alpha Vantage's own sentiment is carried as av_* columns for reference "
            "but is NOT used; FinBERT is the scorer of record for both eras.",
            "Not a forecast, not investment advice, and not a claim of causation.",
        ],
        # None where nothing was written, so a consumer can never follow a
        # path to a file this run deliberately did not create.
        "outputs": {
            "articles": str(articles_path) if wrote_artifacts else None,
            "events": str(events_path) if wrote_artifacts else None,
            "evidence": (
                str(evidence_path) if (wrote_artifacts and len(evidence)) else None
            ),
        },
    }

    run_path = out_dir / f"current_news_run_{version}.json"
    run_path.write_text(json.dumps(summary, indent=1, default=str), encoding="utf-8")
    summary["run_path"] = str(run_path)

    report_path = out_dir / f"current_news_report_{version}.md"
    report_path.write_text(
        render_report(scored, events, evidence, summary), encoding="utf-8"
    )
    summary["report_path"] = str(report_path)
    return summary


def _provider_limit_summary(src, fetch_status) -> dict:
    """What the provider actually allowed versus what was asked for.

    Reported for every run so a small article count is never mistaken for a
    quiet news day when it is really a plan cap. Marketaux's free tier returns
    at most 3 articles per request while `found` can be in the hundreds, so the
    withheld count is the number that matters here.
    """
    out: dict = {
        "source": getattr(src, "name", "injected"),
        "requested_limit": getattr(src, "limit", None),
    }
    if getattr(src, "name", "") == "marketaux":
        from app.data.sources.marketaux_source import (
            FREE_PLAN_ARTICLE_LIMIT,
            FREE_PLAN_DAILY_REQUESTS,
        )

        out.update({
            "free_plan_article_limit_per_request": FREE_PLAN_ARTICLE_LIMIT,
            "free_plan_daily_requests": FREE_PLAN_DAILY_REQUESTS,
            "limits_verified_on": "2026-10-11 from marketaux.com/pricing",
        })
    metas = [s.get("meta") or {} for s in fetch_status]
    found = [m["provider_found"] for m in metas if m.get("provider_found") is not None]
    returned = [
        m["provider_returned"] for m in metas if m.get("provider_returned") is not None
    ]
    withheld = [m["withheld"] for m in metas if m.get("withheld") is not None]
    if found or returned:
        out.update({
            "total_matching_at_provider": sum(found) if found else None,
            "total_returned_by_provider": sum(returned) if returned else None,
            "total_withheld_by_plan": sum(withheld) if withheld else None,
            "any_request_capped_by_plan": any(
                m.get("plan_capped_the_request") for m in metas
            ),
            "per_ticker": [
                {k: m.get(k) for k in (
                    "ticker", "requested_limit", "provider_found", "provider_returned",
                    "provider_limit", "withheld", "truncated",
                )}
                for m in metas if m
            ],
        })
    return out


def _fmt_pct(x) -> str:
    return "n/a" if x is None or pd.isna(x) else f"{float(x) * 100:+.2f}%"


def render_report(
    scored: pd.DataFrame, events: pd.DataFrame, evidence: pd.DataFrame, summary: dict
) -> str:
    agreement = summary.get("classifier_agreement_vs_historical")
    lines = [
        "# Current News vs Phase 2 Historical Sensitivity",
        "",
        summary["purpose"],
        "",
        "## What this is not",
        "",
        "- **Not a forecast.** Every number below is what has historically followed "
        "this ticker+category, not what this article will do.",
        f"- **No forward return is computed.** {summary['forward_returns_note']}",
        "- **Not causal**, and not investment advice.",
        "",
        "## Run",
        "",
        f"- Source: `{summary['source']}` | sentiment: `{summary['scorer']}` "
        f"| credentials: {summary['credentials']['status']}",
        f"- Retrieved at: **{summary.get('retrieved_at', 'n/a')}** (news retrieval "
        "only -- see the price-panel note below)",
        f"- Tickers requested: {', '.join(summary['tickers_requested']) or 'none'}",
        f"- Articles retrieved: **{summary['n_articles_retrieved']}**, "
        f"collapsed to **{summary['n_events']}** (ticker, date, category) events",
        "",
    ]

    limits = summary.get("provider_limits") or {}
    if limits.get("free_plan_article_limit_per_request") or limits.get("per_ticker"):
        lines += ["## What the provider actually returned", ""]
        if limits.get("free_plan_article_limit_per_request"):
            lines.append(
                f"- Marketaux **free plan: "
                f"{limits['free_plan_article_limit_per_request']} articles per news "
                f"request, {limits['free_plan_daily_requests']} requests/day** "
                f"({limits['limits_verified_on']}). `limit` is accepted by the API but "
                "capped by the plan, so a larger request still returns at most "
                f"{limits['free_plan_article_limit_per_request']}."
            )
        if limits.get("requested_limit") is not None:
            lines.append(f"- Limit requested by this run: {limits['requested_limit']}")
        if limits.get("total_matching_at_provider") is not None:
            lines.append(
                f"- Provider matched **{limits['total_matching_at_provider']}** "
                f"articles, returned **{limits['total_returned_by_provider']}**, "
                f"withheld **{limits['total_withheld_by_plan']}** under the plan cap."
            )
        if limits.get("per_ticker"):
            lines += [
                "",
                "| ticker | requested | found | returned | plan limit | withheld |",
                "|---|---|---|---|---|---|",
            ]
            for m in limits["per_ticker"]:
                lines.append(
                    f"| {m.get('ticker')} | {m.get('requested_limit')} | "
                    f"{m.get('provider_found')} | {m.get('provider_returned')} | "
                    f"{m.get('provider_limit')} | {m.get('withheld')} |"
                )
        if limits.get("total_withheld_by_plan"):
            lines += [
                "",
                "> **These articles are a sample, not the full feed.** The counts above "
                "are what the plan allowed through, so the category mix below reflects "
                "whichever few articles were returned and should not be read as this "
                "ticker's complete news profile.",
            ]
        lines.append("")

    if summary["fetch_status"]:
        lines += ["| ticker | status | articles | detail |", "|---|---|---|---|"]
        for st in summary["fetch_status"]:
            detail = (st.get("detail") or "")[:70]
            lines.append(
                f"| {st['ticker']} | {st['status']} | {st['n_articles']} | {detail} |"
            )
        lines.append("")
        empty_rows = [s for s in summary["fetch_status"] if s["status"] == "empty"]
        if empty_rows:
            lines += [
                "> `empty`: the provider accepted the request and returned no articles "
                "for that ticker and window. That is a genuine empty result, distinct "
                "from the refusal statuses below.",
                "",
            ]
        failure_notes = {
            "auth_error": "the token was rejected (`invalid_api_token`)",
            "usage_limit": (
                "the plan's **daily** quota is exhausted (`usage_limit_reached`); it "
                "resets on a daily boundary, so retrying sooner cannot help"
            ),
            "rate_limited": (
                "too many requests in the past 60 seconds (`rate_limit_reached`); this "
                "one does clear within a minute"
            ),
            "plan_restricted": "the endpoint is not available on this plan",
            "provider_error": "a provider-side failure (validation, 5xx, or bad body)",
            "not_configured": "no credential was available",
        }
        seen = {s["status"] for s in summary["fetch_status"]}
        hit = [(k, v) for k, v in failure_notes.items() if k in seen]
        if hit:
            lines += [
                "> **Refused requests are not evidence of no news.** Those tickers are "
                "simply unmeasured in this run:",
                "",
            ]
            lines += [f"> - `{k}`: {v}" for k, v in hit]
            lines.append("")
        if any(s["status"] == "premium_required" for s in summary["fetch_status"]):
            lines += [
                "> **`premium_required`: Alpha Vantage's NEWS_SENTIMENT endpoint is not "
                "included in this account's plan.** The API key authenticated correctly "
                "-- this is a plan entitlement, not a credential problem, and retrying "
                "will not help. No articles were retrieved, which is **not** evidence "
                "that there is no news. Re-run with `--source yfinance` to use the "
                "keyless source through this identical pipeline.",
                "",
            ]
        if any(s["status"] == "rate_limited" for s in summary["fetch_status"]):
            lines += [
                "> A `rate_limited` row means Alpha Vantage refused the request, which "
                "is **not** the same as the ticker having no news. Those tickers are "
                "simply unmeasured in this run.",
                "",
            ]

    if agreement and agreement.get("overall_agreement") is not None:
        lines += [
            "## Are these labels comparable with Phase 2's?",
            "",
            f"The 7-category classifier agrees with the historical labelling convention "
            f"on **{agreement['overall_agreement'] * 100:.1f}%** of "
            f"{agreement['n']:,} historical articles. Phase 2's evidence is keyed on "
            "those historical labels, so every match below inherits this imprecision.",
            "",
            "| category | historical n | recall | precision |",
            "|---|---|---|---|",
        ]
        for p in agreement["per_category"]:
            lines.append(
                f"| {p['category']} | {p['n_reference']:,} | {p['recall']:.3f} | "
                f"{p['precision'] if p['precision'] is not None else 'n/a'} |"
            )
        lines += [
            "",
            f"> {agreement['note']}",
            "",
        ]

    if not len(scored):
        lines += [
            "## No articles retrieved",
            "",
            "**This run retrieved zero articles, so there is nothing to analyse.**",
            "",
            "- No article, event, or evidence artifacts were written. An empty parquet "
            "would look like a completed retrieval and could later be mistaken for "
            "\"we checked and there was no news\".",
            "- Any artifacts from a previous run were removed, so no stale file is left "
            "sitting next to this report where it could be read as current.",
            "- Check the per-ticker status table above: a refusal (auth, quota, rate "
            "limit, plan restriction) means the question was never answered, which is "
            "**not** the same as there being no news.",
            "",
        ]

    if len(scored):
        lines += ["## Retrieved articles", "",
                  "| published (UTC) | ticker | category | sentiment | session | headline |",
                  "|---|---|---|---|---|---|"]
        for _, r in scored.sort_values("published_at", ascending=False).head(40).iterrows():
            sess = (
                str(pd.Timestamp(r["trade_date"]).date())
                if pd.notna(r["trade_date"]) else "**beyond price panel**"
            )
            head = str(r["headline"])[:70].replace("|", "/")
            lines.append(
                f"| {pd.Timestamp(r['published_at']).strftime('%Y-%m-%d %H:%M')} | "
                f"{r['ticker']} | {r['category']} | {r['sentiment_score']:+.2f} "
                f"({r['sentiment_label']}) | {sess} | {head} |"
            )
        lines.append("")

    counts = summary.get("evidence_status_counts") or {}
    if counts:
        total = sum(counts.values())
        lines += [
            "## How much historical evidence backs these events?",
            "",
            "One row per (event x horizon). Read `supported_and_significant` as the only "
            "tier Phase 2 considered reliable.",
            "",
            "| evidence status | rows | share | meaning |",
            "|---|---|---|---|",
        ]
        for status in (
            "supported_and_significant", "supported_not_significant",
            "supported_no_comparison", "exploratory_history", "insufficient_history",
            "no_history",
        ):
            if status not in counts:
                continue
            n = counts[status]
            lines.append(
                f"| `{status}` | {n} | {n / total * 100:.1f}% | {EVIDENCE_STATUS[status]} |"
            )
        lines.append("")
        weak = sum(
            counts.get(k, 0)
            for k in ("exploratory_history", "insufficient_history", "no_history")
        )
        lines += [
            f"**{weak} of {total} event-horizon combinations ({weak / total * 100:.1f}%) "
            "rest on evidence Phase 2 itself judged too thin to rely on.** For those, the "
            "honest answer to \"what does history say?\" is: not enough to say anything.",
            "",
        ]

    if len(evidence):
        actionable = evidence[evidence["evidence_is_actionable"]]
        lines += ["## Events with adequately-supported, FDR-significant history", ""]
        if actionable.empty:
            lines += [
                "**None.** No current event matched a (ticker, category, horizon) cell "
                "that Phase 2 found both adequately sampled and statistically "
                "distinguishable from that stock's own other news. This is the expected "
                "outcome: Phase 2 found only 2 such cells across 447 tickers, 7 "
                "categories and 6 horizons, so a handful of current articles is unlikely "
                "to land on one. Absence of evidence here is not evidence of no effect.",
                "",
            ]
        else:
            lines += [
                "| ticker | category | horizon | event date | hist. mean | vs OTHER | "
                "95% CI | n events | tier |",
                "|---|---|---|---|---|---|---|---|---|",
            ]
            for _, r in actionable.iterrows():
                lines.append(
                    f"| {r['ticker']} | {r['category']} | {r['horizon']}d | "
                    f"{pd.Timestamp(r['event_date']).date()} | "
                    f"{_fmt_pct(r.get('mean_abnormal_return'))} | "
                    f"{_fmt_pct(r.get('vs_other_diff_mean'))} | "
                    f"[{_fmt_pct(r.get('vs_other_ci95_lo'))}, "
                    f"{_fmt_pct(r.get('vs_other_ci95_hi'))}] | "
                    f"{r.get('n_events_valid')} | {r.get('evidence_tier')} |"
                )
            lines.append("")

        lines += [
            "## Per-event detail",
            "",
            "| ticker | event date | category | articles | sentiment | horizon | "
            "hist. mean | n hist. events | evidence |",
            "|---|---|---|---|---|---|---|---|---|",
        ]
        for _, r in evidence.head(120).iterrows():
            lines.append(
                f"| {r['ticker']} | {pd.Timestamp(r['event_date']).date()} | "
                f"{r['category']} | {int(r['n_articles'])} | "
                f"{r['mean_sentiment']:+.2f} | {r['horizon']}d | "
                f"{_fmt_pct(r.get('mean_abnormal_return'))} | "
                f"{r.get('n_events_valid') if pd.notna(r.get('n_events_valid')) else 0} | "
                f"`{r['evidence_status']}` |"
            )
        lines.append("")

    lines += [
        "## Separate limitation: the price panel ends January 2024",
        "",
        "This is independent of which news provider is used, and refreshing the news "
        "feed does **not** fix it.",
        "",
        f"- The curated price panel's last session is "
        f"**{summary.get('last_known_session')}**. Phase 0's benchmark-adjusted "
        "returns, and therefore all Phase 1/2/3 evidence, stop there.",
        "- So a news article retrieved today can be scored and categorised, and it can "
        "be matched against historical evidence for its ticker+category, but **no "
        "forward return can be computed for it**. Articles past the last session are "
        "flagged `beyond_price_panel` rather than silently dropped.",
        "- The Phase 3 rolling results are historical and were computed on data ending "
        "2024-01-08. **They must not be presented as current.** Phase 3 found that at "
        "long horizons the most recent assessable window reaches only 2023-01-05, and "
        "that nothing about recent stability could be established for the two Phase 2 "
        "candidates. A fresh news feed does not make those findings current; only "
        "refreshed price data would.",
        "- Treating a current article's category as predictive on the strength of "
        "pre-2024 evidence assumes the relationship still holds. Phase 3 showed that "
        "assumption is frequently unverifiable.",
        "",
        "## Limitations",
        "",
    ] + [f"- {c}" for c in summary["caveats"]]
    lines += [
        "- Phase 0-3 outputs are read-only inputs here; this pipeline writes only "
        "`current_news_*` artifacts.",
        f"- Categories: {', '.join(CATEGORIES)}.",
    ]
    return "\n".join(lines)
