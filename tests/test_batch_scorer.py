"""Batch sentiment scoring: dedupe, cache, and article-level mapping.

Most tests use a fake scorer so the logic is exercised without loading
FinBERT. The tests that need the real model are gated on availability, and
the whole-artifact assertions skip when the artifact is absent.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from app.nlp.batch_scorer import (
    CACHE_COLUMNS,
    DEFAULT_BATCH_SIZE,
    FINBERT_MODEL,
    OUTPUT_COLUMNS,
    TEXT_COLUMN,
    load_cache,
    save_cache,
    score_news,
    score_texts,
    text_key,
)
from app.nlp.finbert_scorer import finbert_available

NEWS = Path("data/curated/news_fnspid.parquet")
SENTIMENT = Path("data/curated/news_sentiment.parquet")


class FakeScorer:
    """Deterministic stand-in that records exactly what it was asked to score."""

    name = "fake"

    def __init__(self):
        self.seen: list[str] = []
        self.calls = 0

    def score_batch(self, texts, batch_size: int = 32):
        self.calls += 1
        self.seen.extend(texts)
        out = []
        for t in texts:
            pos = (len(t) % 7) / 10.0
            neg = (len(t) % 5) / 10.0
            neu = max(0.0, 1.0 - pos - neg)
            total = pos + neg + neu
            pos, neg, neu = pos / total, neg / total, neu / total
            out.append(
                {
                    "positive": pos,
                    "negative": neg,
                    "neutral": neu,
                    "label": max(
                        ("positive", "negative", "neutral"),
                        key=lambda k: {"positive": pos, "negative": neg, "neutral": neu}[k],
                    ),
                    "score": pos - neg,
                }
            )
        return out


@pytest.fixture
def news() -> pd.DataFrame:
    """Six articles, three distinct headlines -- duplicates across tickers."""
    return pd.DataFrame(
        {
            "article_id": [f"a{i}" for i in range(6)],
            "headline": [
                "Market wrap: stocks climb",   # shared by 3 tickers
                "Market wrap: stocks climb",
                "Market wrap: stocks climb",
                "Company misses estimates",
                "Company misses estimates",
                "A unique headline about chips",
            ],
            "source": ["Wire"] * 6,
            "url": [f"https://example.com/{i}" for i in range(6)],
            "published_at": pd.to_datetime(["2020-01-02"] * 6, utc=True),
            "ticker": ["AAPL", "MSFT", "SPY", "AAPL", "NVDA", "NVDA"],
        }
    )


# --------------------------------------------------------------------------
# Keys
# --------------------------------------------------------------------------


def test_text_key_is_deterministic():
    assert text_key("Apple beats estimates") == text_key("Apple beats estimates")


def test_text_key_normalises_whitespace():
    assert text_key("Apple  beats\nestimates") == text_key("Apple beats estimates")


def test_text_key_separates_models():
    assert text_key("same text", model="a") != text_key("same text", model="b")


def test_text_key_handles_empty():
    assert isinstance(text_key(""), str)
    assert text_key(None) == text_key("")


# --------------------------------------------------------------------------
# Batch scoring mechanics
# --------------------------------------------------------------------------


def test_score_texts_returns_one_row_per_text():
    fake = FakeScorer()
    out = score_texts(["a", "bb", "ccc"], scorer=fake, batch_size=2)
    assert len(out) == 3
    assert set(out.columns) == {"positive", "negative", "neutral", "label", "score"}


def test_score_texts_actually_batches():
    fake = FakeScorer()
    score_texts([f"text {i}" for i in range(10)], scorer=fake, batch_size=4)
    assert fake.calls == 3, "should be ceil(10/4) batches, not 10 calls"


def test_score_texts_empty_input():
    assert len(score_texts([], scorer=FakeScorer())) == 0


def test_probabilities_sum_to_one():
    out = score_texts(["alpha", "beta", "gamma"], scorer=FakeScorer(), batch_size=2)
    total = out["positive"] + out["negative"] + out["neutral"]
    assert (total - 1.0).abs().max() < 1e-9


def test_sentiment_score_is_pos_minus_neg():
    out = score_texts(["alpha", "beta", "gamma"], scorer=FakeScorer(), batch_size=2)
    assert (out["score"] - (out["positive"] - out["negative"])).abs().max() < 1e-12


# --------------------------------------------------------------------------
# Deduplication and mapping
# --------------------------------------------------------------------------


def test_duplicate_text_scored_only_once(news, tmp_path):
    fake = FakeScorer()
    out, stats = score_news(news, scorer=fake, cache_path=tmp_path / "cache.parquet")
    assert stats["unique_texts"] == 3
    assert stats["texts_scored"] == 3
    assert len(fake.seen) == 3, "FinBERT must not see the same text twice"
    assert stats["duplicate_articles_saved"] == 3
    assert len(out) == 6


def test_every_article_record_is_preserved(news, tmp_path):
    out, _ = score_news(news, scorer=FakeScorer(), cache_path=tmp_path / "c.parquet")
    assert len(out) == len(news)
    assert set(out["article_id"]) == set(news["article_id"])
    assert list(out.columns) == OUTPUT_COLUMNS


def test_duplicates_receive_identical_scores(news, tmp_path):
    out, _ = score_news(news, scorer=FakeScorer(), cache_path=tmp_path / "c.parquet")
    wrap = out[out["article_id"].isin(["a0", "a1", "a2"])]
    assert wrap["score"].nunique() == 1, "same text must give the same score"
    assert wrap["label"].nunique() == 1


def test_results_map_back_to_the_right_article(news, tmp_path):
    out, _ = score_news(news, scorer=FakeScorer(), cache_path=tmp_path / "c.parquet")
    joined = news.merge(out, on="article_id", suffixes=("", "_s"))
    # The unique headline must not share a score with the wrap headline.
    unique_row = joined[joined["article_id"] == "a5"]
    wrap_row = joined[joined["article_id"] == "a0"]
    assert unique_row["score"].iloc[0] != wrap_row["score"].iloc[0]
    # Ticker and timestamp survive the round trip.
    assert unique_row["ticker_s"].iloc[0] == "NVDA"
    assert set(out["ticker"]) == set(news["ticker"])


def test_output_carries_join_keys(news, tmp_path):
    out, _ = score_news(news, scorer=FakeScorer(), cache_path=tmp_path / "c.parquet")
    for col in ("article_id", "ticker", "published_at", "text_key"):
        assert col in out.columns
    assert out["published_at"].notna().all()


def test_missing_required_columns_raise(news, tmp_path):
    with pytest.raises(KeyError, match="headline"):
        score_news(news.drop(columns=["headline"]), scorer=FakeScorer(), cache_path=None)
    with pytest.raises(KeyError, match="ticker"):
        score_news(news.drop(columns=["ticker"]), scorer=FakeScorer(), cache_path=None)


# --------------------------------------------------------------------------
# Caching
# --------------------------------------------------------------------------


def test_cache_is_written(news, tmp_path):
    cache_path = tmp_path / "cache.parquet"
    score_news(news, scorer=FakeScorer(), cache_path=cache_path)
    assert cache_path.exists()
    cache = load_cache(cache_path)
    assert len(cache) == 3
    assert list(cache.columns) == CACHE_COLUMNS


def test_rerun_reuses_cache_without_calling_the_model(news, tmp_path):
    cache_path = tmp_path / "cache.parquet"
    first = FakeScorer()
    score_news(news, scorer=first, cache_path=cache_path)
    assert len(first.seen) == 3

    second = FakeScorer()
    out, stats = score_news(news, scorer=second, cache_path=cache_path)
    assert second.seen == [], "FinBERT was re-run despite a warm cache"
    assert second.calls == 0
    assert stats["texts_scored"] == 0
    assert stats["texts_reused_from_cache"] == 3
    assert len(out) == 6


def test_rerun_produces_identical_output(news, tmp_path):
    cache_path = tmp_path / "cache.parquet"
    a, _ = score_news(news, scorer=FakeScorer(), cache_path=cache_path)
    b, _ = score_news(news, scorer=FakeScorer(), cache_path=cache_path)
    pd.testing.assert_frame_equal(a, b)


def test_partial_cache_scores_only_the_new_text(news, tmp_path):
    cache_path = tmp_path / "cache.parquet"
    score_news(news.iloc[:3], scorer=FakeScorer(), cache_path=cache_path)

    later = FakeScorer()
    _, stats = score_news(news, scorer=later, cache_path=cache_path)
    assert stats["texts_reused_from_cache"] == 1
    assert stats["texts_scored"] == 2
    assert len(later.seen) == 2


def test_cache_from_a_different_model_is_not_reused(news, tmp_path):
    cache_path = tmp_path / "cache.parquet"
    score_news(news, scorer=FakeScorer(), model="model-a", cache_path=cache_path)
    later = FakeScorer()
    _, stats = score_news(news, scorer=later, model="model-b", cache_path=cache_path)
    assert stats["texts_scored"] == 3, "another model's scores must not be reused"


def test_no_cache_path_still_works(news):
    out, stats = score_news(news, scorer=FakeScorer(), cache_path=None)
    assert len(out) == 6
    assert stats["texts_scored"] == 3


def test_corrupt_cache_is_rejected(tmp_path):
    bad = tmp_path / "bad.parquet"
    pd.DataFrame({"text_key": ["x"]}).to_parquet(bad, index=False)
    with pytest.raises(ValueError, match="missing columns"):
        load_cache(bad)


def test_save_cache_deduplicates(tmp_path):
    dup = pd.DataFrame(
        {c: ["k1", FINBERT_MODEL, 0.5, 0.2, 0.3, "positive", 0.3] for c in [0]}
    )
    frame = pd.DataFrame(
        [
            ["k1", FINBERT_MODEL, 0.5, 0.2, 0.3, "positive", 0.3],
            ["k1", FINBERT_MODEL, 0.5, 0.2, 0.3, "positive", 0.3],
        ],
        columns=CACHE_COLUMNS,
    )
    del dup
    path = save_cache(frame, tmp_path / "c.parquet")
    assert len(load_cache(path)) == 1


# --------------------------------------------------------------------------
# Real FinBERT (small slice)
# --------------------------------------------------------------------------


@pytest.mark.skipif(not finbert_available(), reason="FinBERT not installed")
def test_real_finbert_batch_matches_single_scoring():
    """Batching must not change the numbers."""
    from app.nlp.finbert_scorer import get_scorer

    scorer = get_scorer("finbert")
    texts = [
        "Company beats earnings estimates and raises guidance",
        "Company misses estimates and warns of falling profits",
        "Earnings Scheduled For January 21, 2014",
    ]
    batched = scorer.score_batch(texts, batch_size=DEFAULT_BATCH_SIZE)
    for text, b in zip(texts, batched, strict=True):
        single = scorer.score_detail(text)
        assert b["label"] == single["label"]
        for key in ("positive", "negative", "neutral", "score"):
            assert b[key] == pytest.approx(single[key], abs=1e-5)


@pytest.mark.skipif(not finbert_available(), reason="FinBERT not installed")
def test_real_finbert_probabilities_sum_to_one():
    from app.nlp.finbert_scorer import get_scorer

    out = score_texts(
        ["Stocks rally on strong data", "Shares plunge after warning"],
        scorer=get_scorer("finbert"),
        batch_size=8,
    )
    total = out["positive"] + out["negative"] + out["neutral"]
    assert (total - 1.0).abs().max() < 1e-5


# --------------------------------------------------------------------------
# The produced artifact
# --------------------------------------------------------------------------


@pytest.mark.skipif(not SENTIMENT.exists(), reason="sentiment artifact not built")
def test_artifact_preserves_every_article():
    news = pd.read_parquet(NEWS)
    sent = pd.read_parquet(SENTIMENT)
    assert len(sent) == len(news) == 57111
    assert set(sent["article_id"]) == set(news["article_id"])
    assert sent["article_id"].is_unique


@pytest.mark.skipif(not SENTIMENT.exists(), reason="sentiment artifact not built")
def test_artifact_is_well_formed():
    sent = pd.read_parquet(SENTIMENT)
    assert list(sent.columns) == OUTPUT_COLUMNS
    total = sent["positive"] + sent["negative"] + sent["neutral"]
    assert (total - 1.0).abs().max() < 1e-5
    assert (sent["score"] - (sent["positive"] - sent["negative"])).abs().max() < 1e-9
    assert set(sent["label"]) <= {"positive", "negative", "neutral"}
    assert sent[["positive", "negative", "neutral"]].notna().all().all()
    assert sent["score"].between(-1, 1).all()


@pytest.mark.skipif(not SENTIMENT.exists(), reason="sentiment artifact not built")
def test_artifact_not_yet_aggregated():
    """Requirement: keep article-level records; do not roll up to ticker-day."""
    sent = pd.read_parquet(SENTIMENT)
    per_ticker_day = sent.groupby(["ticker", sent["published_at"].dt.date]).size()
    assert (per_ticker_day > 1).any(), "looks aggregated to one row per ticker-day"


@pytest.mark.skipif(not SENTIMENT.exists(), reason="sentiment artifact not built")
def test_duplicate_headlines_share_scores_in_the_artifact():
    news = pd.read_parquet(NEWS)
    sent = pd.read_parquet(SENTIMENT)
    merged = news[["article_id", TEXT_COLUMN]].merge(sent, on="article_id")
    grouped = merged.groupby("text_key")["score"].nunique()
    assert (grouped == 1).all(), "identical text produced differing scores"
