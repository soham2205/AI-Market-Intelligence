"""FinBERT smoke tests: the model loads locally and behaves sanely.

Deliberately small -- a handful of real FNSPID headlines, not the corpus.
Skipped when FinBERT is not installed so the suite still runs on a machine
without the ~438 MB model.
"""

from __future__ import annotations

import time
from pathlib import Path

import pandas as pd
import pytest

from app.nlp.finbert_scorer import (
    FinBertScorer,
    LexiconScorer,
    finbert_available,
    get_scorer,
    missing_finbert_requirements,
)

pytestmark = pytest.mark.skipif(
    not finbert_available(), reason="FinBERT dependencies not installed"
)

NEWS = Path("data/curated/news_fnspid.parquet")
LABELS = {"positive", "negative", "neutral"}

# Real FNSPID headlines, kept inline so the tests do not depend on the
# extracted artifact being present.
HEADLINES = [
    "Nvidia Goes Negative (NVDA)",
    "UPDATE: Goldman Sachs Reiterates Buy on Johnson & Johnson",
    "J&J CEO Expects More Consolidation Among Hospitals",
    "J&J Discloses Government Investigations Related to False Claims",
    "Earnings Scheduled For January 21, 2014",
    "Deutsche Bank Downgrades Johnson & Johnson to Hold",
    "Knee Replacements Have Doubled -Bloomberg",
    "Qualcomm CEO Blames Nvidia for Delaying Debut of Android Tablets",
    "Non-Streak Dividend ETFs Impress",
    "Option Alert: Nvidia January 14 Call; Block Trade",
]


@pytest.fixture(scope="module")
def scorer() -> FinBertScorer:
    """Loaded once -- instantiating FinBERT is the expensive part."""
    return get_scorer("finbert")


# --------------------------------------------------------------------------
# Availability and wiring
# --------------------------------------------------------------------------


def test_dependencies_present():
    assert missing_finbert_requirements() == []
    assert finbert_available()


def test_strict_backend_returns_finbert_not_lexicon(scorer):
    """The whole point of strict mode: no silent downgrade."""
    assert isinstance(scorer, FinBertScorer)
    assert not isinstance(scorer, LexiconScorer)
    assert scorer.name == "finbert"


def test_auto_backend_now_resolves_to_finbert():
    assert get_scorer("auto").name == "finbert"


def test_lexicon_backend_still_available():
    """The live path must keep its existing behaviour."""
    assert get_scorer("lexicon").name == "lexicon"


def test_model_runs_on_cpu(scorer):
    import torch

    assert not torch.cuda.is_available(), "environment is expected to be CPU-only"
    assert scorer.score("Apple beats quarterly estimates") is not None


# --------------------------------------------------------------------------
# Output contract
# --------------------------------------------------------------------------


@pytest.mark.parametrize("headline", HEADLINES)
def test_probabilities_are_valid(scorer, headline):
    import math

    d = scorer.score_detail(headline)
    for key in ("positive", "negative", "neutral"):
        assert math.isfinite(d[key]), f"{key} not finite"
        assert 0.0 <= d[key] <= 1.0
    total = d["positive"] + d["negative"] + d["neutral"]
    assert total == pytest.approx(1.0, abs=1e-5), f"probabilities sum to {total}"


@pytest.mark.parametrize("headline", HEADLINES)
def test_label_and_score_contract(scorer, headline):
    d = scorer.score_detail(headline)
    assert d["label"] in LABELS
    assert d["score"] == pytest.approx(d["positive"] - d["negative"], abs=1e-9)
    assert -1.0 <= d["score"] <= 1.0
    # The reported label must be the arg-max of the three probabilities.
    best = max(("positive", "negative", "neutral"), key=lambda k: d[k])
    assert d["label"] == best


def test_score_matches_score_detail(scorer):
    for headline in HEADLINES[:3]:
        assert scorer.score(headline) == pytest.approx(
            scorer.score_detail(headline)["score"], abs=1e-9
        )


def test_scores_are_clipped_to_unit_range(scorer):
    for headline in HEADLINES:
        assert -1.0 <= scorer.score(headline) <= 1.0


# --------------------------------------------------------------------------
# Determinism
# --------------------------------------------------------------------------


def test_repeated_inference_is_deterministic(scorer):
    first = [scorer.score_detail(h) for h in HEADLINES]
    second = [scorer.score_detail(h) for h in HEADLINES]
    for a, b in zip(first, second, strict=True):
        assert a["label"] == b["label"]
        for key in ("positive", "negative", "neutral", "score"):
            assert a[key] == pytest.approx(b[key], abs=1e-9)


# --------------------------------------------------------------------------
# Behaviour on obvious cases
# --------------------------------------------------------------------------


def test_directionally_sensible_on_clear_headlines(scorer):
    """Not an accuracy claim -- just that the model is wired up correctly and
    is not returning constant output."""
    bullish = scorer.score("Company beats earnings estimates and raises guidance")
    bearish = scorer.score("Company misses estimates and warns of falling profits")
    assert bullish > bearish
    assert bullish > 0 > bearish


def test_output_varies_across_headlines(scorer):
    scores = [scorer.score(h) for h in HEADLINES]
    assert len(set(round(s, 4) for s in scores)) > 3, "model output looks constant"


def test_empty_and_short_text_do_not_crash(scorer):
    for text in ["", " ", "N/A"]:
        d = scorer.score_detail(text)
        assert d["label"] in LABELS
        assert d["positive"] + d["negative"] + d["neutral"] == pytest.approx(1.0, abs=1e-5)


# --------------------------------------------------------------------------
# Against the extracted FNSPID artifact (skips when absent)
# --------------------------------------------------------------------------


@pytest.mark.skipif(not NEWS.exists(), reason="FNSPID news artifact not extracted")
def test_smoke_over_real_fnspid_headlines(scorer):
    news = pd.read_parquet(NEWS).drop_duplicates(subset=["headline"]).head(12)
    t0 = time.time()
    results = [scorer.score_detail(h) for h in news["headline"]]
    elapsed = time.time() - t0

    assert len(results) == len(news)
    for d in results:
        assert d["label"] in LABELS
        assert d["positive"] + d["negative"] + d["neutral"] == pytest.approx(1.0, abs=1e-5)
    # Loose bound: catches a pathological regression, not a benchmark.
    assert elapsed < 120, f"{len(news)} headlines took {elapsed:.1f}s"
