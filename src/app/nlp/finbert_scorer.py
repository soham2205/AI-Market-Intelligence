from __future__ import annotations

import re
from collections.abc import Sequence

FIN_POS = {
    "beat", "beats", "beating", "record", "growth", "surge", "surges", "surged",
    "gain", "gains", "rally", "rallies", "upgrade", "upgraded", "outperform",
    "strong", "strength", "profit", "profits", "rose", "rises", "rising", "jump",
    "jumps", "soar", "soars", "tops", "exceeds", "bullish", "optimistic", "buy",
    "dividend", "expansion", "breakthrough", "approval", "wins", "success",
}
FIN_NEG = {
    "miss", "misses", "missed", "drop", "drops", "dropped", "fall", "falls",
    "fell", "plunge", "plunges", "slump", "slumps", "downgrade", "downgraded",
    "underperform", "weak", "weakness", "loss", "losses", "decline", "declines",
    "bearish", "pessimistic", "sell", "lawsuit", "probe", "investigation",
    "recall", "bankruptcy", "cut", "cuts", "warns", "warning", "fraud", "layoffs",
}

_WORD_RE = re.compile(r"[a-z]+")


class LexiconScorer:
    """Deterministic finance lexicon scorer. Fallback when FinBERT is absent."""

    name = "lexicon"

    def score(self, text: str) -> float:
        words = _WORD_RE.findall((text or "").lower())
        if not words:
            return 0.0
        pos = sum(w in FIN_POS for w in words)
        neg = sum(w in FIN_NEG for w in words)
        if pos + neg == 0:
            return 0.0
        return (pos - neg) / (pos + neg)


class FinBertScorer:
    """FinBERT sentiment via HuggingFace transformers (lazy import)."""

    name = "finbert"

    def __init__(self, model_name: str = "ProsusAI/finbert", max_length: int = 256):
        from transformers import pipeline

        self.model_name = model_name
        self.max_length = max_length
        self._pipe = pipeline(
            "text-classification",
            model=model_name,
            truncation=True,
            max_length=max_length,
        )

    LABELS = ("positive", "negative", "neutral")

    def score(self, text: str) -> float:
        """Net sentiment in [-1, 1]: P(positive) - P(negative)."""
        return self.score_detail(text)["score"]

    def score_batch(self, texts: Sequence[str], batch_size: int = 64) -> list[dict]:
        """Score many texts in batches.

        One padded forward pass per batch instead of one per headline. Same
        model, same weights, same numbers as `score_detail` -- only the
        grouping differs.
        """
        texts = list(texts)
        if not texts:
            return []
        raw = self._pipe(
            [t[:512] for t in texts], top_k=None, batch_size=batch_size
        )
        return [self._to_detail(r) for r in raw]

    def _to_detail(self, result: list[dict]) -> dict:
        probs = {r["label"].lower(): float(r["score"]) for r in result}
        pos = probs.get("positive", 0.0)
        neg = probs.get("negative", 0.0)
        neu = probs.get("neutral", 0.0)
        return {
            "positive": pos,
            "negative": neg,
            "neutral": neu,
            "label": max(self.LABELS, key=lambda k: probs.get(k, 0.0)),
            "score": pos - neg,
        }

    def score_detail(self, text: str) -> dict:
        """Full class distribution for one headline.

        Returns the three class probabilities, the arg-max label, and the net
        score. The probabilities come straight from the model's softmax, so
        they sum to 1 up to float error.
        """
        return self._to_detail(self._pipe(text[:512], top_k=None))


FINBERT_REQUIREMENTS = ["transformers", "tokenizers", "safetensors", "huggingface_hub"]


def missing_finbert_requirements() -> list[str]:
    """Which FinBERT dependencies are absent. Empty list means available."""
    import importlib.util

    return [m for m in FINBERT_REQUIREMENTS if importlib.util.find_spec(m) is None]


def finbert_available() -> bool:
    return not missing_finbert_requirements()


def get_scorer(backend: str = "auto"):
    """Return a sentiment scorer.

    backend="finbert" is STRICT: it raises if FinBERT cannot be loaded. The
    research run must know for certain which model produced its scores, so a
    silent downgrade to the lexicon is not acceptable there.

    backend="auto" keeps the original forgiving behaviour for the live
    yfinance path, where a lexicon score beats no score at all.
    """
    if backend == "lexicon":
        return LexiconScorer()
    if backend == "finbert":
        missing = missing_finbert_requirements()
        if missing:
            raise RuntimeError(
                "FinBERT requested but unavailable. Missing packages: "
                + ", ".join(missing)
                + ". Install with: pip install transformers"
            )
        return FinBertScorer()
    if backend != "auto":
        raise ValueError(f"unknown sentiment backend '{backend}'")
    try:
        return FinBertScorer()
    except Exception:
        return LexiconScorer()
