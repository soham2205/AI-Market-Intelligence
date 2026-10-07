"""Batch FinBERT scoring of curated historical news.

    news_fnspid.parquet  ->  dedupe text  ->  FinBERT (batched)  ->  cache
                                                                      |
                         article-level sentiment  <-------------------+

Design notes
------------
Text selection: the HEADLINE only. The curated artifact carries no article
body -- the FNSPID extractor kept `Article_title` and dropped `Article` and
the four summary columns, because streaming 23 GB with bodies attached was
the cost being avoided. That is also the right choice on the merits: at
inference time the live feed yields headlines and nothing else, so training
on headline+body would build in train/inference skew.

Deduplication: 12,211 of 57,111 headlines repeat (market wraps syndicated
across tickers). Identical text gets identical model output, so unique text
is scored once and the result is joined back to every article that shares it.
Every article-level record is preserved.

Caching: results are keyed by a hash of the text plus the model identity, so
a rerun recomputes nothing, and switching models cannot silently reuse the
previous model's numbers.
"""

from __future__ import annotations

import hashlib
import time
from collections.abc import Sequence
from pathlib import Path

import pandas as pd

FINBERT_MODEL = "ProsusAI/finbert"

# Text column consumed from the curated news schema.
TEXT_COLUMN = "headline"

# Measured on this CPU over 512 real headlines:
#
#   batch    len32   len64   len128        headline token lengths:
#       8     11.3     9.1      9.1          p50 16, p90 23, p99 50, max 70
#      16     12.6     9.8      9.7
#      32     12.6     9.3      9.5
#
# max_length 64 leaves 99.98% of headlines untruncated (32 would truncate
# 4.35% for ~28% more speed -- a bad trade for a one-off job). Throughput
# plateaus by batch 16-32; larger batches buy nothing on CPU because the work
# is compute-bound, and padding is dynamic (to the longest item in the batch),
# so max_length only governs truncation.
DEFAULT_BATCH_SIZE = 32
DEFAULT_MAX_LENGTH = 64

DEFAULT_CACHE_PATH = Path("data/curated/sentiment_cache.parquet")
DEFAULT_OUTPUT_PATH = Path("data/curated/news_sentiment.parquet")

CACHE_COLUMNS = ["text_key", "model", "positive", "negative", "neutral", "label", "score"]
OUTPUT_COLUMNS = [
    "article_id",
    "ticker",
    "published_at",
    "text_key",
    "positive",
    "negative",
    "neutral",
    "label",
    "score",
]


def text_key(text: str, model: str = FINBERT_MODEL) -> str:
    """Deterministic key for a piece of text under a given model.

    The model is part of the key so a cache built with one model can never be
    mistaken for another's output.
    """
    normalised = " ".join(str(text or "").split())
    return hashlib.sha1(f"{model}|{normalised}".encode()).hexdigest()[:20]


def load_cache(path: str | Path = DEFAULT_CACHE_PATH) -> pd.DataFrame:
    """Previously scored text. Empty frame when no cache exists yet."""
    path = Path(path)
    if not path.exists():
        return pd.DataFrame(columns=CACHE_COLUMNS)
    cached = pd.read_parquet(path)
    missing = [c for c in CACHE_COLUMNS if c not in cached.columns]
    if missing:
        raise ValueError(f"sentiment cache {path} is missing columns {missing}")
    return cached


def save_cache(cache: pd.DataFrame, path: str | Path = DEFAULT_CACHE_PATH) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    cache[CACHE_COLUMNS].drop_duplicates(subset=["text_key"]).to_parquet(path, index=False)
    return path


def score_texts(
    texts: Sequence[str],
    scorer=None,
    batch_size: int = DEFAULT_BATCH_SIZE,
    progress_every: int = 50,
) -> pd.DataFrame:
    """Run FinBERT over a list of texts, in batches. No caching, no dedupe."""
    texts = list(texts)
    if not texts:
        return pd.DataFrame(columns=["positive", "negative", "neutral", "label", "score"])
    if scorer is None:
        from app.nlp.finbert_scorer import FinBertScorer

        scorer = FinBertScorer(max_length=DEFAULT_MAX_LENGTH)

    rows: list[dict] = []
    started = time.time()
    n_batches = (len(texts) + batch_size - 1) // batch_size
    for i in range(0, len(texts), batch_size):
        chunk = texts[i : i + batch_size]
        rows.extend(scorer.score_batch(chunk, batch_size=batch_size))
        batch_no = i // batch_size
        if progress_every and batch_no % progress_every == 0 and batch_no:
            done = len(rows)
            rate = done / max(time.time() - started, 1e-9)
            print(
                f"     batch {batch_no:>5}/{n_batches}  scored {done:>7,}  "
                f"{rate:6.1f} texts/s",
                flush=True,
            )
    return pd.DataFrame(rows)


def score_news(
    news: pd.DataFrame,
    scorer=None,
    model: str = FINBERT_MODEL,
    batch_size: int = DEFAULT_BATCH_SIZE,
    cache_path: str | Path | None = DEFAULT_CACHE_PATH,
    text_column: str = TEXT_COLUMN,
    progress_every: int = 50,
) -> tuple[pd.DataFrame, dict]:
    """Score every article, reusing cached results for repeated text.

    Returns (article-level sentiment, stats). The output has exactly one row
    per input article -- deduplication happens only at the inference step and
    is joined back afterwards.
    """
    if text_column not in news.columns:
        raise KeyError(f"news frame has no '{text_column}' column")
    for required in ("article_id", "ticker", "published_at"):
        if required not in news.columns:
            raise KeyError(f"news frame has no '{required}' column")

    work = news.copy()
    work["text_key"] = [text_key(t, model) for t in work[text_column]]

    unique_texts = (
        work.drop_duplicates(subset=["text_key"])[["text_key", text_column]]
        .reset_index(drop=True)
    )

    cache = load_cache(cache_path) if cache_path is not None else pd.DataFrame(
        columns=CACHE_COLUMNS
    )
    cache = cache[cache["model"] == model] if len(cache) else cache
    known = set(cache["text_key"]) if len(cache) else set()

    todo = unique_texts[~unique_texts["text_key"].isin(known)]

    started = time.time()
    if len(todo):
        scored = score_texts(
            todo[text_column].tolist(),
            scorer=scorer,
            batch_size=batch_size,
            progress_every=progress_every,
        )
        scored.insert(0, "model", model)
        scored.insert(0, "text_key", todo["text_key"].to_numpy())
        fresh = scored[CACHE_COLUMNS]
        # Avoid concatenating an all-empty frame: pandas warns, and the dtypes
        # of the empty placeholder would otherwise leak into the result.
        cache = fresh if cache.empty else pd.concat([cache, fresh], ignore_index=True)
        if cache_path is not None:
            save_cache(cache, cache_path)
    elapsed = time.time() - started

    # Join back onto EVERY article, including the duplicates.
    merged = work.merge(
        cache[["text_key", "positive", "negative", "neutral", "label", "score"]],
        on="text_key",
        how="left",
        validate="many_to_one",
    )
    if len(merged) != len(news):
        raise AssertionError(
            f"article count changed during scoring: {len(news)} -> {len(merged)}"
        )
    if merged["score"].isna().any():
        raise AssertionError("some articles did not receive a sentiment result")

    out = merged[OUTPUT_COLUMNS].reset_index(drop=True)
    stats = {
        "articles_in": len(news),
        "articles_out": len(out),
        "unique_texts": len(unique_texts),
        "texts_scored": len(todo),
        "texts_reused_from_cache": len(unique_texts) - len(todo),
        "duplicate_articles_saved": len(news) - len(unique_texts),
        "elapsed_s": elapsed,
        "texts_per_s": (len(todo) / elapsed) if elapsed > 0 and len(todo) else 0.0,
        "model": model,
        "batch_size": batch_size,
        "cache_size": len(cache),
    }
    return out, stats


def run(
    news_path: str | Path = "data/curated/news_fnspid.parquet",
    output_path: str | Path = DEFAULT_OUTPUT_PATH,
    cache_path: str | Path = DEFAULT_CACHE_PATH,
    batch_size: int = DEFAULT_BATCH_SIZE,
    scorer=None,
) -> dict:
    """Score the curated news artifact end to end and persist the result."""
    news = pd.read_parquet(news_path)
    out, stats = score_news(
        news, scorer=scorer, batch_size=batch_size, cache_path=cache_path
    )
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(output_path, index=False)
    stats["output_path"] = str(output_path)
    stats["output_bytes"] = output_path.stat().st_size
    stats["cache_path"] = str(cache_path)
    return stats
