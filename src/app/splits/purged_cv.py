from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class Fold:
    fold: int
    train_idx: np.ndarray
    val_idx: np.ndarray


def purge_gap(horizon: int = 1, embargo_days: int = 10) -> int:
    """Date-rank separation required between train and validation blocks.

    Two DISTINCT mechanisms, deliberately not conflated:

    purge = `horizon` (MANDATORY, correctness)
        Row T's label spans [T, T+k] because it needs Close_{T+k}. A training
        row within k ranks of val_start therefore has a label built from
        validation-period prices -- direct leakage. The required separation
        is exactly k, so it MUST be derived from the horizon; any constant is
        wrong at every horizon but one.

    embargo = `embargo_days` (OPTIONAL, hygiene)
        Validation rows near the boundary have FEATURE windows reaching back
        into the training period. That does not move future information into
        training -- training legitimately saw those sessions -- it only makes
        adjacent train/val samples serially correlated, which biases the
        validation estimate optimistic. It is NOT a leakage control and does
        NOT scale with the horizon.
    """
    if horizon < 1:
        raise ValueError(f"horizon must be >= 1, got {horizon}")
    if embargo_days < 0:
        raise ValueError(f"embargo_days must be >= 0, got {embargo_days}")
    return horizon + embargo_days


def date_positions(dates: pd.Series) -> np.ndarray:
    """Map each row's date to its rank among sorted unique dates."""
    unique = np.sort(dates.unique())
    lookup = pd.Series(np.arange(len(unique)), index=unique)
    return lookup.loc[dates].to_numpy()


def _geometry(
    n_unique_dates: int,
    n_folds: int,
    horizon: int,
    embargo_days: int,
) -> tuple[int, int, int]:
    """Return (block, gap, test_start_rank), validating feasibility."""
    if n_folds < 1:
        raise ValueError("n_folds must be >= 1")
    block = n_unique_dates // (n_folds + 2)
    gap = purge_gap(horizon, embargo_days)
    if block <= gap:
        raise ValueError(
            f"horizon={horizon}: block of {block} sessions cannot absorb a purge "
            f"gap of {gap} (= horizon {horizon} + embargo {embargo_days}). "
            f"Need more history or fewer folds; folds are never reduced silently."
        )
    test_start = n_unique_dates - block
    last_val_end = test_start - gap
    if last_val_end <= n_folds * block:
        raise ValueError(
            f"horizon={horizon}: after purging {gap} sessions before the test "
            f"block, the final validation fold would be empty "
            f"(val_start={n_folds * block}, val_end={last_val_end})."
        )
    return block, gap, test_start


def make_split_plan(
    n_unique_dates: int,
    n_folds: int,
    embargo_days: int,
    horizon: int = 1,
) -> tuple[list[Fold], tuple[int, int]]:
    """Purged expanding-window walk-forward over date positions.

    Unique dates are split into contiguous blocks. Fold i trains on
    [0, i*block - gap) and validates on block i. The final block is the
    held-out test range, and the last validation fold stops `gap` ranks
    short of it so no validation label reaches into the test period.
    """
    block, gap, test_start = _geometry(n_unique_dates, n_folds, horizon, embargo_days)

    folds: list[Fold] = []
    for i in range(1, n_folds + 1):
        train_end = i * block - gap
        val_start = i * block
        val_end = (i + 1) * block if i < n_folds else test_start - gap
        folds.append(
            Fold(
                fold=i,
                train_idx=np.arange(0, max(train_end, 1)),
                val_idx=np.arange(val_start, val_end),
            )
        )
    return folds, (test_start, n_unique_dates)


def split_frame(
    df: pd.DataFrame,
    n_folds: int,
    embargo_days: int,
    horizon: int = 1,
) -> tuple[list[Fold], np.ndarray, np.ndarray]:
    """Positional folds, test index, and final-fit mask for a sorted frame.

    Boundaries are computed in date-rank space and converted to row masks,
    so every session moves wholly into one side (critical for multi-ticker
    panels where many rows share one date).

    The third return value is the FINAL-FIT mask: non-test rows minus the
    `gap` ranks immediately preceding the test block. Without it the final
    model -- the one saved, promoted, and scored on test -- would train on
    rows whose labels are built from test-period prices.
    """
    pos = date_positions(df["date"])
    n_unique = int(df["date"].nunique())
    block, gap, test_start = _geometry(n_unique, n_folds, horizon, embargo_days)

    folds: list[Fold] = []
    for i in range(1, n_folds + 1):
        train_end_rank = i * block - gap
        val_start_rank = i * block
        val_end_rank = (i + 1) * block if i < n_folds else test_start - gap
        train_idx = np.where(pos < train_end_rank)[0]
        val_idx = np.where((pos >= val_start_rank) & (pos < val_end_rank))[0]
        folds.append(Fold(fold=i, train_idx=train_idx, val_idx=val_idx))

    test_idx = np.where(pos >= test_start)[0]
    final_fit_idx = np.where(pos < test_start - gap)[0]
    return folds, test_idx, final_fit_idx
