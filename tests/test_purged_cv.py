from __future__ import annotations

import pytest

from app.splits.purged_cv import make_split_plan, split_frame


def test_plan_geometry_and_embargo():
    n_unique_dates, n_folds, embargo_days = 1000, 4, 10
    block = n_unique_dates // (n_folds + 2)
    folds, test_range = make_split_plan(n_unique_dates, n_folds, embargo_days)
    for i, f in enumerate(folds, start=1):
        assert f.val_idx.min() == i * block
        assert f.train_idx.max() < f.val_idx.min()
        assert len(f.val_idx) > 0
        assert earliest_val_rank_gap(f) >= embargo_days
    assert test_range == (n_unique_dates - block, n_unique_dates)


def earliest_val_rank_gap(fold):
    return int(fold.val_idx.min() - fold.train_idx.max() - 1)


def test_folds_are_chronological():
    folds, _ = make_split_plan(n_unique_dates=600, n_folds=3, embargo_days=5)
    ends = [f.val_idx.max() for f in folds]
    starts = [f.val_idx.min() for f in folds]
    assert starts == sorted(starts)
    assert all(a < b for a, b in zip(ends[:-1], starts[1:], strict=True))


def test_too_short_raises():
    with pytest.raises(ValueError):
        make_split_plan(n_unique_dates=50, n_folds=5, embargo_days=30)


def test_split_frame_positions_align_with_dates(make_panel):
    panel = make_panel(tickers=("AAPL",), n=300)
    df = panel.sort_values(["date", "ticker"]).reset_index(drop=True)
    folds, test_idx, final_fit_idx = split_frame(df, n_folds=3, embargo_days=7)

    rank = {d: i for i, d in enumerate(sorted(df["date"].unique()))}
    test_dates = set(df.loc[test_idx, "date"])
    for fold in folds:
        train_ranks = [rank[d] for d in set(df.loc[fold.train_idx, "date"])]
        val_ranks = [rank[d] for d in set(df.loc[fold.val_idx, "date"])]
        assert not set(train_ranks) & set(val_ranks)
        assert min(val_ranks) - max(train_ranks) > 7
        assert not {rank[d] for d in test_dates} & set(train_ranks)
