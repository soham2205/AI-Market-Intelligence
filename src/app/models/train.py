from __future__ import annotations

import joblib
import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype

from app.evaluation.metrics import classification_metrics, summarize_folds
from app.features.build import is_feature_column
from app.labeling.targets import is_target_column, label_col
from app.models.registry import BASELINE_NAMES, create_model
from app.splits.purged_cv import purge_gap, split_frame

# Columns carried through the combined matrix for joins and reporting that
# must never reach a model. `source` is the data-provenance tag: it records
# which vendor supplied the bar, so it correlates with which tickers were
# re-sourced during the data audit, not with future returns. Feeding it would
# let the model key on data lineage -- and it is a string, so it would also
# break the numeric imputer.
NON_FEATURE_METADATA = frozenset({"date", "ticker", "source"})


def feature_columns(
    df: pd.DataFrame,
    feature_cols: list[str] | None = None,
    cat_cols: list[str] | None = None,
) -> tuple[list[str], list[str]]:
    """Split columns into (numeric features, categorical features).

    Prefix-driven: EVERY fwd_ret_*/label_up_* column is dropped regardless of
    horizon, so training horizon k can never see horizon j's target. Metadata
    in NON_FEATURE_METADATA is dropped by name, and anything left that is not
    numeric is dropped rather than handed to the numeric imputer.

    `feature_cols` and `cat_cols` override the inference. The pooled
    multi-ticker matrix passes its own list (features.matrix.
    model_feature_columns) and an EMPTY cat_cols: a one-hot over 447 tickers
    cannot generalise to a ticker the model never saw, which is the entire
    point of a pooled model. Passing the list explicitly also keeps training
    and inference reading the same definition of "what feeds the model".
    """
    cat = ["ticker"] if cat_cols is None else list(cat_cols)
    if feature_cols is not None:
        num = [c for c in feature_cols if c not in cat]
    else:
        candidates = [
            c
            for c in df.columns
            if is_feature_column(c) and c not in NON_FEATURE_METADATA and c not in cat
        ]
        num = [c for c in candidates if is_numeric_dtype(df[c])]
    return num, cat


def _assert_no_target_leakage(num_cols: list[str], cat_cols: list[str]) -> None:
    """Fail loud rather than train a quietly leaky model."""
    leaked = [c for c in (*num_cols, *cat_cols) if is_target_column(c)]
    if leaked:
        raise AssertionError(
            f"target columns leaked into the feature matrix: {leaked}. "
            "Every fwd_ret_*/label_up_* column must be excluded by prefix."
        )


def run_training(
    features: pd.DataFrame,
    model_name: str,
    n_folds: int,
    embargo_days: int,
    horizon: int = 1,
    objective: str = "classification",
    class_weight: str | None = "balanced",
    feature_cols: list[str] | None = None,
    cat_cols: list[str] | None = None,
) -> dict:
    """Train one model for ONE horizon with horizon-aware purged walk-forward.

    Independent model per horizon (no multi-output): horizon k selects the
    label column label_up_{k}d, sizes its own purge gap, and produces its own
    artifact, predictions, and champion entry.
    """
    target = label_col(horizon)
    if target not in features.columns:
        raise KeyError(
            f"horizon {horizon} requires column '{target}'; found targets: "
            f"{[c for c in features.columns if is_target_column(c)]}. "
            "Re-run featurize with this horizon configured."
        )

    df = (
        features.dropna(subset=[target])
        .sort_values(["date", "ticker"])
        .reset_index(drop=True)
    )
    y = df[target].astype(int)
    folds, test_idx, final_fit_idx = split_frame(
        df, n_folds, embargo_days, horizon=horizon
    )
    num_cols, cat_cols = feature_columns(df, feature_cols, cat_cols)
    _assert_no_target_leakage(num_cols, cat_cols)
    missing = [c for c in (*num_cols, *cat_cols) if c not in df.columns]
    if missing:
        raise KeyError(f"requested feature columns absent from the matrix: {missing}")
    X = df[num_cols + cat_cols]

    per_fold = []
    val_frames = []
    for fold in folds:
        model = create_model(model_name, num_cols, cat_cols, objective, class_weight)
        model.fit(X.iloc[fold.train_idx], y.iloc[fold.train_idx])
        proba = model.predict_proba(X.iloc[fold.val_idx])[:, 1]
        metrics = classification_metrics(y.iloc[fold.val_idx].to_numpy(), proba)
        metrics["fold"] = fold.fold
        # The validation PERIOD, not just its size: a walk-forward result is
        # only interpretable against the span it was measured over.
        val_dates = df.loc[fold.val_idx, "date"]
        train_dates = df.loc[fold.train_idx, "date"]
        metrics["val_start"] = str(val_dates.min().date())
        metrics["val_end"] = str(val_dates.max().date())
        metrics["train_start"] = str(train_dates.min().date())
        metrics["train_end"] = str(train_dates.max().date())
        metrics["n_train"] = float(len(fold.train_idx))
        per_fold.append(metrics)
        val_frames.append(
            pd.DataFrame(
                {
                    "date": df.loc[fold.val_idx, "date"].to_numpy(),
                    "ticker": df.loc[fold.val_idx, "ticker"].to_numpy(),
                    "proba": proba,
                    "split": f"val_fold_{fold.fold}",
                }
            )
        )

    summary = summarize_folds(per_fold)

    # Final model trains on non-test rows MINUS the purge gap. Rows inside the
    # gap carry labels built from Close_{T+k} that lands in the test block.
    final_model = create_model(model_name, num_cols, cat_cols, objective, class_weight)
    final_model.fit(X.iloc[final_fit_idx], y.iloc[final_fit_idx])

    test_proba = (
        final_model.predict_proba(X.iloc[test_idx])[:, 1]
        if len(test_idx)
        else np.array([])
    )
    test_metrics = (
        classification_metrics(y.iloc[test_idx].to_numpy(), test_proba)
        if len(test_idx)
        else {}
    )

    predictions = pd.concat(
        val_frames
        + [
            pd.DataFrame(
                {
                    "date": df.loc[test_idx, "date"].to_numpy(),
                    "ticker": df.loc[test_idx, "ticker"].to_numpy(),
                    "proba": test_proba,
                    "split": "test",
                }
            )
        ],
        ignore_index=True,
    )
    # Horizon travels WITH the predictions so a 21-day signal can never be
    # mistaken for a 1-day one downstream.
    predictions["horizon"] = horizon
    predictions["objective"] = objective
    predictions = predictions[
        ["date", "ticker", "horizon", "objective", "proba", "split"]
    ]

    baseline_summary = None
    if model_name not in BASELINE_NAMES:
        base_model = create_model("majority", num_cols, cat_cols, objective)
        train_end = max(f.train_idx.max() for f in folds)
        base_model.fit(X.iloc[: train_end + 1], y.iloc[: train_end + 1])
        baseline_fold_metrics = []
        for fold in folds:
            proba_b = base_model.predict_proba(X.iloc[fold.val_idx])[:, 1]
            baseline_fold_metrics.append(
                classification_metrics(y.iloc[fold.val_idx].to_numpy(), proba_b)
            )
        baseline_summary = summarize_folds(baseline_fold_metrics)

    return {
        "summary": summary,
        "per_fold": per_fold,
        "test_metrics": test_metrics,
        "baseline_summary": baseline_summary,
        "predictions": predictions,
        "model": final_model,
        "n_rows": len(df),
        "horizon": horizon,
        "objective": objective,
        "purge_gap": purge_gap(horizon, embargo_days),
        "n_train_final": len(final_fit_idx),
        "num_cols": num_cols,
        "cat_cols": cat_cols,
        "target": target,
        "test_start": str(df.loc[test_idx, "date"].min().date()) if len(test_idx) else None,
        "test_end": str(df.loc[test_idx, "date"].max().date()) if len(test_idx) else None,
    }


def save_model(model: object, artifacts_dir, model_name: str, run_id: str) -> str:
    from pathlib import Path

    out_dir = Path(artifacts_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"model_{model_name}_{run_id}.joblib"
    joblib.dump(model, path)
    return str(path)
