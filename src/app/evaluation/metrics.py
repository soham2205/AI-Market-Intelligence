from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    auc,
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
    log_loss,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)


def classification_metrics(y_true: np.ndarray, proba_up: np.ndarray) -> dict[str, float | None]:
    y_true = np.asarray(y_true).astype(int)
    proba_up = np.clip(np.asarray(proba_up, dtype=float), 1e-9, 1 - 1e-9)

    metrics: dict[str, float | None] = {}
    metrics["roc_auc"] = (
        roc_auc_score(y_true, proba_up) if len(np.unique(y_true)) > 1 else None
    )
    if len(np.unique(y_true)) > 1:
        precision, recall, _ = precision_recall_curve(y_true, proba_up)
        metrics["pr_auc"] = float(auc(recall, precision))
    else:
        metrics["pr_auc"] = None
    pred = (proba_up >= 0.5).astype(int)
    metrics["precision"] = float(precision_score(y_true, pred, zero_division=0))
    metrics["recall"] = float(recall_score(y_true, pred, zero_division=0))
    metrics["f1"] = float(f1_score(y_true, pred, zero_division=0))
    metrics["balanced_accuracy"] = balanced_accuracy_score(y_true, pred)
    metrics["brier"] = float(brier_score_loss(y_true, proba_up))
    metrics["log_loss"] = float(log_loss(y_true, proba_up, labels=[0, 1]))
    # Base rate travels with every metric block: at long horizons equity drift
    # pushes it toward 0.8, where a majority vote "scores" 0.8 accuracy. Any
    # accuracy-like number must be read against this.
    metrics["pos_rate"] = float(np.mean(y_true))
    metrics["n_samples"] = float(len(y_true))
    return metrics


def summarize_folds(per_fold: list[dict[str, float | None]]) -> dict[str, float]:
    """Mean/std across folds for every NUMERIC metric.

    Fold dicts also carry descriptive fields (the validation period's start and
    end dates, for instance). Those are not averageable, so they are skipped
    rather than coerced -- averaging a date string is meaningless and numpy
    raises on it.
    """
    summary: dict[str, float] = {}
    keys = [
        k
        for k, v in per_fold[0].items()
        if k != "n_samples" and v is not None and isinstance(v, (int, float))
    ]
    for key in keys:
        values = [m[key] for m in per_fold if m.get(key) is not None]
        summary[f"{key}_mean"] = float(np.mean(values))
        summary[f"{key}_std"] = float(np.std(values))
    summary["n_samples_total"] = float(sum(m["n_samples"] for m in per_fold))  # type: ignore[arg-type]
    return summary
