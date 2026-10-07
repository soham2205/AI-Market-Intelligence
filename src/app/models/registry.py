from __future__ import annotations

from app.models.base import (
    BoostedTreesModel,
    LightGBMModel,
    LogisticModel,
    MajorityBaseline,
)

MODEL_REGISTRY = {
    MajorityBaseline.name: MajorityBaseline,
    LogisticModel.name: LogisticModel,
    LightGBMModel.name: LightGBMModel,
    BoostedTreesModel.name: BoostedTreesModel,
}

# Regression variants register here once implemented; the objective seam is
# already threaded through create_model / run_training / the runs table so
# adding them needs no schema migration.
REGRESSION_REGISTRY: dict[str, type] = {}

BASELINE_NAMES = {MajorityBaseline.name}
CHAMPION_METRIC = "roc_auc_mean"
OBJECTIVES = ("classification", "regression")


def create_model(
    name: str,
    num_cols: list[str],
    cat_cols: list[str],
    objective: str = "classification",
    class_weight: str | None = "balanced",
):
    if objective not in OBJECTIVES:
        raise ValueError(f"unknown objective '{objective}'. available: {OBJECTIVES}")
    if objective == "regression":
        if name not in REGRESSION_REGISTRY:
            raise NotImplementedError(
                f"regression objective not implemented for '{name}'. "
                "The classification pipeline ships first by design; register a "
                "regressor in REGRESSION_REGISTRY to enable it."
            )
        return REGRESSION_REGISTRY[name](num_cols=num_cols, cat_cols=cat_cols)
    if name not in MODEL_REGISTRY:
        raise KeyError(f"unknown model '{name}'. available: {sorted(MODEL_REGISTRY)}")
    return MODEL_REGISTRY[name](
        num_cols=num_cols, cat_cols=cat_cols, class_weight=class_weight
    )
