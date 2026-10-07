from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, field_validator


class UniverseConfig(BaseModel):
    tickers: list[str]
    start: str
    end: str | None = None

    @field_validator("tickers")
    @classmethod
    def _clean_tickers(cls, v: list[str]) -> list[str]:
        cleaned = [t.strip().upper() for t in v if t.strip()]
        if not cleaned:
            raise ValueError("universe.tickers must contain at least one ticker")
        if len(set(cleaned)) != len(cleaned):
            raise ValueError("duplicate tickers in universe config")
        return cleaned


class PathsConfig(BaseModel):
    raw_dir: Path = Path("data/raw/prices")
    curated_dir: Path = Path("data/curated")
    artifacts_dir: Path = Path("data/artifacts")
    meta_db: Path = Path("data/meta.sqlite")


class ValidationConfig(BaseModel):
    drop_rows_with_nulls_ohlcv: bool = True
    min_price: float = 0.01
    allow_zero_volume: bool = False


class TrainConfig(BaseModel):
    model: str = "logistic"
    horizons: list[int] = Field(default_factory=lambda: [1, 5, 21, 63, 126, 252])
    objective: str = "classification"
    n_folds: int = 5
    embargo_days: int = 10
    """Serial-correlation hygiene only. The leakage-critical purge is the
    label horizon k, added on top of this per horizon (see splits.purged_cv)."""
    test_fraction: float = 0.2
    positive_threshold: float = 0.0

    @field_validator("horizons")
    @classmethod
    def _clean_horizons(cls, v: list[int]) -> list[int]:
        if not v:
            raise ValueError("train.horizons must contain at least one horizon")
        if any(h < 1 for h in v):
            raise ValueError("train.horizons must be >= 1 trading day")
        if len(set(v)) != len(v):
            raise ValueError("duplicate horizons in train config")
        return sorted(v)

    @field_validator("objective")
    @classmethod
    def _clean_objective(cls, v: str) -> str:
        if v not in {"classification", "regression"}:
            raise ValueError("train.objective must be 'classification' or 'regression'")
        return v


class BacktestConfig(BaseModel):
    entry_threshold: float = 0.55
    exit_threshold: float = 0.50
    cost_bps: float = 10.0


class PipelineConfig(BaseModel):
    universe: UniverseConfig
    paths: PathsConfig = Field(default_factory=PathsConfig)
    validation: ValidationConfig = Field(default_factory=ValidationConfig)
    train: TrainConfig = Field(default_factory=TrainConfig)
    backtest: BacktestConfig = Field(default_factory=BacktestConfig)


def load_config(path: str | Path) -> PipelineConfig:
    with open(path, encoding="utf-8") as f:
        raw: dict[str, Any] = yaml.safe_load(f) or {}
    return PipelineConfig.model_validate(raw)
