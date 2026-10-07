from __future__ import annotations

from collections.abc import Sequence

import pandas as pd

DEFAULT_HORIZONS: tuple[int, ...] = (1, 5, 21, 63, 126, 252)

FWD_RET_PREFIX = "fwd_ret_"
LABEL_PREFIX = "label_up_"
TARGET_PREFIXES: tuple[str, ...] = (FWD_RET_PREFIX, LABEL_PREFIX)


def fwd_ret_col(horizon: int) -> str:
    return f"{FWD_RET_PREFIX}{horizon}d"


def label_col(horizon: int) -> str:
    return f"{LABEL_PREFIX}{horizon}d"


def is_target_column(col: str) -> bool:
    """True for any horizon's target column.

    Prefix-based on purpose: adding a horizon must never silently widen the
    feature matrix. A literal allow/deny list is how cross-horizon leakage
    gets introduced.
    """
    return col.startswith(TARGET_PREFIXES)


def target_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if is_target_column(c)]


def forward_return(df: pd.DataFrame, horizon: int = 1) -> pd.Series:
    """Cumulative forward return over `horizon` sessions, aligned to day T.

    r_{T,k} = Close_{T+k} / Close_T - 1

    The shift runs INSIDE each ticker group via transform, so the trailing k
    rows of one ticker can never borrow the leading rows of the next.
    """
    if horizon < 1:
        raise ValueError(f"horizon must be >= 1, got {horizon}")
    return df.groupby("ticker")["close"].transform(
        lambda s: s.shift(-horizon) / s - 1
    )


def add_labels(
    df: pd.DataFrame,
    threshold: float = 0.0,
    horizons: Sequence[int] = (1,),
) -> pd.DataFrame:
    """Attach fwd_ret_{k}d / label_up_{k}d for every requested horizon.

    y_{T,k} = 1 if Close_{T+k}/Close_T - 1 > threshold else 0.
    The trailing k rows per ticker have no T+k and stay NaN.
    """
    out = df.sort_values(["ticker", "date"]).copy()
    for k in horizons:
        fwd = forward_return(out, k)
        out[fwd_ret_col(k)] = fwd
        out[label_col(k)] = (fwd > threshold).astype(float).where(fwd.notna())
    return out.reset_index(drop=True)
