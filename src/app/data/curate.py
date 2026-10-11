from __future__ import annotations

from pathlib import Path

import pandas as pd

from app.data.seams import SOURCE_COLUMN
from app.data.sources.base import REQUIRED_OHLCV

CURATED_SCHEMA = ["date", "ticker", *REQUIRED_OHLCV]


def save_curated(
    df: pd.DataFrame,
    curated_dir: str | Path,
    ticker: str,
    source: str | None = None,
) -> str:
    """Persist one ticker's curated OHLCV.

    `source` records which provider the rows came from. Feature building reads
    it back to decide which data seams apply, so provenance travels with the
    data instead of relying on a caller to remember. Omitting it writes no
    provenance column, which resolves to "no seams".
    """
    out_dir = Path(curated_dir) / "prices"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"ticker={ticker}.parquet"
    if source is not None:
        df = df.copy()
        df[SOURCE_COLUMN] = str(source)
    df.to_parquet(out_path, index=False)
    return str(out_path)


def load_curated(
    curated_dir: str | Path,
    tickers: list[str] | None = None,
    start: str | None = None,
    end: str | None = None,
) -> pd.DataFrame:
    base = Path(curated_dir) / "prices"
    frames: list[pd.DataFrame] = []
    for path in sorted(base.glob("ticker=*.parquet")):
        t = path.stem.removeprefix("ticker=")
        if tickers is not None and t not in tickers:
            continue
        frames.append(pd.read_parquet(path))
    if not frames:
        return pd.DataFrame(columns=CURATED_SCHEMA)
    combined = pd.concat(frames, ignore_index=True)
    combined["date"] = pd.to_datetime(combined["date"])
    if start:
        combined = combined[combined["date"] >= pd.Timestamp(start)]
    if end:
        combined = combined[combined["date"] <= pd.Timestamp(end)]
    return combined.sort_values(["ticker", "date"]).reset_index(drop=True)
