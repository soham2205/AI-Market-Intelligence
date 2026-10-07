from __future__ import annotations

from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse

app = FastAPI(
    title="AI Market Intelligence Platform",
    description="Read-only API over curated market data and model artifacts.",
    version="0.1.0",
)

CURATED = Path("data/curated")
ARTIFACTS = Path("data/artifacts")


def _read_parquet(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"not found: {path}")
    return pd.read_parquet(path)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/tickers")
def tickers() -> list[str]:
    base = CURATED / "prices"
    if not base.exists():
        raise HTTPException(status_code=404, detail="no curated data")
    return sorted(p.stem.removeprefix("ticker=") for p in base.glob("ticker=*.parquet"))


@app.get("/stocks/{ticker}/prices")
def prices(
    ticker: str,
    start: str | None = Query(default=None),
    end: str | None = Query(default=None),
) -> JSONResponse:
    df = _read_parquet(CURATED / "prices" / f"ticker={ticker.upper()}.parquet")
    df["date"] = pd.to_datetime(df["date"])
    if start:
        df = df[df["date"] >= pd.Timestamp(start)]
    if end:
        df = df[df["date"] <= pd.Timestamp(end)]
    df["date"] = df["date"].dt.strftime("%Y-%m-%d")
    return JSONResponse(df.to_dict(orient="records"))


@app.get("/stocks/{ticker}/sentiment")
def sentiment(ticker: str) -> JSONResponse:
    df = _read_parquet(CURATED / "sentiment_daily.parquet")
    df = df[df["ticker"] == ticker.upper()].copy()
    if df.empty:
        raise HTTPException(status_code=404, detail=f"no sentiment for {ticker}")
    df["trade_date"] = df["trade_date"].astype(str)
    return JSONResponse(df.to_dict(orient="records"))


@app.get("/models")
def models() -> JSONResponse:
    import sqlite3

    db = Path("data/meta.sqlite")
    if not db.exists():
        raise HTTPException(status_code=404, detail="no runs recorded")
    with sqlite3.connect(db) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT run_id, model_name, created_at, metrics_json FROM runs"
            " ORDER BY created_at DESC LIMIT 50"
        ).fetchall()
    return JSONResponse([dict(r) for r in rows])


@app.get("/predictions/{run_id}")
def predictions(run_id: str) -> JSONResponse:
    df = _read_parquet(ARTIFACTS / f"predictions_{run_id}.parquet")
    df["date"] = df["date"].astype(str)
    return JSONResponse(df.to_dict(orient="records"))
