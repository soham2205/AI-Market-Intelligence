# AI Market Intelligence & Short-Term Forecasting Platform

Educational platform combining market data, technical indicators, news sentiment,
ML-based short-term forecasting, and backtesting — with strict point-in-time
correctness (no look-ahead bias, purged walk-forward validation).

## Pipeline

```
ingest-prices ──► ingest-news ──► featurize ──► train ──► backtest ──► serve-api / dashboard
     │                │               │            │           │
  yfinance        headlines +      features +   models vs   T+1 fills,
  OHLCV →         FinBERT/lexicon  labels       baselines   costs,
  validate        → session                     purged CV   equity curve
                  bucketing
```

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -e ".[dev]"
```

Optional extras: `pip install -e ".[nlp]"` (real FinBERT via torch),
`pip install -e ".[server]"` (FastAPI + Streamlit). Without them the
sentiment scorer falls back to a deterministic finance lexicon.

## Run the full pipeline

```bash
python src/app/cli.py ingest-prices --config configs/universe.yaml
python src/app/cli.py ingest-news   --config configs/universe.yaml
python src/app/cli.py featurize     --config configs/universe.yaml
python src/app/cli.py train         --config configs/universe.yaml --model logistic
python src/app/cli.py train         --config configs/universe.yaml --model hgb
python src/app/cli.py backtest      --config configs/universe.yaml --model logistic
```

Models: `majority` (baseline), `logistic`, `hgb` (hist gradient boosting).
Every training run reports CV metrics **and** the majority-class baseline;
every backtest compares against buy&hold and an SMA20>50 rule with identical costs.

## Methodology guarantees

- Feature row for day T uses only information observable at close of day T
  (rolling indicators are backward-looking; daily OHLCV is known at T's
  close). The label describes the FOLLOWING session:
  `y_T = 1 iff Close_{T+1} > Close_T` — a decision from row T is executable
  at T+1's open, exactly the backtester's execution model.
- Purged expanding-window walk-forward splits with embargo between train and
  validation; test period is touched exactly once per run.
- Backtest executes signals at next open (entry earns open→close; exits earn
  overnight only) and charges `cost_bps` on every turnover.
- News published during/after a session maps to the NEXT trading day
  (conservative anti-leakage bucketing); overnight news is available to that
  session's feature row.

## Storage layout

```
data/
  curated/prices/ticker=*.parquet    # cleaned OHLCV (date,ticker,OHLCV)
  curated/news.parquet               # deduped, scored headlines
  curated/sentiment_daily.parquet    # per ticker/day aggregates + EWMA
  curated/features.parquet           # wide feature matrix + label_up_1d
  artifacts/model_*.joblib           # trained pipelines
  artifacts/predictions_*.parquet    # val-fold + test probabilities
  meta.sqlite                        # run registry + backtest records
```

## API & dashboard (optional extras)

```bash
python src/app/cli.py serve-api          # FastAPI at http://127.0.0.1:8000/docs
streamlit run src/app/dashboard/app.py   # interactive dashboard
```

## Tests

```bash
pytest
```

Covers: price validation, no-lookahead feature construction, label math,
purged-CV geometry/embargo, model registry, metrics, backtester fills and
costs (hand-computed scenarios), sentiment bucketing/aggregation, and an
end-to-end training smoke test.

## Extending

- New data source: implement the `PriceSource` protocol in
  `src/app/data/sources/base.py`.
- New model: add a class in `src/app/models/base.py` + register it in
  `registry.py`. Nothing else changes.
- Real FinBERT: `pip install -e ".[nlp]"` then re-run `ingest-news`.

**Disclaimer:** educational research tool — not investment advice.
