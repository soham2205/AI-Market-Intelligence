# AI Market Intelligence

**An event-driven stock research tool that investigates which types of company news are associated with stock moves, how large those moves are, and how the relationship varies by ticker and time horizon.**

Rather than relying on a single general-purpose UP/DOWN prediction model, the project studies historical news events and compares subsequent stock returns with a market benchmark. It then evaluates category-level, sector-level, stock-specific, and rolling historical relationships.

> **Status:** The historical analytics pipeline and current-news processing code are implemented. Live news retrieval still requires a working provider credential, and the historical price/news archive is not current enough to support real-time conclusions. The end-to-end live product is not yet complete.

## What it does

- **Benchmark-adjusted outcomes:** computes forward stock returns relative to SPY over 1, 5, 20, 60, 126, and 252 trading-session horizons.
- **News-category analysis:** compares categories such as earnings, analyst actions, mergers and acquisitions, regulatory/legal events, management changes, and major company news.
- **Sector analysis:** examines whether category associations differ across GICS sectors.
- **Stock-specific sensitivity:** compares each ticker's category-level outcomes against that ticker's own `OTHER` news baseline, with sample-size gates, uncertainty estimates, shrinkage, and multiple-testing correction.
- **Rolling analysis:** checks whether historical relationships strengthen, weaken, reverse, or remain consistent across rolling windows.
- **Current-news pipeline:** retrieves recent articles from a selected provider, scores headlines with FinBERT, assigns one of seven categories, and joins the result to the available historical ticker/category evidence.

The output is **historical evidence, not a guaranteed forecast**. A past association does not mean the next article will produce the same return.

## Architecture

```text
Historical prices + classified news archive
                  |
                  v
       Phase 0: benchmark-adjusted returns
                  |
                  v
       Phase 1: category x horizon x sector
                  |
                  v
       Phase 2: ticker-specific sensitivity
                  |
                  v
       Phase 3: rolling historical stability

Recent news provider
        |
        v
   deduplication -> FinBERT sentiment -> 7-category classifier
        |
        v
   ticker/category match to Phase 2 historical evidence
        |
        v
   current-news report with evidence and limitations
```

## Analysis phases and CLI

Run commands from the repository root after installing the project in editable mode. The analysis commands expect the relevant curated data artifacts to exist locally; large source datasets and generated analysis outputs are not included in Git.

### Install

Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -e ".[dev,nlp]"
```

The `nlp` extra installs Transformers and PyTorch for FinBERT. For the optional API and Streamlit dashboard, install `pip install -e ".[server]"`.

### Historical analysis

```bash
# Phase 0: create benchmark-adjusted forward-return outcomes
python -m app.cli benchmark-returns

# Phase 1: analyze category, horizon, and sector relationships
python -m app.cli category-horizon-returns

# Phase 2: estimate ticker-specific category sensitivity
python -m app.cli stock-sensitivity

# Phase 3: evaluate rolling historical stability
python -m app.cli rolling-impact
```

These commands use default paths defined in the code and support path/version overrides. Run `python -m app.cli --help` and `python -m app.cli COMMAND --help` for available options. They require the curated price panel and the preceding phase's artifacts; they do not download or rebuild the large historical archive automatically.

### Current-news processing

```bash
python -m app.cli current-news --tickers BEN --source marketaux --provider-limit 3 --scorer finbert
```

Provider configuration:

- **Marketaux:** configure `MARKETAUX_API_TOKEN` in the environment or in a local, git-ignored `.env` file. The free plan used during implementation was limited to three articles per request; the pipeline reports provider truncation rather than treating a small sample as complete coverage.
- **Alpha Vantage:** configure `ALPHAVANTAGE_API_KEY`. The `NEWS_SENTIMENT` endpoint returned a premium-access restriction during implementation, so an API key alone may not grant access.
- **Yahoo Finance:** can be selected without a credential, but its news feed was returning zero items during the latest development checks.

Do not commit API keys or put them in command-line arguments. Keep credentials in environment variables or a local `.env` file, and rotate any key that may have been exposed.

The current-news pipeline matches new headlines to historical evidence. It does **not** calculate future returns for newly published articles, and it must not be interpreted as a live return forecast.

## Categories

The deterministic, priority-ordered headline classifier assigns one of:

- `EARNINGS_RESULTS`
- `ANALYST_ACTION`
- `M_AND_A`
- `REGULATORY_LEGAL`
- `MANAGEMENT`
- `MAJOR_COMPANY_NEWS`
- `OTHER`

The historical category labels were supplied by an artifact without its original generating code. The new classifier was measured at approximately 91% overall agreement with those labels, but agreement varies by category; `MAJOR_COMPANY_NEWS` is notably weaker. Cross-era comparisons should therefore be treated as approximate.

FinBERT is used for sentiment scoring. Sentiment is not a substitute for event classification: the pipeline treats them as separate signals.

## Data, outputs, and reproducibility

The historical analysis was built from a curated FNSPID news/price archive and a 447-ticker curated panel. Local data artifacts are intentionally excluded from the repository; their exact availability depends on the local setup.

The last audited historical panel ended on **January 8, 2024**. For rolling analysis, the latest assessable date is earlier at longer forward-return horizons because the full outcome window must be observed. Until newer historical data and a working current-news source are available, the project is best described as **historical news-impact analysis with an implemented but not yet verified live-ingestion path**.

Typical generated outputs include:

- benchmark-adjusted return outcomes and summary reports;
- category/horizon and sector summaries;
- ticker-specific sensitivity summaries and comparison tables;
- rolling category/stock results, temporal-change summaries, freshness metadata, and reports;
- current-news article, event, evidence, run metadata, and report artifacts.

Exact filenames and output locations are documented in the relevant CLI help and module defaults.

## Statistical and methodological cautions

- Forward-return horizons overlap, so uncertainty estimation accounts for temporal dependence where implemented; long-horizon results remain exploratory when the effective sample is small.
- Multiple-testing corrections are applied to relevant comparison families. Exploratory results are not independent confirmations.
- Sparse ticker/category combinations may have insufficient evidence. An empty or inconclusive result does **not** establish that a news category has no effect.
- The sector mapping is a current-universe snapshot, not a point-in-time historical membership record. Back-extending today's constituents introduces survivorship bias.
- The analysis uses market-adjusted returns against SPY, not a full factor model or a beta-adjusted CAPM abnormal-return estimate.
- The historical news archive is date-dominated and some publication timestamps are placeholders. Conservative next-trading-session assignment is used to reduce look-ahead risk.
- Historical category-label comparability is imperfect, and live provider coverage may be incomplete or capped by the plan.
- This is an educational research project, not investment advice.

## Tests and code quality

Run the test suite from the repository root:

```bash
pytest
ruff check .
```

The repository includes tests for benchmark returns, category/horizon analysis, stock sensitivity, rolling analysis, news-provider adapters, sentiment/classification, data validation, feature construction, and the original forecasting/backtesting pipeline. Passing tests validate the tested code paths; they do not establish that an external news provider is currently available or that the live product works end to end.

## Original forecasting pipeline

The repository also retains the earlier technical-feature and machine-learning pipeline, including feature construction, purged walk-forward evaluation, model training, and backtesting. Initial broad-universe experiments were close to chance-level ROC-AUC, which motivated the pivot toward investigating **which news events matter for which stocks and over what horizons**, rather than presenting a generic directional model as reliable.

## Repository structure

```text
src/app/
  data/       data sources, curation, and validation
  nlp/        FinBERT scoring, bucketing, and category classification
  events/     benchmark returns, category analysis, stock sensitivity,
              rolling analysis, and current-news evidence
  features/   technical and sentiment feature construction
  models/     model implementations and registry
  backtest/   backtesting engine and baselines
  api/        optional API
  dashboard/  optional Streamlit dashboard
tests/        automated tests for analytical and pipeline components
configs/      universe and sector configuration
```

## License and disclaimer

No investment recommendation is made by this software. Verify data-source terms and redistribution rights before sharing datasets or derived content. **Educational research tool only — not investment advice.**
