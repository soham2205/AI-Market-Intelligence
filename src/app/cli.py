from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from app.config import PipelineConfig, load_config


def _cmd_ingest_prices(cfg: PipelineConfig, source_name: str = "yfinance") -> int:
    from app.data.curate import save_curated
    from app.data.sources.yfinance_source import YFinanceSource
    from app.data.validate import validate_prices

    if source_name == "fnspid":
        from app.data.sources.fnspid_price_source import FnspidPriceSource

        source = FnspidPriceSource()
        if not source.available():
            print("[ERROR] FNSPID price archive not cached; run download_fnspid_prices()")
            return 1
    else:
        source = YFinanceSource()
    # Provenance is written alongside the prices so feature building can look
    # up this source's data seams without any caller passing them along.
    print(f"[INFO] price source: {source.name}")
    exit_code = 0
    for ticker in cfg.universe.tickers:
        try:
            raw = source.fetch(ticker, cfg.universe.start, cfg.universe.end)
            clean, report = validate_prices(
                raw,
                ticker=ticker,
                min_price=cfg.validation.min_price,
                allow_zero_volume=cfg.validation.allow_zero_volume,
            )
        except Exception as exc:
            print(f"[ERROR] {ticker}: {exc}")
            exit_code = 1
            continue
        if clean.empty:
            print(f"[WARN] {ticker}: no valid rows after validation, skipping save")
            exit_code = 1
            continue
        path = save_curated(clean, cfg.paths.curated_dir, ticker, source=source.name)
        span0, span1 = clean["date"].min().date(), clean["date"].max().date()
        print(f"[OK] {report.summary()}")
        print(f"     saved -> {path} ({span0} .. {span1})")
    return exit_code


def _cmd_ingest_news(
    cfg: PipelineConfig,
    source_name: str = "yfinance",
    scorer_name: str = "auto",
    start: str | None = None,
    end: str | None = None,
) -> int:
    import pandas as pd

    from app.data.curate import load_curated
    from app.data.sources.news_source import YFinanceNewsSource, dedupe_news
    from app.nlp.aggregate import add_ewma, aggregate_daily
    from app.nlp.bucketing import first_tradable_session
    from app.nlp.finbert_scorer import get_scorer

    curated = load_curated(cfg.paths.curated_dir)
    sessions = pd.DatetimeIndex(
        pd.to_datetime(curated["date"]).dt.normalize().unique()
    ).sort_values()

    # Strict when asked for FinBERT: the research run must know exactly which
    # model produced its scores rather than silently degrading to the lexicon.
    try:
        scorer = get_scorer(scorer_name)
    except RuntimeError as exc:
        print(f"[ERROR] {exc}")
        return 1
    print(f"[INFO] sentiment backend: {scorer.name} (requested: {scorer_name})")

    if source_name == "fnspid":
        from app.data.sources.fnspid_source import FnspidNewsSource

        source = FnspidNewsSource()
        if not source.available():
            print("[ERROR] FNSPID artifact missing; run `fnspid-extract` first")
            return 1
    else:
        source = YFinanceNewsSource()
    print(f"[INFO] news source: {source.name}  window: {start or 'all'} .. {end or 'all'}")

    frames = []
    for ticker in cfg.universe.tickers:
        try:
            news = dedupe_news(source.fetch(ticker, start, end))
        except Exception as exc:
            print(f"[WARN] {ticker}: news fetch failed ({exc})")
            continue
        if news.empty:
            print(f"[WARN] {ticker}: no news found (free feeds are recent-only)")
            continue
        news["trade_date"] = news["published_at"].map(
            lambda ts: first_tradable_session(ts, sessions)
        )
        # Bucketing now rolls onto a real session, so NaT only means "later
        # than the last known session" rather than "landed on a holiday".
        news = news[news["trade_date"].notna()]
        news["score"] = news["headline"].map(scorer.score)
        news["score"] = news["score"].clip(-1, 1)
        frames.append(news)
        print(f"[OK] {ticker}: scored {len(news)} articles")

    if not frames:
        print("[WARN] no news collected; sentiment table not written")
        return 1

    all_news = pd.concat(frames, ignore_index=True)
    all_news.attrs["scorer"] = scorer.name
    news_path = Path(cfg.paths.curated_dir) / "news.parquet"
    all_news.to_parquet(news_path, index=False)
    sentiment_daily = add_ewma(aggregate_daily(all_news))
    sent_path = Path(cfg.paths.curated_dir) / "sentiment_daily.parquet"
    sentiment_daily.to_parquet(sent_path, index=False)
    print(f"[OK] saved -> {news_path} and {sent_path}")
    return 0


def _cmd_featurize(cfg: PipelineConfig) -> int:
    import pandas as pd

    from app.data.curate import load_curated
    from app.features.build import build_features
    from app.labeling.targets import label_col

    curated = load_curated(cfg.paths.curated_dir)
    if curated.empty:
        print("[ERROR] no curated prices found; run ingest-prices first")
        return 1

    sent_path = Path(cfg.paths.curated_dir) / "sentiment_daily.parquet"
    # build_features keys sentiment on `trade_date` (the bucketed session), so
    # the column is passed through untouched -- renaming it here is what used
    # to raise KeyError: 'trade_date'.
    sentiment = pd.read_parquet(sent_path) if sent_path.exists() else None
    if sentiment is None:
        print("[INFO] no sentiment table found; building price-only features")

    feats = build_features(
        curated,
        sentiment=sentiment,
        positive_threshold=cfg.train.positive_threshold,
        horizons=cfg.train.horizons,
    )
    out_path = Path(cfg.paths.curated_dir) / "features.parquet"
    feats.to_parquet(out_path, index=False)
    print(f"[OK] features -> {out_path} rows={len(feats)} cols={len(feats.columns)}")
    for k in cfg.train.horizons:
        col = label_col(k)
        labeled = int(feats[col].notna().sum())
        pos = float(feats[col].mean())
        print(f"     h={k:<4} labeled={labeled:<7} pos_rate={pos:.3f}")
    return 0


def _resolve_feature_set(
    features, feature_set: str
) -> tuple[list[str] | None, list[str] | None, str]:
    """Decide which columns feed the model, and say so out loud.

    "matrix"  -- the pooled combined matrix (features.matrix): its own
                 model_feature_columns list, and NO ticker one-hot, because a
                 one-hot over hundreds of tickers cannot generalise to an
                 unseen ticker.
    "legacy"  -- the original 9-ticker features.parquet: inferred columns with
                 ticker as a categorical.
    "auto"    -- "matrix" when the frame carries the matrix's metadata columns,
                 else "legacy".
    """
    from app.features.matrix import METADATA_COLUMNS, model_feature_columns

    resolved = feature_set
    if feature_set == "auto":
        looks_like_matrix = all(c in features.columns for c in METADATA_COLUMNS)
        resolved = "matrix" if looks_like_matrix else "legacy"
    if resolved == "matrix":
        return model_feature_columns(features), [], resolved
    return None, None, resolved


def _cmd_train(
    cfg: PipelineConfig,
    model: str | None,
    horizons: str | None,
    features_path: str | None = None,
    models: str | None = None,
    artifacts_dir: str | None = None,
    meta_db: str | None = None,
    feature_set: str = "auto",
    report_path: str | None = None,
) -> int:
    import json
    import uuid

    import pandas as pd

    from app.models.train import run_training, save_model
    from app.storage import get_champion, record_run, set_champion

    # An explicit --features path is honoured verbatim so a specific matrix
    # version can be trained without any chance of picking up a stale default.
    feats_path = (
        Path(features_path)
        if features_path
        else Path(cfg.paths.curated_dir) / "features.parquet"
    )
    if not feats_path.exists():
        print(f"[ERROR] features not found at {feats_path}")
        return 1
    features = pd.read_parquet(feats_path)
    print(
        f"[OK] features <- {feats_path} rows={len(features)} "
        f"cols={len(features.columns)}"
    )

    out_dir = Path(artifacts_dir) if artifacts_dir else Path(cfg.paths.artifacts_dir)
    db_path = Path(meta_db) if meta_db else Path(cfg.paths.meta_db)
    feature_cols, cat_cols, resolved_set = _resolve_feature_set(features, feature_set)
    n_feat = len(feature_cols) if feature_cols is not None else "inferred"
    print(
        f"[OK] feature set = {resolved_set} ({n_feat} numeric, "
        f"ticker as feature: {bool(cat_cols)})"
    )
    print(f"[OK] artifacts -> {out_dir} | meta db -> {db_path}")

    model_names = (
        [m.strip() for m in models.split(",") if m.strip()]
        if models
        else [model or cfg.train.model]
    )
    selected = (
        [int(h) for h in horizons.split(",")] if horizons else list(cfg.train.horizons)
    )
    unknown = [h for h in selected if h not in cfg.train.horizons]
    if unknown:
        print(f"[ERROR] horizons {unknown} not in config train.horizons")
        return 1

    batch_id = uuid.uuid4().hex[:12]
    exit_code = 0
    report: dict = {
        "batch_id": batch_id,
        "input_matrix": str(feats_path),
        "input_rows": int(len(features)),
        "feature_set": resolved_set,
        "validation": {
            "scheme": "purged expanding-window walk-forward",
            "n_folds": cfg.train.n_folds,
            "embargo_days": cfg.train.embargo_days,
            "positive_threshold": cfg.train.positive_threshold,
        },
        "horizons": selected,
        "models": model_names,
        "runs": [],
    }
    for model_name, horizon in [(m, h) for h in selected for m in model_names]:
        try:
            result = run_training(
                features,
                model_name=model_name,
                n_folds=cfg.train.n_folds,
                embargo_days=cfg.train.embargo_days,
                horizon=horizon,
                objective=cfg.train.objective,
                feature_cols=feature_cols,
                cat_cols=cat_cols,
            )
        except (ValueError, KeyError, NotImplementedError) as exc:
            print(f"[ERROR] {model_name} h={horizon}: {exc}")
            report["runs"].append(
                {"model": model_name, "horizon": horizon, "error": str(exc)}
            )
            exit_code = 1
            continue

        print(
            f"\n=== {model_name} h={horizon} ({cfg.train.n_folds}-fold purged "
            f"walk-forward, purge gap {result['purge_gap']} sessions "
            f"= horizon {horizon} + embargo {cfg.train.embargo_days}) ==="
        )
        for key, value in sorted(result["summary"].items()):
            print(f"  {key}: {value:.4f}" if isinstance(value, float) else f"  {key}: {value}")
        if result["baseline_summary"]:
            print("--- baseline (majority class) ---")
            for key, value in sorted(result["baseline_summary"].items()):
                print(f"  {key}: {value:.4f}" if isinstance(value, float) else f"  {key}: {value}")
        if result["test_metrics"]:
            print("--- TEST PERIOD (touched once) ---")
            for key, value in sorted(result["test_metrics"].items()):
                print(f"  {key}: {value}")

        run_id = uuid.uuid4().hex[:12]
        artifact = save_model(result["model"], out_dir, model_name, run_id)
        run_config = {
            "input_matrix": str(feats_path),
            "feature_set": resolved_set,
            "target": result["target"],
            "target_definition": (
                f"label_up_{horizon}d = 1 if Close(T+{horizon})/Close(T) - 1 > "
                f"{cfg.train.positive_threshold}"
            ),
            "feature_columns": result["num_cols"],
            "categorical_columns": result["cat_cols"],
            "n_features": len(result["num_cols"]) + len(result["cat_cols"]),
            "validation_scheme": "purged expanding-window walk-forward",
            "n_folds": cfg.train.n_folds,
            "embargo_days": cfg.train.embargo_days,
            "purge_gap": result["purge_gap"],
            "horizon": horizon,
            "objective": cfg.train.objective,
            "positive_threshold": cfg.train.positive_threshold,
            "n_train_final": result["n_train_final"],
            "test_start": result["test_start"],
            "test_end": result["test_end"],
        }
        record_run(
            db_path,
            model_name=model_name,
            config=run_config,
            metrics={
                "cv": result["summary"],
                "baseline_cv": result["baseline_summary"],
                "test": result["test_metrics"],
            },
            artifact_path=artifact,
            run_id=run_id,
            horizon=horizon,
            objective=cfg.train.objective,
            batch_id=batch_id,
        )

        pred_path = out_dir / f"predictions_{run_id}.parquet"
        result["predictions"].to_parquet(pred_path, index=False)

        champion_metric = float(result["summary"].get("roc_auc_mean", float("-inf")))
        promoted = set_champion(
            db_path,
            run_id=run_id,
            model_name=model_name,
            metric_value=champion_metric,
            horizon=horizon,
            objective=cfg.train.objective,
        )
        if promoted:
            print(
                f"[OK] h={horizon} promoted to champion "
                f"(CV ROC-AUC {champion_metric:.4f})"
            )
        else:
            champ = get_champion(db_path, horizon=horizon)
            print(f"[INFO] h={horizon} not promoted; champion remains "
                  f"{champ['model_name']} (CV ROC-AUC {champ['metric_value']:.4f})")
        print(f"[OK] run={run_id} h={horizon} model={artifact} predictions={pred_path}")

        report["runs"].append(
            {
                "run_id": run_id,
                "model": model_name,
                "horizon": horizon,
                "config": run_config,
                "cv_summary": result["summary"],
                "per_fold": result["per_fold"],
                "baseline_cv_summary": result["baseline_summary"],
                "test_metrics": result["test_metrics"],
                "n_rows_used": result["n_rows"],
                "artifact_path": artifact,
                "predictions_path": str(pred_path),
                "promoted": promoted,
            }
        )

    for horizon in selected:
        champ = get_champion(db_path, horizon=horizon)
        if champ:
            report.setdefault("champions", {})[str(horizon)] = champ
    if report_path:
        Path(report_path).parent.mkdir(parents=True, exist_ok=True)
        Path(report_path).write_text(json.dumps(report, indent=1, default=str))
        print(f"[OK] report -> {report_path}")

    print(f"\n[OK] batch={batch_id} horizons={selected} models={model_names}")
    return exit_code


def _cmd_backtest(cfg: PipelineConfig, run_id: str | None) -> int:
    import pandas as pd

    from app.backtest.baselines import buy_hold_returns, sma_crossover_predictions
    from app.backtest.engine import performance_metrics, run_backtest
    from app.data.curate import load_curated
    from app.storage import latest_run, record_backtest

    # The backtester is still the 1-day engine; it is deliberately untouched
    # by the multi-horizon work, so it may only ever consume a 1-day run.
    meta = latest_run(cfg.paths.meta_db, run_id, horizon=1)
    if meta is None:
        print("[ERROR] no 1-day training runs found; run train first")
        return 1
    run_id = meta["run_id"]
    pred_path = Path(cfg.paths.artifacts_dir) / f"predictions_{run_id}.parquet"
    if not pred_path.exists():
        print(f"[ERROR] predictions for run {run_id} missing at {pred_path}")
        return 1

    predictions = pd.read_parquet(pred_path)
    if "horizon" in predictions.columns:
        found = sorted(predictions["horizon"].unique())
        if found != [1]:
            print(f"[ERROR] backtester is 1-day only; run {run_id} holds horizons {found}")
            return 1
    prices = load_curated(cfg.paths.curated_dir)
    test_pred = predictions[predictions["split"] == "test"]
    start, end = str(test_pred["date"].min().date()), str(test_pred["date"].max().date())

    bt = run_backtest(
        test_pred,
        prices,
        entry_threshold=cfg.backtest.entry_threshold,
        exit_threshold=cfg.backtest.exit_threshold,
        cost_bps=cfg.backtest.cost_bps,
    )
    bh = buy_hold_returns(prices, start=start, end=end)
    bh_metrics = performance_metrics(bh.to_numpy(), bh.index)
    sma_pred = sma_crossover_predictions(prices)
    sma_pred = sma_pred[
        (sma_pred["date"] >= pd.Timestamp(start)) & (sma_pred["date"] <= pd.Timestamp(end))
    ]
    sma_bt = run_backtest(
        sma_pred,
        prices,
        entry_threshold=0.5,
        exit_threshold=0.5,
        cost_bps=cfg.backtest.cost_bps,
    )

    def show(label: str, m: dict) -> None:
        print(
            f"{label:<22} total_return={m['total_return']:>8.2%} cagr={m['cagr']:>8.2%}"
            f" sharpe={m['sharpe']:>6.2f} max_dd={m['max_drawdown']:>8.2%}"
        )

    print(f"\n=== Backtest {start} .. {end} (costs {cfg.backtest.cost_bps} bps/turnover) ===")
    show("model", bt["metrics"])
    show("buy&hold universe", bh_metrics)
    show("sma20>sma50 rule", sma_bt["metrics"])
    print(f"model cost drag: {bt['metrics']['total_cost_drag']:.4f}, "
          f"avg turnover: {bt['metrics']['avg_daily_turnover']:.4f}")

    equity_json = {
        "dates": [str(d.date()) for d in bt["equity_curve"].index],
        "equity": bt["equity_curve"].tolist(),
    }
    record_backtest(
        cfg.paths.meta_db,
        run_id=run_id,
        params={
            "entry_threshold": cfg.backtest.entry_threshold,
            "exit_threshold": cfg.backtest.exit_threshold,
            "cost_bps": cfg.backtest.cost_bps,
            "period": [start, end],
        },
        metrics={
            "model": bt["metrics"],
            "buy_hold": bh_metrics,
            "sma_rule": sma_bt["metrics"],
        },
        equity_curve=equity_json,
    )
    eq_path = Path(cfg.paths.artifacts_dir) / f"equity_curve_{run_id}.parquet"
    bt["equity_curve"].rename("equity").to_frame().to_parquet(eq_path)
    print(f"[OK] recorded backtest for run {run_id}; curve -> {eq_path}")
    return 0


def _cmd_fnspid_extract(cfg: PipelineConfig, skip_download: bool) -> int:
    from app.data.sources.fnspid_source import (
        download_fnspid,
        extract_universe,
        summarize,
    )

    if not skip_download:
        download_fnspid()
    stats = extract_universe(cfg.universe.tickers)
    print("\n=== FNSPID extraction ===")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    print("\n=== per-ticker coverage ===")
    print(summarize().to_string())
    return 0


def _cmd_serve_api(cfg: PipelineConfig, host: str, port: int) -> int:
    import uvicorn

    uvicorn.run("app.api.main:app", host=host, port=port)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="app.cli")
    sub = parser.add_subparsers(dest="command", required=True)

    def add_config(p: argparse.ArgumentParser) -> None:
        p.add_argument("--config", default="configs/universe.yaml")

    p = sub.add_parser("ingest-prices", help="Fetch + validate + store OHLCV")
    add_config(p)
    p.add_argument("--source", default="yfinance", choices=["yfinance", "fnspid"])

    p = sub.add_parser("ingest-news", help="Fetch headlines, score sentiment, bucket to sessions")
    add_config(p)
    p.add_argument("--source", default="yfinance", choices=["yfinance", "fnspid"])
    p.add_argument(
        "--scorer",
        default="auto",
        choices=["auto", "lexicon", "finbert"],
        help="'finbert' fails loudly if FinBERT is unavailable (research runs)",
    )
    p.add_argument("--start", default=None)
    p.add_argument("--end", default=None)

    p = sub.add_parser("fnspid-extract", help="Download + chunk-filter FNSPID to our universe")
    add_config(p)
    p.add_argument("--skip-download", action="store_true")

    p = sub.add_parser("featurize", help="Build point-in-time feature matrix with labels")
    add_config(p)

    p = sub.add_parser("train", help="Train per-horizon models with purged walk-forward CV")
    add_config(p)
    p.add_argument(
        "--model",
        default=None,
        choices=["majority", "logistic", "lightgbm", "hgb"],
    )
    p.add_argument(
        "--horizons",
        default=None,
        help="Comma-separated subset of config train.horizons (default: all)",
    )
    p.add_argument(
        "--features",
        default=None,
        help="Explicit path to the feature matrix (default: curated/features.parquet)",
    )
    p.add_argument(
        "--models",
        default=None,
        help="Comma-separated model names to train in one batch (overrides --model)",
    )
    p.add_argument("--artifacts-dir", default=None, help="Override artifact output dir")
    p.add_argument(
        "--meta-db", default=None, help="Override run/champion database path"
    )
    p.add_argument(
        "--feature-set",
        default="auto",
        choices=["auto", "matrix", "legacy"],
        help="Which column convention to use (see _resolve_feature_set)",
    )
    p.add_argument("--report", default=None, help="Write a JSON training report here")

    p = sub.add_parser("backtest", help="Backtest latest trained run vs baselines")
    add_config(p)
    p.add_argument("--model", default=None, help="Model name of the run to backtest")

    p = sub.add_parser("serve-api", help="Start the FastAPI server")
    add_config(p)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)

    args = parser.parse_args(argv)
    cfg = load_config(args.config)

    if args.command == "ingest-prices":
        return _cmd_ingest_prices(cfg, args.source)
    if args.command == "ingest-news":
        return _cmd_ingest_news(cfg, args.source, args.scorer, args.start, args.end)
    if args.command == "fnspid-extract":
        return _cmd_fnspid_extract(cfg, args.skip_download)
    if args.command == "featurize":
        return _cmd_featurize(cfg)
    if args.command == "train":
        return _cmd_train(
            cfg,
            args.model,
            args.horizons,
            features_path=args.features,
            models=args.models,
            artifacts_dir=args.artifacts_dir,
            meta_db=args.meta_db,
            feature_set=args.feature_set,
            report_path=args.report,
        )
    if args.command == "backtest":
        return _cmd_backtest(cfg, args.model)
    if args.command == "serve-api":
        return _cmd_serve_api(cfg, args.host, args.port)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
