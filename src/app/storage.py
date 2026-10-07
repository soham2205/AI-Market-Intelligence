from __future__ import annotations

import json
import sqlite3
import uuid
from pathlib import Path


def _connect(db_path: str | Path) -> sqlite3.Connection:
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS runs (
            run_id TEXT PRIMARY KEY,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            model_name TEXT,
            config_json TEXT,
            metrics_json TEXT,
            artifact_path TEXT,
            horizon INTEGER,
            objective TEXT,
            batch_id TEXT
        )
        """
    )
    # Migrate pre-multi-horizon databases: the columns are additive and stay
    # NULL on legacy rows, which readers interpret as horizon=1/classification.
    existing = {r[1] for r in conn.execute("PRAGMA table_info(runs)")}
    for col, decl in (
        ("horizon", "INTEGER"),
        ("objective", "TEXT"),
        ("batch_id", "TEXT"),
    ):
        if col not in existing:
            conn.execute(f"ALTER TABLE runs ADD COLUMN {col} {decl}")
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS backtests (
            backtest_id TEXT PRIMARY KEY,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            run_id TEXT,
            params_json TEXT,
            metrics_json TEXT,
            equity_curve_json TEXT
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS champion (
            id INTEGER PRIMARY KEY CHECK (id = 1),
            run_id TEXT NOT NULL,
            model_name TEXT NOT NULL,
            metric_name TEXT NOT NULL,
            metric_value REAL NOT NULL,
            promoted_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    # Per-horizon champions. The legacy `champion` table carries
    # CHECK (id = 1), which SQLite cannot drop, so this is a new table; the
    # old one is left in place read-only for migration.
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS champions (
            horizon      INTEGER NOT NULL,
            objective    TEXT NOT NULL DEFAULT 'classification',
            run_id       TEXT NOT NULL,
            model_name   TEXT NOT NULL,
            metric_name  TEXT NOT NULL,
            metric_value REAL NOT NULL,
            promoted_at  TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (horizon, objective)
        )
        """
    )
    return conn


def record_run(
    db_path: str | Path,
    model_name: str,
    config: dict,
    metrics: dict,
    artifact_path: str,
    run_id: str | None = None,
    horizon: int = 1,
    objective: str = "classification",
    batch_id: str | None = None,
) -> str:
    run_id = run_id or uuid.uuid4().hex[:12]
    with _connect(db_path) as conn:
        conn.execute(
            "INSERT INTO runs (run_id, model_name, config_json, metrics_json,"
            " artifact_path, horizon, objective, batch_id)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                run_id,
                model_name,
                json.dumps(config, default=str),
                json.dumps(metrics, default=str),
                artifact_path,
                int(horizon),
                objective,
                batch_id,
            ),
        )
    return run_id


def record_backtest(
    db_path: str | Path,
    run_id: str,
    params: dict,
    metrics: dict,
    equity_curve: dict,
) -> str:
    backtest_id = uuid.uuid4().hex[:12]
    with _connect(db_path) as conn:
        conn.execute(
            "INSERT INTO backtests"
            " (backtest_id, run_id, params_json, metrics_json, equity_curve_json)"
            " VALUES (?, ?, ?, ?, ?)",
            (
                backtest_id,
                run_id,
                json.dumps(params, default=str),
                json.dumps(metrics, default=str),
                json.dumps(equity_curve, default=str),
            ),
        )
    return backtest_id


def latest_run(
    db_path: str | Path,
    model_name: str | None = None,
    horizon: int | None = None,
) -> dict | None:
    query = "SELECT * FROM runs"
    clauses: list[str] = []
    params: tuple = ()
    if model_name:
        clauses.append("model_name = ?")
        params += (model_name,)
    if horizon is not None:
        # Legacy rows predate the column; they were all 1-day runs.
        clauses.append("(horizon = ? OR (horizon IS NULL AND ? = 1))")
        params += (int(horizon), int(horizon))
    if clauses:
        query += " WHERE " + " AND ".join(clauses)
    query += " ORDER BY created_at DESC LIMIT 1"
    with _connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute(query, params).fetchone()
    return dict(row) if row else None


def set_champion(
    db_path: str | Path,
    run_id: str,
    model_name: str,
    metric_value: float,
    metric_name: str = "roc_auc_mean",
    horizon: int = 1,
    objective: str = "classification",
) -> bool:
    """Promote a run to champion FOR ITS HORIZON if it beats that horizon's
    incumbent. Horizons compete only against themselves -- a 252-day model
    and a 1-day model are different questions, not rival answers."""
    champion = get_champion(db_path, horizon=horizon, objective=objective)
    if champion is not None and metric_value <= float(champion["metric_value"]):
        return False
    with _connect(db_path) as conn:
        conn.execute(
            "INSERT OR REPLACE INTO champions (horizon, objective, run_id, model_name,"
            " metric_name, metric_value, promoted_at)"
            " VALUES (?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)",
            (
                int(horizon),
                objective,
                run_id,
                model_name,
                metric_name,
                float(metric_value),
            ),
        )
    return True


def get_champion(
    db_path: str | Path,
    horizon: int = 1,
    objective: str = "classification",
) -> dict | None:
    with _connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            "SELECT * FROM champions WHERE horizon = ? AND objective = ?",
            (int(horizon), objective),
        ).fetchone()
    return dict(row) if row else None


def all_champions(db_path: str | Path) -> list[dict]:
    with _connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT * FROM champions ORDER BY objective, horizon"
        ).fetchall()
    return [dict(r) for r in rows]
