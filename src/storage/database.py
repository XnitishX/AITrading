"""
Strategy Storage – SQLite Backend
──────────────────────────────────
Persists saved strategies, backtest results, and copilot sessions
so they survive across application restarts.
"""

import json
import logging
import sqlite3
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Optional

from config.settings import DATA_DIR

logger = logging.getLogger(__name__)

DB_PATH = DATA_DIR / "aitrading.db"

# ── Schema ────────────────────────────────────────────────────────────────

_SCHEMA = """
CREATE TABLE IF NOT EXISTS strategies (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    name        TEXT NOT NULL UNIQUE,
    type        TEXT NOT NULL,          -- sma_crossover | rsi_mean_reversion | vix_regime | custom
    params      TEXT NOT NULL DEFAULT '{}',  -- JSON blob
    description TEXT DEFAULT '',
    created_at  TEXT NOT NULL,
    updated_at  TEXT NOT NULL,
    is_active   INTEGER NOT NULL DEFAULT 1
);

CREATE TABLE IF NOT EXISTS backtest_runs (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    strategy_id     INTEGER,
    strategy_name   TEXT NOT NULL,
    params          TEXT NOT NULL DEFAULT '{}',
    initial_capital REAL NOT NULL,
    final_capital   REAL NOT NULL,
    total_return    REAL,
    annual_return   REAL,
    sharpe_ratio    REAL,
    sortino_ratio   REAL,
    calmar_ratio    REAL,
    max_drawdown    REAL,
    total_trades    INTEGER,
    win_rate        REAL,
    avg_pnl         REAL,
    profit_factor   REAL,
    max_consecutive_losses INTEGER DEFAULT 0,
    exposure_time    REAL DEFAULT 0,
    benchmark_return REAL DEFAULT 0,
    alpha           REAL DEFAULT 0,
    risk_reward_ratio REAL DEFAULT 0,
    max_dd_duration INTEGER DEFAULT 0,
    avg_trade_duration REAL DEFAULT 0,
    information_ratio REAL DEFAULT 0,
    equity_json     TEXT,             -- serialised equity curve (sampled)
    trades_json     TEXT,             -- serialised trade list
    run_at          TEXT NOT NULL,
    data_start      TEXT,
    data_end        TEXT,
    FOREIGN KEY (strategy_id) REFERENCES strategies(id)
);

CREATE TABLE IF NOT EXISTS copilot_sessions (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    messages    TEXT NOT NULL DEFAULT '[]', -- JSON array of {role, content}
    created_at  TEXT NOT NULL,
    updated_at  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS market_events (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    name        TEXT NOT NULL,
    category    TEXT NOT NULL,
    start_date  TEXT NOT NULL,
    end_date    TEXT,
    description TEXT DEFAULT '',
    source      TEXT DEFAULT '',
    region      TEXT DEFAULT 'global',
    impact      TEXT DEFAULT 'medium',
    tags        TEXT DEFAULT '[]',       -- JSON array of tag strings
    created_at  TEXT NOT NULL
);
"""


# ── Connection Helper ─────────────────────────────────────────────────────

@contextmanager
def _get_db():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db():
    """Create tables if they don't exist and seed default strategies."""
    with _get_db() as conn:
        conn.executescript(_SCHEMA)
        # Migrate: add new columns to existing tables if missing
        _migrate_backtest_runs(conn)
        logger.info("Database initialised at %s", DB_PATH)
        _seed_defaults(conn)


def _migrate_backtest_runs(conn: sqlite3.Connection):
    """Add columns introduced after initial schema to existing tables."""
    cursor = conn.execute("PRAGMA table_info(backtest_runs)")
    existing_cols = {row["name"] for row in cursor.fetchall()}
    migrations = [
        ("sortino_ratio", "REAL DEFAULT 0"),
        ("calmar_ratio", "REAL DEFAULT 0"),
        ("max_consecutive_losses", "INTEGER DEFAULT 0"),
        ("exposure_time", "REAL DEFAULT 0"),
        ("benchmark_return", "REAL DEFAULT 0"),
        ("alpha", "REAL DEFAULT 0"),
        ("information_ratio", "REAL DEFAULT 0"),
        ("risk_reward_ratio", "REAL DEFAULT 0"),
        ("max_dd_duration", "INTEGER DEFAULT 0"),
        ("avg_trade_duration", "REAL DEFAULT 0"),
    ]
    for col_name, col_def in migrations:
        if col_name not in existing_cols:
            try:
                conn.execute(f"ALTER TABLE backtest_runs ADD COLUMN {col_name} {col_def}")
                logger.info("Migrated backtest_runs: added column %s", col_name)
            except sqlite3.OperationalError:
                pass  # column already exists (race condition guard)


def _seed_defaults(conn: sqlite3.Connection):
    """Insert the six core strategies with Investopedia-standard defaults."""
    defaults = [
        {
            "name": "SMA Crossover (50/200)",
            "type": "sma_crossover",
            "params": {"fast_window": 50, "slow_window": 200},
            "description": "Golden Cross / Death Cross — 50-day vs 200-day SMA crossover.",
        },
        {
            "name": "RSI Mean Reversion",
            "type": "rsi_mean_reversion",
            "params": {"oversold": 30, "overbought": 70},
            "description": "Wilder 14-period RSI; buy < 30 oversold, sell > 70 overbought.",
        },
        {
            "name": "MACD Crossover",
            "type": "macd_crossover",
            "params": {"fast_ema": 12, "slow_ema": 26, "signal_period": 9},
            "description": "Appel MACD 12/26/9 — trade on signal-line crossover.",
        },
        {
            "name": "Bollinger Bands",
            "type": "bollinger_band",
            "params": {"window": 20, "num_std": 2.0},
            "description": "20-day SMA ± 2σ Bollinger Bands mean reversion.",
        },
        {
            "name": "ATR Breakout",
            "type": "atr_breakout",
            "params": {"sma_window": 20, "atr_period": 14, "atr_multiplier": 1.5},
            "description": "Breakout beyond 20-SMA ± 1.5× Wilder 14-day ATR.",
        },
        {
            "name": "VIX Regime",
            "type": "vix_regime",
            "params": {"buy_below": 15, "sell_above": 25},
            "description": "India VIX regime filter — long when calm (< 15), exit when fearful (> 25).",
        },
        {
            "name": "EMA Crossover (12/26)",
            "type": "ema_crossover",
            "params": {"fast_span": 12, "slow_span": 26},
            "description": "EMA Crossover — buy when fast EMA crosses above slow EMA, faster than SMA.",
        },
        {
            "name": "Stochastic Oscillator",
            "type": "stochastic_oscillator",
            "params": {"k_period": 14, "d_period": 3, "oversold": 20, "overbought": 80},
            "description": "George Lane Stochastic %K/%D crossover with overbought/oversold zones.",
        },
        {
            "name": "Mean Reversion Z-Score",
            "type": "mean_reversion_zscore",
            "params": {"lookback": 20, "entry_z": -2.0, "exit_z": 0.0},
            "description": "Buy when z-score falls below -2σ (deeply oversold), sell above +2σ.",
        },
        {
            "name": "MACD Histogram",
            "type": "macd_histogram",
            "params": {"fast_ema": 12, "slow_ema": 26, "signal_period": 9},
            "description": "MACD Histogram reversal — trade when histogram flips sign (Aspray 1986).",
        },
        {
            "name": "Composite Sniper",
            "type": "composite_sniper",
            "params": {
                "sma_period": 50, "rsi_period": 14,
                "rsi_oversold": 35, "rsi_overbought": 65,
                "vix_calm_threshold": 20,
                "bb_window": 20, "bb_num_std": 2.0,
            },
            "description": "Multi-factor sniper: trend (SMA) + pullback (RSI) + value (BB) + regime (VIX). Fewer but higher-conviction trades.",
        },
    ]
    now = datetime.utcnow().isoformat()
    for d in defaults:
        existing = conn.execute("SELECT id FROM strategies WHERE name = ?", (d["name"],)).fetchone()
        if not existing:
            conn.execute(
                "INSERT INTO strategies (name, type, params, description, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?)",
                (d["name"], d["type"], json.dumps(d["params"]), d["description"], now, now),
            )
    logger.info("Default strategies seeded.")


# ── Strategy CRUD ─────────────────────────────────────────────────────────

def list_strategies() -> list[dict]:
    with _get_db() as conn:
        rows = conn.execute("SELECT * FROM strategies ORDER BY id").fetchall()
        return [dict(r) for r in rows]


def get_strategy(strategy_id: int) -> Optional[dict]:
    with _get_db() as conn:
        row = conn.execute("SELECT * FROM strategies WHERE id = ?", (strategy_id,)).fetchone()
        return dict(row) if row else None


def get_strategy_by_name(name: str) -> Optional[dict]:
    with _get_db() as conn:
        row = conn.execute("SELECT * FROM strategies WHERE name = ?", (name,)).fetchone()
        return dict(row) if row else None


def create_strategy(name: str, stype: str, params: dict, description: str = "") -> dict:
    now = datetime.utcnow().isoformat()
    with _get_db() as conn:
        cur = conn.execute(
            "INSERT INTO strategies (name, type, params, description, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?)",
            (name, stype, json.dumps(params), description, now, now),
        )
        # Read back using the SAME connection (commit hasn't happened yet)
        row = conn.execute("SELECT * FROM strategies WHERE id = ?", (cur.lastrowid,)).fetchone()
        return dict(row) if row else {}


def update_strategy(strategy_id: int, name: str = None, params: dict = None, description: str = None, is_active: int = None) -> Optional[dict]:
    now = datetime.utcnow().isoformat()
    with _get_db() as conn:
        # Read current state within the SAME connection/transaction
        row = conn.execute("SELECT * FROM strategies WHERE id = ?", (strategy_id,)).fetchone()
        if not row:
            return None
        strat = dict(row)
        conn.execute(
            "UPDATE strategies SET name=?, params=?, description=?, is_active=?, updated_at=? WHERE id=?",
            (
                name or strat["name"],
                json.dumps(params) if params else strat["params"],
                description if description is not None else strat["description"],
                is_active if is_active is not None else strat["is_active"],
                now,
                strategy_id,
            ),
        )
        # Read back updated row within the same transaction
        updated = conn.execute("SELECT * FROM strategies WHERE id = ?", (strategy_id,)).fetchone()
        return dict(updated) if updated else None


def delete_strategy(strategy_id: int) -> bool:
    with _get_db() as conn:
        cur = conn.execute("DELETE FROM strategies WHERE id = ?", (strategy_id,))
        return cur.rowcount > 0


# ── Backtest Run Storage ─────────────────────────────────────────────────

def save_backtest_run(
    strategy_id: Optional[int],
    strategy_name: str,
    params: dict,
    result_dict: dict,
    equity_json: str = "[]",
    trades_json: str = "[]",
    data_start: str = "",
    data_end: str = "",
) -> int:
    now = datetime.utcnow().isoformat()
    with _get_db() as conn:
        cur = conn.execute(
            """INSERT INTO backtest_runs
               (strategy_id, strategy_name, params, initial_capital, final_capital,
                total_return, annual_return, sharpe_ratio, sortino_ratio, calmar_ratio,
                max_drawdown, total_trades, win_rate, avg_pnl, profit_factor,
                max_consecutive_losses, risk_reward_ratio, max_dd_duration,
                avg_trade_duration,
                exposure_time, benchmark_return, alpha,
                information_ratio,
                equity_json, trades_json, run_at, data_start, data_end)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                strategy_id,
                strategy_name,
                json.dumps(params),
                result_dict["initial_capital"],
                result_dict["final_capital"],
                result_dict["total_return_pct"],
                result_dict["annual_return_pct"],
                result_dict["sharpe_ratio"],
                result_dict.get("sortino_ratio", 0),
                result_dict.get("calmar_ratio", 0),
                result_dict["max_drawdown_pct"],
                result_dict["total_trades"],
                result_dict["win_rate_pct"],
                result_dict["avg_trade_pnl"],
                result_dict["profit_factor"],
                result_dict.get("max_consecutive_losses", 0),
                result_dict.get("risk_reward_ratio", 0),
                result_dict.get("max_drawdown_duration", 0),
                result_dict.get("avg_trade_duration", 0),
                result_dict.get("exposure_time_pct", 0),
                result_dict.get("benchmark_return_pct", 0),
                result_dict.get("alpha_pct", 0),
                result_dict.get("information_ratio", 0),
                equity_json,
                trades_json,
                now,
                data_start,
                data_end,
            ),
        )
        return cur.lastrowid


def list_backtest_runs(limit: int = 50, offset: int = 0) -> list[dict]:
    with _get_db() as conn:
        rows = conn.execute(
            "SELECT * FROM backtest_runs ORDER BY id DESC LIMIT ? OFFSET ?", (limit, offset)
        ).fetchall()
        return [dict(r) for r in rows]


def count_backtest_runs() -> int:
    """Return the total number of backtest runs."""
    with _get_db() as conn:
        row = conn.execute("SELECT COUNT(*) as cnt FROM backtest_runs").fetchone()
        return row["cnt"] if row else 0


def get_backtest_run(run_id: int) -> Optional[dict]:
    with _get_db() as conn:
        row = conn.execute("SELECT * FROM backtest_runs WHERE id = ?", (run_id,)).fetchone()
        return dict(row) if row else None


def delete_backtest_run(run_id: int) -> bool:
    """Delete a single backtest run by ID. Returns True if deleted."""
    with _get_db() as conn:
        cur = conn.execute("DELETE FROM backtest_runs WHERE id = ?", (run_id,))
        return cur.rowcount > 0


def delete_all_backtest_runs() -> int:
    """Delete all backtest runs. Returns number of rows deleted."""
    with _get_db() as conn:
        cur = conn.execute("DELETE FROM backtest_runs")
        return cur.rowcount


# ── Copilot Sessions ─────────────────────────────────────────────────────

def create_copilot_session() -> int:
    now = datetime.utcnow().isoformat()
    with _get_db() as conn:
        cur = conn.execute(
            "INSERT INTO copilot_sessions (messages, created_at, updated_at) VALUES (?, ?, ?)",
            ("[]", now, now),
        )
        return cur.lastrowid


def get_copilot_session(session_id: int) -> Optional[dict]:
    with _get_db() as conn:
        row = conn.execute("SELECT * FROM copilot_sessions WHERE id = ?", (session_id,)).fetchone()
        if row:
            d = dict(row)
            d["messages"] = json.loads(d["messages"])
            return d
        return None


def update_copilot_session(session_id: int, messages: list[dict]):
    now = datetime.utcnow().isoformat()
    with _get_db() as conn:
        conn.execute(
            "UPDATE copilot_sessions SET messages=?, updated_at=? WHERE id=?",
            (json.dumps(messages), now, session_id),
        )


# ── Market Events ─────────────────────────────────────────────────────────

def sync_events_to_db():
    """Sync the curated event catalog from src.data.events into the SQLite table."""
    from src.data.events import get_all_events
    now = datetime.utcnow().isoformat()
    events = get_all_events()
    with _get_db() as conn:
        conn.execute("DELETE FROM market_events")  # full refresh
        for e in events:
            conn.execute(
                """INSERT INTO market_events
                   (name, category, start_date, end_date, description, source, region, impact, tags, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    e.name, e.category,
                    str(e.start_date), str(e.end_date) if e.end_date else None,
                    e.description, e.source, e.region, e.impact,
                    json.dumps(e.tags), now,
                ),
            )
        logger.info("Synced %d curated events to market_events table", len(events))


def list_market_events(category: str = None, region: str = None, limit: int = 200) -> list[dict]:
    """List market events, optionally filtered by category/region."""
    with _get_db() as conn:
        query = "SELECT * FROM market_events WHERE 1=1"
        params = []
        if category:
            query += " AND category = ?"
            params.append(category)
        if region:
            query += " AND (region = ? OR region = 'both')"
            params.append(region)
        query += " ORDER BY start_date DESC LIMIT ?"
        params.append(limit)
        rows = conn.execute(query, params).fetchall()
        result = []
        for r in rows:
            d = dict(r)
            d["tags"] = json.loads(d.get("tags", "[]"))
            result.append(d)
        return result
