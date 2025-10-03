"""
SQLite persistence layer for logging trades, agent metrics, and suspension events.
"""

import os
import sqlite3
import threading
from datetime import datetime
from typing import Any, Dict, Optional


class SQLitePersistence:
    """Lightweight SQLite persistence with simple schemas and thread-safety."""

    def __init__(self, database_url: str):
        # Support URLs like sqlite:///relative_or_absolute_path.db
        self.db_path = self._resolve_db_path(database_url)

        # Ensure parent directory exists
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)

        # Create connection
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self.conn.execute("PRAGMA journal_mode=WAL;")
        self.lock = threading.Lock()

        self._init_schema()

    def _resolve_db_path(self, database_url: str) -> str:
        if database_url.startswith("sqlite:///"):
            path = database_url.replace("sqlite:///", "", 1)
        else:
            path = database_url

        # If path is not absolute, place it under DATA_DIR
        data_dir = os.getenv("DATA_DIR", "/tmp/cta-data")
        os.makedirs(data_dir, exist_ok=True)
        if not os.path.isabs(path):
            path = os.path.join(data_dir, path)

        return path

    def _init_schema(self) -> None:
        with self.lock, self.conn:
            self.conn.execute(
                """
                CREATE TABLE IF NOT EXISTS trades (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    agent_id TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    action TEXT,
                    quantity REAL,
                    price REAL,
                    confidence REAL,
                    reasoning TEXT,
                    status TEXT,
                    alpaca_order_id TEXT,
                    raw_result TEXT
                )
                """
            )

            self.conn.execute(
                """
                CREATE TABLE IF NOT EXISTS agent_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    cycle INTEGER NOT NULL,
                    agent_id TEXT NOT NULL,
                    total_return REAL,
                    sharpe_ratio REAL,
                    max_drawdown REAL,
                    win_rate REAL,
                    profit_factor REAL,
                    total_trades INTEGER,
                    winning_trades INTEGER,
                    losing_trades INTEGER,
                    avg_win REAL,
                    avg_loss REAL
                )
                """
            )

            self.conn.execute(
                """
                CREATE TABLE IF NOT EXISTS suspensions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    agent_id TEXT NOT NULL,
                    event TEXT NOT NULL, -- 'suspend' | 'lift'
                    level INTEGER NOT NULL,
                    reason TEXT,
                    profit_trades_required INTEGER,
                    profit_trades_remaining INTEGER
                )
                """
            )

    def log_trade(self, agent_id: str, decision: Dict[str, Any], trade_result: Optional[Dict[str, Any]]) -> None:
        ts = decision.get("timestamp")
        if isinstance(ts, datetime):
            ts = ts.isoformat()
        status = (trade_result or {}).get("status")
        order_id = (trade_result or {}).get("id") or (trade_result or {}).get("order_id")

        with self.lock, self.conn:
            self.conn.execute(
                """
                INSERT INTO trades (timestamp, agent_id, symbol, action, quantity, price, confidence, reasoning, status, alpaca_order_id, raw_result)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    ts,
                    agent_id,
                    decision.get("symbol"),
                    decision.get("action"),
                    float(decision.get("quantity", 0) or 0),
                    float(decision.get("price", 0) or 0),
                    float(decision.get("confidence", 0) or 0),
                    decision.get("reasoning"),
                    status,
                    order_id,
                    str(trade_result) if trade_result is not None else None,
                ),
            )

    def log_agent_metrics(self, cycle: int, agent_id: str, metrics: Dict[str, Any]) -> None:
        now = datetime.utcnow().isoformat()
        with self.lock, self.conn:
            self.conn.execute(
                """
                INSERT INTO agent_metrics (
                    timestamp, cycle, agent_id, total_return, sharpe_ratio, max_drawdown, win_rate, profit_factor,
                    total_trades, winning_trades, losing_trades, avg_win, avg_loss
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    now,
                    int(cycle),
                    agent_id,
                    float(metrics.get("total_return", 0) or 0),
                    float(metrics.get("sharpe_ratio", 0) or 0),
                    float(metrics.get("max_drawdown", 0) or 0),
                    float(metrics.get("win_rate", 0) or 0),
                    float(metrics.get("profit_factor", 0) or 0),
                    int(metrics.get("total_trades", 0) or 0),
                    int(metrics.get("winning_trades", 0) or 0),
                    int(metrics.get("losing_trades", 0) or 0),
                    float(metrics.get("avg_win", 0) or 0),
                    float(metrics.get("avg_loss", 0) or 0),
                ),
            )

    def log_suspension_event(
        self,
        agent_id: str,
        event: str,
        level: int,
        reason: Optional[str] = None,
        profit_trades_required: Optional[int] = None,
        profit_trades_remaining: Optional[int] = None,
    ) -> None:
        now = datetime.utcnow().isoformat()
        with self.lock, self.conn:
            self.conn.execute(
                """
                INSERT INTO suspensions (
                    timestamp, agent_id, event, level, reason, profit_trades_required, profit_trades_remaining
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    now,
                    agent_id,
                    event,
                    int(level),
                    reason,
                    int(profit_trades_required) if profit_trades_required is not None else None,
                    int(profit_trades_remaining) if profit_trades_remaining is not None else None,
                ),
            )

    def close(self) -> None:
        try:
            with self.lock:
                self.conn.close()
        except Exception:
            pass





