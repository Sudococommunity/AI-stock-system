"""
PostgreSQL database layer — schema definition and CRUD helpers.
Replaces the original SQLite backend for production-scale workloads.

Requires:
    pip install psycopg2-binary

Connection is configured via CONFIG.data.db_url, which defaults to
    postgresql://postgres:postgres@localhost:5432/stock_recommender
Override at runtime with the env var STOCK_RECOMMENDER_DB_URL or by
calling load_config_overrides() before constructing DatabaseManager.

All public methods keep the exact same signatures as the SQLite version
so the rest of the codebase requires zero changes.
"""
import json
import os
import time
import logging
from contextlib import contextmanager
from typing import Any, Dict, List, Optional, Tuple

import psycopg2
import psycopg2.pool
from psycopg2.extras import RealDictCursor, execute_values

from stock_recommender.config import CONFIG

logger = logging.getLogger(__name__)

# ── Schema ────────────────────────────────────────────────────────────────────
# PostgreSQL-native types and syntax.
# BIGSERIAL replaces INTEGER PRIMARY KEY AUTOINCREMENT.
# ON CONFLICT DO NOTHING replaces INSERT OR IGNORE.
# EXTRACT(EPOCH FROM NOW()) replaces unixepoch().
# DOUBLE PRECISION replaces REAL / BLOB (embeddings stay JSON TEXT).

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS stocks (
    stock_id    BIGSERIAL PRIMARY KEY,
    ticker      TEXT             NOT NULL UNIQUE,
    name        TEXT,
    sector      TEXT,
    market_cap  DOUBLE PRECISION,
    is_active   INTEGER          DEFAULT 1,
    created_at  DOUBLE PRECISION DEFAULT EXTRACT(EPOCH FROM NOW())
);

CREATE TABLE IF NOT EXISTS price_history (
    id          BIGSERIAL PRIMARY KEY,
    stock_id    BIGINT           NOT NULL REFERENCES stocks(stock_id),
    date        TEXT             NOT NULL,
    open        DOUBLE PRECISION NOT NULL,
    high        DOUBLE PRECISION NOT NULL,
    low         DOUBLE PRECISION NOT NULL,
    close       DOUBLE PRECISION NOT NULL,
    volume      DOUBLE PRECISION NOT NULL,
    UNIQUE(stock_id, date)
);
CREATE INDEX IF NOT EXISTS idx_price_history_stock_date
    ON price_history(stock_id, date);

CREATE TABLE IF NOT EXISTS users (
    user_id             BIGSERIAL PRIMARY KEY,
    username            TEXT             UNIQUE,
    risk_tolerance      TEXT             DEFAULT 'moderate',
    capital_range       TEXT             DEFAULT 'medium',
    investment_horizon  TEXT             DEFAULT 'medium',
    preferred_sectors   TEXT             DEFAULT '[]',
    created_at          DOUBLE PRECISION DEFAULT EXTRACT(EPOCH FROM NOW())
);

CREATE TABLE IF NOT EXISTS user_events (
    event_id       BIGSERIAL PRIMARY KEY,
    user_id        BIGINT           NOT NULL REFERENCES users(user_id),
    stock_id       BIGINT           NOT NULL REFERENCES stocks(stock_id),
    event_type     TEXT             NOT NULL,
    value          DOUBLE PRECISION,
    price_at_event DOUBLE PRECISION,
    reward         DOUBLE PRECISION,
    timestamp      DOUBLE PRECISION DEFAULT EXTRACT(EPOCH FROM NOW())
);
CREATE INDEX IF NOT EXISTS idx_user_events_user
    ON user_events(user_id, timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_user_events_stock
    ON user_events(stock_id, timestamp DESC);

CREATE TABLE IF NOT EXISTS user_embeddings (
    user_id       BIGINT PRIMARY KEY REFERENCES users(user_id),
    embedding     TEXT             NOT NULL,
    model_version TEXT,
    updated_at    DOUBLE PRECISION DEFAULT EXTRACT(EPOCH FROM NOW())
);

CREATE TABLE IF NOT EXISTS stock_embeddings (
    stock_id      BIGINT PRIMARY KEY REFERENCES stocks(stock_id),
    embedding     TEXT             NOT NULL,
    model_version TEXT,
    updated_at    DOUBLE PRECISION DEFAULT EXTRACT(EPOCH FROM NOW())
);

CREATE TABLE IF NOT EXISTS recommendation_log (
    rec_id        BIGSERIAL PRIMARY KEY,
    user_id       BIGINT           NOT NULL,
    stock_id      BIGINT           NOT NULL,
    rank          INTEGER,
    score         DOUBLE PRECISION,
    model_version TEXT,
    shown_at      DOUBLE PRECISION DEFAULT EXTRACT(EPOCH FROM NOW()),
    clicked       INTEGER          DEFAULT 0,
    acted_on      INTEGER          DEFAULT 0
);

CREATE TABLE IF NOT EXISTS corporate_actions (
    action_id   BIGSERIAL PRIMARY KEY,
    stock_id    BIGINT           NOT NULL REFERENCES stocks(stock_id),
    action_date TEXT,
    action_type TEXT,
    title       TEXT,
    description TEXT,
    source      TEXT,
    raw_payload TEXT,
    created_at  DOUBLE PRECISION DEFAULT EXTRACT(EPOCH FROM NOW()),
    UNIQUE(stock_id, action_date, action_type, title)
);
CREATE INDEX IF NOT EXISTS idx_corporate_actions_stock_date
    ON corporate_actions(stock_id, action_date DESC);

CREATE TABLE IF NOT EXISTS model_checkpoints (
    checkpoint_id BIGSERIAL PRIMARY KEY,
    model_type    TEXT             NOT NULL,
    version       TEXT             NOT NULL,
    file_path     TEXT             NOT NULL,
    metrics       TEXT,
    is_production INTEGER          DEFAULT 0,
    created_at    DOUBLE PRECISION DEFAULT EXTRACT(EPOCH FROM NOW())
);

CREATE TABLE IF NOT EXISTS training_metrics (
    id          BIGSERIAL PRIMARY KEY,
    model_type  TEXT             NOT NULL,
    metric_name TEXT             NOT NULL,
    value       DOUBLE PRECISION NOT NULL,
    step        INTEGER,
    recorded_at DOUBLE PRECISION DEFAULT EXTRACT(EPOCH FROM NOW())
);

CREATE TABLE IF NOT EXISTS fno_snapshots (
    snapshot_id   BIGSERIAL PRIMARY KEY,
    stock_id      BIGINT           NOT NULL REFERENCES stocks(stock_id),
    snapshot_date TEXT             NOT NULL,
    pcr           DOUBLE PRECISION,
    oi_calls      DOUBLE PRECISION,
    oi_puts       DOUBLE PRECISION,
    total_oi      DOUBLE PRECISION,
    delivery_pct  DOUBLE PRECISION,
    created_at    DOUBLE PRECISION DEFAULT EXTRACT(EPOCH FROM NOW()),
    UNIQUE(stock_id, snapshot_date)
);
CREATE INDEX IF NOT EXISTS idx_fno_snapshots_stock_date
    ON fno_snapshots(stock_id, snapshot_date DESC);
"""


def _split_schema(sql: str) -> List[str]:
    """Split a multi-statement SQL string into individual statements."""
    return [s.strip() for s in sql.split(";") if s.strip()]


class DatabaseManager:
    """
    PostgreSQL connection manager with a thread-safe connection pool.
    Drop-in replacement for the original SQLite DatabaseManager —
    every public method signature is identical.
    """

    def __init__(
        self,
        db_url: str = None,
    ):
        url = (
            db_url
            or os.getenv("STOCK_RECOMMENDER_DB_URL")
            or getattr(CONFIG.data, "db_url", None)
            or "postgresql://postgres:postgres@localhost:5432/stock_recommender"
        )
        self.db_url = url
        # min=1  max=10  thread-safe pool
        self._pool = psycopg2.pool.ThreadedConnectionPool(1, 10, url)
        logger.info("[DB] Connected to PostgreSQL: %s", url.split("@")[-1])
        self._ensure_schema()

    def _ensure_schema(self) -> None:
        with self.connection() as conn:
            with conn.cursor() as cur:
                for stmt in _split_schema(SCHEMA_SQL):
                    cur.execute(stmt)

    @contextmanager
    def connection(self):
        """Yield a pooled connection; commit on success, rollback on error."""
        conn = self._pool.getconn()
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            self._pool.putconn(conn)

    def _cur(self, conn) -> RealDictCursor:
        """Return a RealDictCursor so rows behave like dict-like mappings."""
        return conn.cursor(cursor_factory=RealDictCursor)

    # ── Stock operations ──────────────────────────────────────────────────────

    def upsert_stock(
        self, ticker: str, name: str = "", sector: str = "", market_cap: float = 0.0
    ) -> int:
        with self.connection() as conn:
            with self._cur(conn) as cur:
                cur.execute(
                    """
                    INSERT INTO stocks (ticker, name, sector, market_cap)
                    VALUES (%s, %s, %s, %s)
                    ON CONFLICT (ticker) DO NOTHING
                    """,
                    (ticker, name, sector, market_cap),
                )
                cur.execute("SELECT stock_id FROM stocks WHERE ticker=%s", (ticker,))
                return int(cur.fetchone()["stock_id"])

    def get_stock_id(self, ticker: str) -> Optional[int]:
        with self.connection() as conn:
            with self._cur(conn) as cur:
                cur.execute("SELECT stock_id FROM stocks WHERE ticker=%s", (ticker,))
                row = cur.fetchone()
                return int(row["stock_id"]) if row else None

    def get_all_stock_ids(self) -> List[int]:
        with self.connection() as conn:
            with self._cur(conn) as cur:
                cur.execute("SELECT stock_id FROM stocks WHERE is_active=1")
                return [int(r["stock_id"]) for r in cur.fetchall()]

    def get_all_stocks(self) -> List[Dict]:
        with self.connection() as conn:
            with self._cur(conn) as cur:
                cur.execute("SELECT * FROM stocks WHERE is_active=1 ORDER BY ticker")
                return [dict(r) for r in cur.fetchall()]

    def get_stock_info(self, stock_id: int) -> Optional[Dict]:
        with self.connection() as conn:
            with self._cur(conn) as cur:
                cur.execute("SELECT * FROM stocks WHERE stock_id=%s", (stock_id,))
                row = cur.fetchone()
                return dict(row) if row else None

    def get_stock_info_by_ticker(self, ticker: str) -> Optional[Dict]:
        with self.connection() as conn:
            with self._cur(conn) as cur:
                cur.execute("SELECT * FROM stocks WHERE ticker=%s", (ticker,))
                row = cur.fetchone()
                return dict(row) if row else None

    # ── Price history ─────────────────────────────────────────────────────────

    def insert_price_batch(self, stock_id: int, records: List[Dict]) -> None:
        """Bulk-insert OHLCV rows. Uses execute_values for high throughput."""
        if not records:
            return
        rows = [
            (stock_id, r["date"], r["open"], r["high"], r["low"], r["close"], r["volume"])
            for r in records
        ]
        with self.connection() as conn:
            with conn.cursor() as cur:
                execute_values(
                    cur,
                    """
                    INSERT INTO price_history
                        (stock_id, date, open, high, low, close, volume)
                    VALUES %s
                    ON CONFLICT (stock_id, date) DO NOTHING
                    """,
                    rows,
                    page_size=500,
                )

    def get_price_history(self, stock_id: int, limit: int = 500) -> List[Dict]:
        with self.connection() as conn:
            with self._cur(conn) as cur:
                cur.execute(
                    """
                    SELECT date, open, high, low, close, volume
                    FROM price_history
                    WHERE stock_id=%s
                    ORDER BY date DESC
                    LIMIT %s
                    """,
                    (stock_id, limit),
                )
                return [dict(r) for r in reversed(cur.fetchall())]

    def get_latest_price_date(self, stock_id: int) -> Optional[str]:
        with self.connection() as conn:
            with self._cur(conn) as cur:
                cur.execute(
                    "SELECT MAX(date) AS d FROM price_history WHERE stock_id=%s",
                    (stock_id,),
                )
                row = cur.fetchone()
                return row["d"] if row else None

    # ── Users ─────────────────────────────────────────────────────────────────

    def create_user(
        self,
        username: str,
        risk_tolerance: str = "moderate",
        capital_range: str = "medium",
        investment_horizon: str = "medium",
        preferred_sectors: List[str] = None,
    ) -> int:
        with self.connection() as conn:
            with self._cur(conn) as cur:
                cur.execute(
                    """
                    INSERT INTO users
                        (username, risk_tolerance, capital_range,
                         investment_horizon, preferred_sectors)
                    VALUES (%s, %s, %s, %s, %s)
                    ON CONFLICT (username) DO NOTHING
                    """,
                    (
                        username, risk_tolerance, capital_range,
                        investment_horizon, json.dumps(preferred_sectors or []),
                    ),
                )
                cur.execute("SELECT user_id FROM users WHERE username=%s", (username,))
                return int(cur.fetchone()["user_id"])

    def get_user(self, user_id: int) -> Optional[Dict]:
        with self.connection() as conn:
            with self._cur(conn) as cur:
                cur.execute("SELECT * FROM users WHERE user_id=%s", (user_id,))
                row = cur.fetchone()
                if not row:
                    return None
                d = dict(row)
                d["preferred_sectors"] = json.loads(d.get("preferred_sectors") or "[]")
                return d

    def update_user_risk_tolerance(self, user_id: int, risk_tolerance: str) -> None:
        with self.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "UPDATE users SET risk_tolerance=%s WHERE user_id=%s",
                    (risk_tolerance, user_id),
                )

    # ── Events ────────────────────────────────────────────────────────────────

    def log_event(
        self,
        user_id: int,
        stock_id: int,
        event_type: str,
        value: float = 0.0,
        price_at_event: float = 0.0,
        timestamp: Optional[float] = None,
    ) -> int:
        with self.connection() as conn:
            with self._cur(conn) as cur:
                if timestamp is None:
                    cur.execute(
                        """
                        INSERT INTO user_events
                            (user_id, stock_id, event_type, value, price_at_event)
                        VALUES (%s, %s, %s, %s, %s)
                        RETURNING event_id
                        """,
                        (user_id, stock_id, event_type, value, price_at_event),
                    )
                else:
                    cur.execute(
                        """
                        INSERT INTO user_events
                            (user_id, stock_id, event_type, value, price_at_event, timestamp)
                        VALUES (%s, %s, %s, %s, %s, %s)
                        RETURNING event_id
                        """,
                        (user_id, stock_id, event_type, value, price_at_event, float(timestamp)),
                    )
                return int(cur.fetchone()["event_id"])

    def get_user_events(
        self,
        user_id: int,
        event_types: Optional[List[str]] = None,
        limit: int = 200,
    ) -> List[Dict]:
        with self.connection() as conn:
            with self._cur(conn) as cur:
                if event_types:
                    cur.execute(
                        """
                        SELECT * FROM user_events
                        WHERE user_id=%s AND event_type = ANY(%s)
                        ORDER BY timestamp DESC
                        LIMIT %s
                        """,
                        (user_id, event_types, limit),
                    )
                else:
                    cur.execute(
                        """
                        SELECT * FROM user_events
                        WHERE user_id=%s
                        ORDER BY timestamp DESC
                        LIMIT %s
                        """,
                        (user_id, limit),
                    )
                return [dict(r) for r in cur.fetchall()]

    def update_event_reward(self, event_id: int, reward: float) -> None:
        with self.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "UPDATE user_events SET reward=%s WHERE event_id=%s",
                    (reward, event_id),
                )

    def get_unresolved_events(self, older_than_seconds: int = 5 * 86400) -> List[Dict]:
        cutoff = time.time() - older_than_seconds
        with self.connection() as conn:
            with self._cur(conn) as cur:
                cur.execute(
                    """
                    SELECT * FROM user_events
                    WHERE reward IS NULL
                      AND timestamp < %s
                      AND event_type IN ('view', 'watchlist_add', 'rate')
                    LIMIT 1000
                    """,
                    (cutoff,),
                )
                return [dict(r) for r in cur.fetchall()]

    # ── Embeddings ────────────────────────────────────────────────────────────

    def save_user_embedding(
        self, user_id: int, embedding: List[float], version: str = ""
    ) -> None:
        with self.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO user_embeddings (user_id, embedding, model_version, updated_at)
                    VALUES (%s, %s, %s, %s)
                    ON CONFLICT (user_id) DO UPDATE SET
                        embedding     = EXCLUDED.embedding,
                        model_version = EXCLUDED.model_version,
                        updated_at    = EXCLUDED.updated_at
                    """,
                    (user_id, json.dumps(embedding), version, time.time()),
                )

    def get_user_embedding(self, user_id: int) -> Optional[List[float]]:
        with self.connection() as conn:
            with self._cur(conn) as cur:
                cur.execute(
                    "SELECT embedding FROM user_embeddings WHERE user_id=%s", (user_id,)
                )
                row = cur.fetchone()
                return json.loads(row["embedding"]) if row else None

    def save_stock_embedding(
        self, stock_id: int, embedding: List[float], version: str = ""
    ) -> None:
        with self.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO stock_embeddings (stock_id, embedding, model_version, updated_at)
                    VALUES (%s, %s, %s, %s)
                    ON CONFLICT (stock_id) DO UPDATE SET
                        embedding     = EXCLUDED.embedding,
                        model_version = EXCLUDED.model_version,
                        updated_at    = EXCLUDED.updated_at
                    """,
                    (stock_id, json.dumps(embedding), version, time.time()),
                )

    def get_stock_embedding(self, stock_id: int) -> Optional[List[float]]:
        with self.connection() as conn:
            with self._cur(conn) as cur:
                cur.execute(
                    "SELECT embedding FROM stock_embeddings WHERE stock_id=%s", (stock_id,)
                )
                row = cur.fetchone()
                return json.loads(row["embedding"]) if row else None

    def get_all_stock_embeddings(self) -> Dict[int, List[float]]:
        with self.connection() as conn:
            with self._cur(conn) as cur:
                cur.execute("SELECT stock_id, embedding FROM stock_embeddings")
                return {
                    int(r["stock_id"]): json.loads(r["embedding"])
                    for r in cur.fetchall()
                }

    # ── Recommendations log ───────────────────────────────────────────────────

    def log_recommendation(
        self,
        user_id: int,
        stock_id: int,
        rank: int,
        score: float,
        version: str = "",
    ) -> None:
        with self.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO recommendation_log
                        (user_id, stock_id, rank, score, model_version)
                    VALUES (%s, %s, %s, %s, %s)
                    """,
                    (user_id, stock_id, rank, score, version),
                )

    def mark_recommendation_clicked(self, user_id: int, stock_id: int) -> None:
        with self.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    UPDATE recommendation_log SET clicked=1
                    WHERE rec_id = (
                        SELECT rec_id FROM recommendation_log
                        WHERE user_id=%s AND stock_id=%s AND shown_at > %s
                        ORDER BY shown_at DESC
                        LIMIT 1
                    )
                    """,
                    (user_id, stock_id, time.time() - 86400),
                )

    # ── Corporate actions ─────────────────────────────────────────────────────

    def insert_corporate_actions(self, stock_id: int, actions: List[Dict]) -> int:
        if not actions:
            return 0
        inserted = 0
        with self.connection() as conn:
            with conn.cursor() as cur:
                for action in actions:
                    cur.execute(
                        """
                        INSERT INTO corporate_actions
                            (stock_id, action_date, action_type, title,
                             description, source, raw_payload)
                        VALUES (%s, %s, %s, %s, %s, %s, %s)
                        ON CONFLICT (stock_id, action_date, action_type, title) DO NOTHING
                        """,
                        (
                            stock_id,
                            action.get("action_date"),
                            action.get("action_type"),
                            action.get("title"),
                            action.get("description"),
                            action.get("source", ""),
                            json.dumps(action.get("raw_payload", {})),
                        ),
                    )
                    inserted += cur.rowcount
        return inserted

    def get_corporate_actions(self, stock_id: int, limit: int = 100) -> List[Dict]:
        with self.connection() as conn:
            with self._cur(conn) as cur:
                cur.execute(
                    """
                    SELECT action_id, action_date, action_type, title,
                           description, source, raw_payload
                    FROM corporate_actions
                    WHERE stock_id=%s
                    ORDER BY action_date DESC
                    LIMIT %s
                    """,
                    (stock_id, limit),
                )
                result = []
                for row in cur.fetchall():
                    item = dict(row)
                    item["raw_payload"] = json.loads(item.get("raw_payload") or "{}")
                    result.append(item)
                return result

    def mark_corporate_action_applied(self, action_id: int) -> None:
        """Mark a corporate action as price-adjusted so it is not re-applied."""
        with self.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "UPDATE corporate_actions SET source = source || ':adjusted' "
                    "WHERE action_id=%s AND source NOT LIKE '%%:adjusted'",
                    (action_id,),
                )

    def replace_price_history(self, stock_id: int, records: List[Dict]) -> None:
        """
        Overwrite all price_history rows for a stock with the supplied records.
        Used after applying corporate action adjustments.
        """
        with self.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "DELETE FROM price_history WHERE stock_id=%s", (stock_id,)
                )
            with conn.cursor() as cur:
                execute_values(
                    cur,
                    """
                    INSERT INTO price_history
                        (stock_id, date, open, high, low, close, volume)
                    VALUES %s
                    """,
                    [
                        (stock_id, r["date"], r["open"], r["high"],
                         r["low"], r["close"], r["volume"])
                        for r in records
                    ],
                    page_size=500,
                )

    # ── F&O snapshots ─────────────────────────────────────────────────────────

    def upsert_fno_snapshot(
        self,
        stock_id: int,
        snapshot_date: str,
        pcr: Optional[float] = None,
        oi_calls: Optional[float] = None,
        oi_puts: Optional[float] = None,
        total_oi: Optional[float] = None,
        delivery_pct: Optional[float] = None,
    ) -> None:
        with self.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO fno_snapshots
                        (stock_id, snapshot_date, pcr, oi_calls, oi_puts, total_oi, delivery_pct)
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (stock_id, snapshot_date) DO UPDATE SET
                        pcr          = EXCLUDED.pcr,
                        oi_calls     = EXCLUDED.oi_calls,
                        oi_puts      = EXCLUDED.oi_puts,
                        total_oi     = EXCLUDED.total_oi,
                        delivery_pct = EXCLUDED.delivery_pct
                    """,
                    (stock_id, snapshot_date, pcr, oi_calls, oi_puts, total_oi, delivery_pct),
                )

    def get_fno_snapshots(self, stock_id: int, limit: int = 60) -> List[Dict]:
        with self.connection() as conn:
            with self._cur(conn) as cur:
                cur.execute(
                    """
                    SELECT snapshot_date, pcr, oi_calls, oi_puts, total_oi, delivery_pct
                    FROM fno_snapshots
                    WHERE stock_id=%s
                    ORDER BY snapshot_date DESC
                    LIMIT %s
                    """,
                    (stock_id, limit),
                )
                return [dict(r) for r in reversed(cur.fetchall())]

    # ── Metrics ───────────────────────────────────────────────────────────────

    def log_metric(
        self, model_type: str, metric_name: str, value: float, step: int = 0
    ) -> None:
        with self.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO training_metrics (model_type, metric_name, value, step)
                    VALUES (%s, %s, %s, %s)
                    """,
                    (model_type, metric_name, value, step),
                )

    def get_metrics(
        self, model_type: str, metric_name: str, last_n: int = 100
    ) -> List[Tuple]:
        with self.connection() as conn:
            with self._cur(conn) as cur:
                cur.execute(
                    """
                    SELECT step, value, recorded_at
                    FROM training_metrics
                    WHERE model_type=%s AND metric_name=%s
                    ORDER BY recorded_at DESC
                    LIMIT %s
                    """,
                    (model_type, metric_name, last_n),
                )
                return [(r["step"], r["value"], r["recorded_at"]) for r in cur.fetchall()]

    def close(self) -> None:
        """Return all connections to the pool and close it."""
        self._pool.closeall()
        logger.info("[DB] Connection pool closed.")
