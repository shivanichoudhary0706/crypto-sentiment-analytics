"""
TimescaleDB I/O for the feature job: schema, reads, idempotent upsert,
run bookkeeping and a single-writer lock. No feature logic lives here.
"""
from __future__ import annotations

import logging
from datetime import datetime

import pandas as pd
from psycopg2 import sql
from psycopg2.extras import execute_values

from features.transforms import CANDLE_COLUMNS, FEATURE_COLUMNS, SCORE_COLUMNS

logger = logging.getLogger(__name__)

JOB_NAME = "feature_job"
# pg_advisory_lock key: any constant bigint unique to this job.
_LOCK_KEY = 7_042_026_001

_INT_COLS = {"n_candles", "n_articles", "n_positive", "n_negative",
             "articles_roll_short", "articles_roll_long"}
_TEXT_COLS = {"symbol", "coin", "freq"}


def _col_type(col: str) -> str:
    if col == "bucket":
        return "TIMESTAMPTZ NOT NULL"
    if col in _TEXT_COLS:
        return "TEXT NOT NULL"
    if col == "is_complete":
        return "BOOLEAN NOT NULL"
    if col in _INT_COLS:
        return "INTEGER NOT NULL" if col in {"n_candles", "n_articles"} else "INTEGER"
    return "DOUBLE PRECISION"


def ensure_schema(conn, table: str) -> None:
    """Create the features hypertable and job_runs table. Safe to re-run."""
    cols = sql.SQL(",\n    ").join(
        sql.SQL("{} {}").format(sql.Identifier(c), sql.SQL(_col_type(c))) for c in FEATURE_COLUMNS
    )
    ddl = sql.SQL("""
CREATE TABLE IF NOT EXISTS {table} (
    {cols},
    computed_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (symbol, freq, bucket)
);
CREATE INDEX IF NOT EXISTS {idx} ON {table} (freq, bucket DESC);
CREATE TABLE IF NOT EXISTS job_runs (
    id           BIGSERIAL PRIMARY KEY,
    job_name     TEXT        NOT NULL,
    started_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
    finished_at  TIMESTAMPTZ,
    status       TEXT        NOT NULL,      -- running | success | failed
    mode         TEXT,                      -- full | incremental | pinned
    write_from   TIMESTAMPTZ,
    rows_written INTEGER,
    details      TEXT
);
""").format(table=sql.Identifier(table), cols=cols, idx=sql.Identifier(f"{table}_freq_bucket_idx"))

    # Hypertable only when TimescaleDB is present (plain Postgres in unit tests).
    hyper = sql.SQL("""
DO $$
BEGIN
    IF EXISTS (SELECT 1 FROM pg_extension WHERE extname = 'timescaledb') THEN
        PERFORM create_hypertable({tbl}, 'bucket', if_not_exists => TRUE,
                                  chunk_time_interval => INTERVAL '30 days');
    END IF;
END $$;
""").format(tbl=sql.Literal(table))

    with conn.cursor() as cur:
        cur.execute(ddl)
        cur.execute(hyper)
    conn.commit()


# ------------------------------------------------------------------ locking
def try_lock(conn) -> bool:
    """Only one feature job may write at a time (e.g. the --every loop plus a
    manual --full-refresh). Session-level lock: released on unlock or disconnect."""
    with conn.cursor() as cur:
        cur.execute("SELECT pg_try_advisory_lock(%s)", (_LOCK_KEY,))
        ok = cur.fetchone()[0]
    conn.commit()
    return bool(ok)


def unlock(conn) -> None:
    with conn.cursor() as cur:
        cur.execute("SELECT pg_advisory_unlock(%s)", (_LOCK_KEY,))
    conn.commit()


# ------------------------------------------------------------------ reads
def fetch_watermarks(conn, table: str) -> dict[str, datetime]:
    """Latest bar already stored, per frequency."""
    q = sql.SQL("SELECT freq, max(bucket) FROM {} GROUP BY freq").format(sql.Identifier(table))
    with conn.cursor() as cur:
        cur.execute(q)
        rows = cur.fetchall()
    conn.commit()
    return {freq: ts for freq, ts in rows if ts is not None}


def _time_filter(start: datetime | None, end: datetime | None) -> tuple[sql.Composable, dict]:
    parts, params = [], {}
    if start is not None:
        parts.append(sql.SQL("time >= %(start)s"))
        params["start"] = start
    if end is not None:
        parts.append(sql.SQL("time < %(end)s"))
        params["end"] = end
    clause = sql.SQL(" AND ").join(parts) if parts else sql.SQL("TRUE")
    return clause, params


def fetch_candles(conn, symbols: tuple[str, ...], start: datetime | None,
                  end: datetime | None) -> pd.DataFrame:
    where, params = _time_filter(start, end)
    params["symbols"] = list(symbols)
    q = sql.SQL("""
SELECT time, symbol,
       open::double precision, high::double precision, low::double precision,
       close::double precision, volume::double precision
FROM market_data
WHERE symbol = ANY(%(symbols)s) AND {where}
ORDER BY symbol, time
""").format(where=where)
    with conn.cursor() as cur:
        cur.execute(q, params)
        rows = cur.fetchall()
    conn.commit()
    df = pd.DataFrame(rows, columns=list(CANDLE_COLUMNS))
    df["time"] = pd.to_datetime(df["time"], utc=True)
    return df


def fetch_scores(conn, coins: tuple[str, ...], start: datetime | None,
                 end: datetime | None) -> pd.DataFrame:
    where, params = _time_filter(start, end)
    params["coins"] = list(coins)
    q = sql.SQL("""
SELECT time, url, coin,
       finbert_compound::double precision, vader_compound::double precision, finbert_label
FROM sentiment_scores
WHERE coin = ANY(%(coins)s) AND {where}
""").format(where=where)
    with conn.cursor() as cur:
        cur.execute(q, params)
        rows = cur.fetchall()
    conn.commit()
    df = pd.DataFrame(rows, columns=list(SCORE_COLUMNS))
    df["time"] = pd.to_datetime(df["time"], utc=True)
    return df


# ------------------------------------------------------------------ write
def _to_records(df: pd.DataFrame) -> list[tuple]:
    """DataFrame -> plain Python tuples (NaN -> NULL, numpy scalars -> Python)."""
    obj = df[list(FEATURE_COLUMNS)].astype(object)
    obj = obj.where(pd.notna(obj), None)
    records = []
    for row in obj.itertuples(index=False, name=None):
        records.append(tuple(v.item() if hasattr(v, "item") and not isinstance(v, pd.Timestamp) else v
                             for v in row))
    return records


def upsert_features(conn, table: str, df: pd.DataFrame, page_size: int = 1000) -> int:
    """Idempotent write: re-running over the same bars updates them in place."""
    if df.empty:
        return 0
    cols = sql.SQL(", ").join(sql.Identifier(c) for c in FEATURE_COLUMNS)
    updates = sql.SQL(", ").join(
        sql.SQL("{c} = EXCLUDED.{c}").format(c=sql.Identifier(c))
        for c in FEATURE_COLUMNS if c not in ("bucket", "symbol", "freq")
    )
    q = sql.SQL("""
INSERT INTO {table} ({cols}) VALUES %s
ON CONFLICT (symbol, freq, bucket) DO UPDATE SET {updates}, computed_at = now()
""").format(table=sql.Identifier(table), cols=cols, updates=updates)
    records = _to_records(df)
    with conn.cursor() as cur:
        execute_values(cur, q.as_string(conn), records, page_size=page_size)
    conn.commit()
    return len(records)


# ------------------------------------------------------------------ run log
def start_run(conn, mode: str, write_from: datetime | None) -> int:
    with conn.cursor() as cur:
        cur.execute(
            "INSERT INTO job_runs (job_name, status, mode, write_from) VALUES (%s, 'running', %s, %s) RETURNING id",
            (JOB_NAME, mode, write_from),
        )
        run_id = cur.fetchone()[0]
    conn.commit()
    return run_id


def finish_run(conn, run_id: int, status: str, rows_written: int | None = None,
               details: str | None = None) -> None:
    with conn.cursor() as cur:
        cur.execute(
            "UPDATE job_runs SET finished_at = now(), status = %s, rows_written = %s, details = %s WHERE id = %s",
            (status, rows_written, (details or "")[:2000] or None, run_id),
        )
    conn.commit()