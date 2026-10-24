"""Read-only PostgreSQL/TimescaleDB access for the API.

One ThreadedConnectionPool shared by FastAPI's worker threads (sync endpoints run
in a threadpool). Sessions are read-only + autocommit, so a dashboard bug can never
write to or lock the pipeline's tables.
"""
import logging
import os
from contextlib import contextmanager

import psycopg2
import psycopg2.extras
from dotenv import load_dotenv
from psycopg2.pool import ThreadedConnectionPool

load_dotenv()
logger = logging.getLogger(__name__)
_pool: ThreadedConnectionPool | None = None


def init_pool(minconn: int = 1, maxconn: int = 5) -> None:
    global _pool
    if _pool is not None:
        return
    _pool = ThreadedConnectionPool(
        minconn, maxconn,
        host=os.getenv("TIMESCALEDB_HOST", "localhost"),
        port=int(os.getenv("TIMESCALEDB_PORT", "5432")),
        dbname=os.getenv("TIMESCALEDB_DB", "crypto_analytics"),
        user=os.getenv("TIMESCALEDB_USER", "crypto_user"),
        password=os.getenv("TIMESCALEDB_PASSWORD", "crypto_pass"),
        connect_timeout=5,
    )
    logger.info("DB pool ready (%d-%d connections)", minconn, maxconn)


def close_pool() -> None:
    global _pool
    if _pool is not None:
        _pool.closeall()
        _pool = None


@contextmanager
def _connection():
    if _pool is None:
        init_pool()
    conn = _pool.getconn()
    broken = False
    try:
        conn.set_session(readonly=True, autocommit=True)
        yield conn
    except (psycopg2.OperationalError, psycopg2.InterfaceError):
        broken = True  # drop dead connections instead of returning them to the pool
        raise
    finally:
        _pool.putconn(conn, close=broken)


def fetch_all(sql: str, params=None) -> list[dict]:
    with _connection() as conn, conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(sql, params)
        return [dict(r) for r in cur.fetchall()]