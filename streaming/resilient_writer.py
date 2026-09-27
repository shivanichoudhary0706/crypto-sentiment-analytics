"""
Shared TimescaleDB writer for Kafka consumers.

Reconnects with exponential backoff on connection-level errors. If the DB is
still unreachable after max_retries, the error is re-raised so the caller can
stop WITHOUT committing the Kafka offset (at-least-once delivery).
"""

import time

import psycopg2

from config.logging_config import get_logger
from streaming.db import get_db_connection

logger = get_logger(__name__)

DB_CONNECTION_ERRORS = (psycopg2.OperationalError, psycopg2.InterfaceError)


class ResilientWriter:
    """Owns one DB connection and executes a single idempotent INSERT per row."""

    def __init__(self, insert_sql: str, name: str = "writer",
                 max_retries: int = 5, backoff_base_seconds: int = 2) -> None:
        self.insert_sql = insert_sql
        self.name = name
        self.max_retries = max_retries
        self.backoff_base_seconds = backoff_base_seconds  # waits 2, 4, 8, 16s -> ~30s total
        self.conn = None
        self._connect()

    def _connect(self) -> None:
        self.close()
        self.conn = get_db_connection()
        self.conn.autocommit = True

    def close(self) -> None:
        if self.conn is not None and not self.conn.closed:
            try:
                self.conn.close()
            except Exception:  # closing a broken connection can itself fail
                pass
        self.conn = None

    def write(self, row: dict) -> bool:
        """Insert one row. Returns True if inserted, False if duplicate.
        Raises a DB connection error if the DB stays down after all retries."""
        for attempt in range(1, self.max_retries + 1):
            try:
                if self.conn is None or self.conn.closed:
                    self._connect()
                with self.conn.cursor() as cur:
                    cur.execute(self.insert_sql, row)
                    return cur.rowcount == 1
            except DB_CONNECTION_ERRORS as e:
                self.close()
                if attempt == self.max_retries:
                    raise
                wait = self.backoff_base_seconds ** attempt
                logger.warning(f"[{self.name}] DB connection error (attempt {attempt}/"
                               f"{self.max_retries}): {e}. Reconnecting in {wait}s...")
                time.sleep(wait)
        return False  # unreachable