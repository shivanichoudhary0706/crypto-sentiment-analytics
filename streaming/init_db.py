"""
Applies the database schema. Safe to re-run (everything uses IF NOT EXISTS).
Run with: python -m streaming.init_db
"""

from pathlib import Path

import psycopg2

from config.logging_config import get_logger
from config.settings import settings

logger = get_logger(__name__)

SCHEMA_PATH = Path(__file__).resolve().parent / "db_schema.sql"


def run() -> None:
    logger.info(f"Connecting to TimescaleDB at {settings.timescaledb_host}:{settings.timescaledb_port}")
    conn = psycopg2.connect(
        host=settings.timescaledb_host,
        port=settings.timescaledb_port,
        dbname=settings.timescaledb_db,
        user=settings.timescaledb_user,
        password=settings.timescaledb_password,
    )
    conn.autocommit = True  # required for CREATE EXTENSION / create_hypertable

    schema_sql = SCHEMA_PATH.read_text()

    with conn.cursor() as cur:
        cur.execute(schema_sql)

    conn.close()
    logger.info("Schema applied successfully: market_data and raw_sentiment hypertables ready.")


if __name__ == "__main__":
    run()