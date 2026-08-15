"""
Sanity check for the storage layer — confirms both tables exist,
are real TimescaleDB hypertables, and have data.

Run with: python tests/test_storage.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from streaming.db import get_db_connection

conn = get_db_connection()

with conn.cursor() as cur:
    cur.execute("SELECT hypertable_name FROM timescaledb_information.hypertables;")
    hypertables = {row[0] for row in cur.fetchall()}

    assert "market_data" in hypertables, "FAIL: market_data is not a hypertable"
    assert "raw_sentiment" in hypertables, "FAIL: raw_sentiment is not a hypertable"
    print("PASS: Both tables are confirmed TimescaleDB hypertables.")

    cur.execute("SELECT count(*) FROM market_data;")
    market_count = cur.fetchone()[0]
    print(f"market_data row count: {market_count}")

    cur.execute("SELECT count(*) FROM raw_sentiment;")
    sentiment_count = cur.fetchone()[0]
    print(f"raw_sentiment row count: {sentiment_count}")

    assert market_count > 0, "FAIL: market_data has no rows — run market ingestion + storage consumer for 2+ minutes."
    assert sentiment_count > 0, "FAIL: raw_sentiment has no rows."
    print("\nPASS: Storage layer is working end-to-end.")

conn.close()