"""
Consumes raw-market-data from Kafka and writes closed candles to TimescaleDB.
Run standalone with: python -m streaming.market_consumer
(or via run_storage_consumers.py to run alongside the sentiment consumer)
"""

import json
from datetime import datetime, timezone

from kafka import KafkaConsumer

from config.logging_config import get_logger
from config.settings import settings
from streaming.db import get_db_connection

logger = get_logger(__name__)

# Only store finalized 1-min candles, not every ~1-2s partial update.
# A partial candle updates dozens of times before closing; storing all of
# them would bloat the table ~30-60x for no analytical benefit, since
# lag detection and backtesting both operate on completed candles.
STORE_ONLY_CLOSED_CANDLES = True

INSERT_SQL = """
    INSERT INTO market_data
        (time, symbol, exchange, open, high, low, close, volume, kline_close_time)
    VALUES
        (%(time)s, %(symbol)s, %(exchange)s, %(open)s, %(high)s, %(low)s,
         %(close)s, %(volume)s, %(kline_close_time)s)
    ON CONFLICT (symbol, time) DO NOTHING
"""


def _to_row(message: dict) -> dict:
    return {
        "time": datetime.fromtimestamp(message["kline_start_time"] / 1000, tz=timezone.utc),
        "symbol": message["symbol"],
        "exchange": message["exchange"],
        "open": message["open"],
        "high": message["high"],
        "low": message["low"],
        "close": message["close"],
        "volume": message["volume"],
        "kline_close_time": datetime.fromtimestamp(message["kline_close_time"] / 1000, tz=timezone.utc),
    }


def run() -> None:
    consumer = KafkaConsumer(
        settings.kafka_topics["market_raw"],
        bootstrap_servers=settings.kafka_bootstrap_servers,
        group_id=settings.kafka_consumer_group,
        auto_offset_reset="earliest",
        enable_auto_commit=True,
        value_deserializer=lambda v: json.loads(v.decode("utf-8")),
    )
    conn = get_db_connection()
    conn.autocommit = True

    logger.info("Market consumer started. Writing to TimescaleDB market_data table.")
    written = 0
    skipped_partial = 0

    try:
        for msg in consumer:
            data = msg.value

            if STORE_ONLY_CLOSED_CANDLES and not data.get("is_closed", False):
                skipped_partial += 1
                continue

            row = _to_row(data)
            try:
                with conn.cursor() as cur:
                    cur.execute(INSERT_SQL, row)
                written += 1
                if written % 20 == 0:
                    logger.info(f"Written {written} candles so far "
                                f"(skipped {skipped_partial} partial updates)")
            except Exception as e:
                logger.error(f"Failed to write market row for {row['symbol']} @ {row['time']}: {e}")

    except KeyboardInterrupt:
        logger.info("Market consumer stopped by user.")
    finally:
        consumer.close()
        conn.close()
        logger.info(f"Market consumer closed. Total written: {written}")


if __name__ == "__main__":
    run()