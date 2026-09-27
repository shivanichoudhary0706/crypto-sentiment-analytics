"""
Consumes raw-market-data from Kafka and writes closed 1-min candles to TimescaleDB.

Delivery guarantee: AT-LEAST-ONCE.
  - Offsets are committed manually, only after a candle is safely written or a
    message is deliberately skipped (partial candle / malformed).
  - Inserts are idempotent (ON CONFLICT (symbol, time) DO NOTHING).

Data-quality rule: candles failing price sanity checks (non-numeric, <= 0,
NaN/inf, high < low, etc.) are rejected and logged. A single corrupt price
creates an extreme fake return that can distort lag detection.

Run standalone with: python -m streaming.market_consumer
(or via run_storage_consumers.py to run alongside the sentiment consumer)
"""

import json
import math
from datetime import datetime, timezone

import psycopg2
from kafka import KafkaConsumer

from config.logging_config import get_logger
from config.settings import settings
from streaming.resilient_writer import DB_CONNECTION_ERRORS, ResilientWriter

logger = get_logger(__name__)

# Only store finalized 1-min candles, not every ~1-2s partial update.
# A partial candle updates dozens of times before closing; storing all of
# them would bloat the table ~30-60x for no analytical benefit, since
# lag detection and backtesting both operate on completed candles.
STORE_ONLY_CLOSED_CANDLES = True
LOG_EVERY = 20

INSERT_SQL = """
    INSERT INTO market_data
        (time, symbol, exchange, open, high, low, close, volume, kline_close_time)
    VALUES
        (%(time)s, %(symbol)s, %(exchange)s, %(open)s, %(high)s, %(low)s,
         %(close)s, %(volume)s, %(kline_close_time)s)
    ON CONFLICT (symbol, time) DO NOTHING
"""

REQUIRED_FIELDS = ("kline_start_time", "kline_close_time", "symbol", "exchange",
                   "open", "high", "low", "close", "volume")


class MalformedMessageError(ValueError):
    """A message that can never be written, no matter how often we retry."""


# ---------------------------------------------------------------------------
# Parsing / validation
# ---------------------------------------------------------------------------

def _decode(raw) -> dict:
    if raw is None:
        raise MalformedMessageError("empty message value")
    try:
        data = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as e:
        raise MalformedMessageError(f"not valid UTF-8 JSON: {e}") from e
    if not isinstance(data, dict):
        raise MalformedMessageError(f"expected a JSON object, got {type(data).__name__}")
    return data


def _to_float(value, field: str) -> float:
    """Accepts numbers or numeric strings (Binance sends prices as strings)."""
    if isinstance(value, bool):
        raise MalformedMessageError(f"{field} must be numeric, got bool")
    try:
        f = float(value)
    except (TypeError, ValueError) as e:
        raise MalformedMessageError(f"{field} not numeric: {value!r}") from e
    if not math.isfinite(f):
        raise MalformedMessageError(f"{field} is not finite: {value!r}")
    return f


def _ms_to_dt(value, field: str) -> datetime:
    ms = _to_float(value, field)
    try:
        return datetime.fromtimestamp(ms / 1000, tz=timezone.utc)
    except (OverflowError, OSError, ValueError) as e:
        raise MalformedMessageError(f"{field} out of range: {value!r}") from e


def _to_row(message: dict) -> dict:
    # `in (None, "")` rather than `not value`: volume can legitimately be 0.
    missing = [f for f in REQUIRED_FIELDS if message.get(f) in (None, "")]
    if missing:
        raise MalformedMessageError(f"missing required field(s): {missing}")

    o = _to_float(message["open"], "open")
    h = _to_float(message["high"], "high")
    l = _to_float(message["low"], "low")
    c = _to_float(message["close"], "close")
    v = _to_float(message["volume"], "volume")

    if min(o, h, l, c) <= 0:
        raise MalformedMessageError(f"non-positive price: o={o} h={h} l={l} c={c}")
    if v < 0:
        raise MalformedMessageError(f"negative volume: {v}")
    if h < max(o, c, l) or l > min(o, c):
        raise MalformedMessageError(f"inconsistent OHLC: o={o} h={h} l={l} c={c}")

    start = _ms_to_dt(message["kline_start_time"], "kline_start_time")
    close_time = _ms_to_dt(message["kline_close_time"], "kline_close_time")
    if close_time <= start:
        raise MalformedMessageError(f"kline_close_time {close_time} not after start {start}")

    return {
        "time": start,
        "symbol": message["symbol"],
        "exchange": message["exchange"],
        "open": o, "high": h, "low": l, "close": c, "volume": v,
        "kline_close_time": close_time,
    }


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def run() -> None:
    consumer = KafkaConsumer(
        settings.kafka_topics["market_raw"],
        bootstrap_servers=settings.kafka_bootstrap_servers,
        group_id=settings.kafka_consumer_group,
        auto_offset_reset="earliest",
        enable_auto_commit=False,   # commit manually, only after a safe write/skip
    )
    writer = ResilientWriter(INSERT_SQL, name="market")
    written = duplicates = skipped_partial = malformed = 0

    logger.info("Market consumer started (at-least-once, manual commit). "
                "Writing closed candles to TimescaleDB market_data table.")

    try:
        for msg in consumer:
            where = f"{msg.topic}[{msg.partition}]@{msg.offset}"

            # 1) Decode, filter partials, validate.
            try:
                data = _decode(msg.value)
                if STORE_ONLY_CLOSED_CANDLES and not data.get("is_closed", False):
                    skipped_partial += 1
                    consumer.commit()
                    continue
                row = _to_row(data)
            except MalformedMessageError as e:
                malformed += 1
                logger.warning(f"Skipping malformed market message {where}: {e}")
                consumer.commit()
                continue

            # 2) Write. Connection errors propagate (no commit); data errors are skipped.
            try:
                inserted = writer.write(row)
            except DB_CONNECTION_ERRORS:
                logger.critical(f"Database unavailable after retries at {where}. "
                                f"Stopping WITHOUT committing; message will be re-read on restart.")
                raise
            except psycopg2.Error as e:
                malformed += 1
                logger.error(f"Data error writing {row['symbol']} @ {row['time']} ({where}), skipping: {e}")
                consumer.commit()
                continue

            # 3) Safe to acknowledge.
            consumer.commit()
            if inserted:
                written += 1
                if written % LOG_EVERY == 0:
                    logger.info(f"Progress: written={written}, duplicates={duplicates}, "
                                f"partial_skipped={skipped_partial}, malformed={malformed}")
            else:
                duplicates += 1

    except KeyboardInterrupt:
        logger.info("Market consumer stopped by user.")
    finally:
        consumer.close()
        writer.close()
        logger.info(f"Market consumer closed. written={written}, duplicates={duplicates}, "
                    f"partial_skipped={skipped_partial}, malformed={malformed}")


if __name__ == "__main__":
    run()