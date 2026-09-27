"""
Consumes raw-sentiment-data from Kafka and writes articles to TimescaleDB.

Delivery guarantee: AT-LEAST-ONCE.
  - Kafka offsets are committed manually, only AFTER a row is safely written
    (or a message is deliberately skipped as permanently malformed).
  - Inserts are idempotent (ON CONFLICT (time, url) DO NOTHING), so a message
    re-delivered after a crash or restart is harmless.

Failure policy:
  - Malformed message (bad JSON, missing fields, bad timestamp) -> log, skip,
    commit offset. Retrying can never fix it (a "poison message").
  - Database connection error -> reconnect with exponential backoff. If the DB
    is still down after DB_MAX_RETRIES, stop WITHOUT committing, so the message
    is re-read on restart. Fail loudly rather than lose data silently.

Run standalone with: python -m streaming.sentiment_consumer
"""

import json
import time
from datetime import datetime, timezone

import psycopg2
from dateutil import parser as date_parser
from kafka import KafkaConsumer

from config.logging_config import get_logger
from config.settings import settings
from streaming.db import get_db_connection

logger = get_logger(__name__)

INSERT_SQL = """
    INSERT INTO raw_sentiment
        (time, source_type, source_name, title, summary, url, coin_tags)
    VALUES
        (%(time)s, %(source_type)s, %(source_name)s, %(title)s, %(summary)s,
         %(url)s, %(coin_tags)s)
    ON CONFLICT (time, url) DO NOTHING
"""

REQUIRED_FIELDS = ("source_type", "source_name", "title", "url", "published_at")
DB_MAX_RETRIES = 5
DB_BACKOFF_BASE_SECONDS = 2   # waits 2, 4, 8, 16s -> ~30s total, well under Kafka's 300s poll limit
LOG_EVERY = 20

DB_CONNECTION_ERRORS = (psycopg2.OperationalError, psycopg2.InterfaceError)


class MalformedMessageError(ValueError):
    """A message that can never be written, no matter how often we retry."""


# ---------------------------------------------------------------------------
# Parsing / validation
# ---------------------------------------------------------------------------

def _decode(raw) -> dict:
    """Decode raw Kafka bytes into a dict. Done here (not via
    value_deserializer) so bad bytes raise inside OUR try/except instead of
    crashing kafka-python's internal iterator."""
    if raw is None:
        raise MalformedMessageError("empty message value")
    try:
        data = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as e:
        raise MalformedMessageError(f"not valid UTF-8 JSON: {e}") from e
    if not isinstance(data, dict):
        raise MalformedMessageError(f"expected a JSON object, got {type(data).__name__}")
    return data


def _parse_time(value) -> datetime:
    """Parse the article's publish time. Never falls back to 'now': that would
    silently shift sentiment later in time and bias the lag estimate."""
    try:
        dt = date_parser.parse(value)
    except (ValueError, TypeError, OverflowError) as e:
        raise MalformedMessageError(f"unparseable published_at {value!r}: {e}") from e
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def _to_row(message: dict) -> dict:
    missing = [f for f in REQUIRED_FIELDS if not message.get(f)]
    if missing:
        raise MalformedMessageError(f"missing required field(s): {missing}")

    coin_tags = message.get("coin_tags") or []
    if not isinstance(coin_tags, list):
        raise MalformedMessageError(f"coin_tags must be a list, got {type(coin_tags).__name__}")

    return {
        "time": _parse_time(message["published_at"]),
        "source_type": message["source_type"],
        "source_name": message["source_name"],
        "title": message["title"],
        "summary": message.get("summary") or "",
        "url": message["url"],
        "coin_tags": coin_tags,
    }


# ---------------------------------------------------------------------------
# Database writer with reconnect
# ---------------------------------------------------------------------------

class SentimentWriter:
    """Owns the DB connection; reconnects with exponential backoff on
    connection-level errors."""

    def __init__(self) -> None:
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
        """Insert one row. Returns True if inserted, False if it was a duplicate.
        Raises a DB connection error if the DB stays down after all retries."""
        for attempt in range(1, DB_MAX_RETRIES + 1):
            try:
                if self.conn is None or self.conn.closed:
                    self._connect()
                with self.conn.cursor() as cur:
                    cur.execute(INSERT_SQL, row)
                    return cur.rowcount == 1
            except DB_CONNECTION_ERRORS as e:
                self.close()
                if attempt == DB_MAX_RETRIES:
                    raise
                wait = DB_BACKOFF_BASE_SECONDS ** attempt
                logger.warning(f"DB connection error (attempt {attempt}/{DB_MAX_RETRIES}): {e}. "
                               f"Reconnecting in {wait}s...")
                time.sleep(wait)
        return False  # unreachable; keeps type checkers happy


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def run() -> None:
    consumer = KafkaConsumer(
        settings.kafka_topics["sentiment_raw"],
        bootstrap_servers=settings.kafka_bootstrap_servers,
        group_id=settings.kafka_consumer_group,
        auto_offset_reset="earliest",
        enable_auto_commit=False,   # we commit manually, only after a safe write
    )
    writer = SentimentWriter()
    written = duplicates = skipped = 0

    logger.info("Sentiment consumer started (at-least-once, manual commit). "
                "Writing to TimescaleDB raw_sentiment table.")

    try:
        for msg in consumer:
            where = f"{msg.topic}[{msg.partition}]@{msg.offset}"

            # 1) Validate. Poison messages are skipped and committed.
            try:
                row = _to_row(_decode(msg.value))
            except MalformedMessageError as e:
                skipped += 1
                logger.warning(f"Skipping malformed message {where}: {e}")
                consumer.commit()
                continue

            # 2) Write. Connection errors propagate (no commit); data errors are skipped.
            try:
                inserted = writer.write(row)
            except DB_CONNECTION_ERRORS:
                logger.critical(f"Database unavailable after {DB_MAX_RETRIES} attempts at {where}. "
                                f"Stopping WITHOUT committing; message will be re-read on restart.")
                raise
            except psycopg2.Error as e:
                skipped += 1
                logger.error(f"Data error writing {row['url']} ({where}), skipping: {e}")
                consumer.commit()
                continue

            # 3) Only now is it safe to tell Kafka we're done with this message.
            if inserted:
                written += 1
            else:
                duplicates += 1
            consumer.commit()

            if (written + duplicates + skipped) % LOG_EVERY == 0:
                logger.info(f"Progress: written={written}, duplicates={duplicates}, skipped={skipped}")

    except KeyboardInterrupt:
        logger.info("Sentiment consumer stopped by user.")
    finally:
        consumer.close()
        writer.close()
        logger.info(f"Sentiment consumer closed. written={written}, "
                    f"duplicates={duplicates}, skipped={skipped}")


if __name__ == "__main__":
    run()