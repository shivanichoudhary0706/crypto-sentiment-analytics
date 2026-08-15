"""
Consumes raw-sentiment-data from Kafka and writes articles to TimescaleDB.
Run standalone with: python -m streaming.sentiment_consumer
"""

import json
from datetime import datetime, timezone
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


def _parse_time(value: str) -> datetime:
    try:
        dt = date_parser.parse(value)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except (ValueError, TypeError):
        return datetime.now(timezone.utc)


def _to_row(message: dict) -> dict:
    return {
        "time": _parse_time(message.get("published_at")),
        "source_type": message["source_type"],
        "source_name": message["source_name"],
        "title": message["title"],
        "summary": message.get("summary", ""),
        "url": message["url"],
        "coin_tags": message.get("coin_tags", []),
    }


def run() -> None:
    consumer = KafkaConsumer(
        settings.kafka_topics["sentiment_raw"],
        bootstrap_servers=settings.kafka_bootstrap_servers,
        group_id=settings.kafka_consumer_group,
        auto_offset_reset="earliest",
        enable_auto_commit=True,
        value_deserializer=lambda v: json.loads(v.decode("utf-8")),
    )
    conn = get_db_connection()
    conn.autocommit = True

    logger.info("Sentiment consumer started. Writing to TimescaleDB raw_sentiment table.")
    written = 0
    duplicates = 0

    try:
        for msg in consumer:
            data = msg.value
            row = _to_row(data)
            try:
                with conn.cursor() as cur:
                    cur.execute(INSERT_SQL, row)
                    if cur.rowcount == 0:
                        duplicates += 1
                    else:
                        written += 1
                if (written + duplicates) % 20 == 0:
                    logger.info(f"Written {written} articles so far "
                                f"(skipped {duplicates} duplicates)")
            except Exception as e:
                logger.error(f"Failed to write sentiment row for {row['url']}: {e}")

    except KeyboardInterrupt:
        logger.info("Sentiment consumer stopped by user.")
    finally:
        consumer.close()
        conn.close()
        logger.info(f"Sentiment consumer closed. Total written: {written}, duplicates skipped: {duplicates}")


if __name__ == "__main__":
    run()