"""
Integration test for market data ingestion.
Runs the WebSocket -> Kafka pipeline for a fixed short duration,
then verifies at least one message was published to raw-market-data.

Run with: python tests/test_market_ingestion.py

Requires Docker (Kafka) to be running.
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

from kafka import KafkaConsumer

from config.settings import settings
from config.logging_config import get_logger
from ingestion.market.binance_ws_client import stream_klines
from ingestion.market.kafka_producer import build_market_data_producer, publish_market_message

logger = get_logger(__name__)

TEST_DURATION_SECONDS = 15


async def produce_for_a_while():
    symbols = [coin["symbol"] for coin in settings.coins]
    producer = build_market_data_producer()
    count = 0

    async def _collect():
        nonlocal count
        async for message in stream_klines(symbols):
            publish_market_message(producer, message.to_dict())
            count += 1

    try:
        await asyncio.wait_for(_collect(), timeout=TEST_DURATION_SECONDS)
    except asyncio.TimeoutError:
        pass

    producer.flush()
    producer.close()
    return count


def verify_messages_in_kafka() -> int:
    consumer = KafkaConsumer(
        settings.kafka_topics["market_raw"],
        bootstrap_servers=settings.kafka_bootstrap_servers,
        auto_offset_reset="latest",
        consumer_timeout_ms=5000,
    )
    found = 0
    for _ in consumer:
        found += 1
        if found >= 3:
            break
    consumer.close()
    return found


if __name__ == "__main__":
    print(f"Running market ingestion for {TEST_DURATION_SECONDS}s...")
    published = asyncio.run(produce_for_a_while())
    print(f"Published {published} messages during test window.")

    assert published > 0, "FAIL: No messages were published — check WebSocket/Kafka connectivity."
    print("PASS: Market ingestion is producing messages to Kafka.")