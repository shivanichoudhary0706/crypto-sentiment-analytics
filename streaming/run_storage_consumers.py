"""
Runs the market and sentiment consumers concurrently.
Kafka consumers are synchronous/blocking (unlike our asyncio producers),
so they run as threads here rather than asyncio tasks.

Run with: python -m streaming.run_storage_consumers
"""

import threading

from config.logging_config import get_logger
from streaming import market_consumer, sentiment_consumer

logger = get_logger(__name__)


def main() -> None:
    market_thread = threading.Thread(target=market_consumer.run, name="market-consumer", daemon=True)
    sentiment_thread = threading.Thread(target=sentiment_consumer.run, name="sentiment-consumer", daemon=True)

    market_thread.start()
    sentiment_thread.start()

    logger.info("Both storage consumers running. Press Ctrl+C to stop.")

    try:
        market_thread.join()
        sentiment_thread.join()
    except KeyboardInterrupt:
        logger.info("Storage consumers stopped by user.")


if __name__ == "__main__":
    main()