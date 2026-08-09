"""
Main entry point for live market data ingestion.
Run with: python -m ingestion.market.run_market_ingestion
"""

import asyncio
import sys

import websockets

from config.logging_config import get_logger
from config.settings import settings
from ingestion.market.binance_ws_client import stream_klines
from ingestion.market.kafka_producer import build_market_data_producer, publish_market_message

# Windows' default ProactorEventLoop has a known SSL-teardown bug (harmless but noisy)
# when an SSL connection closes after the loop shuts down. SelectorEventLoop avoids it.
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

logger = get_logger(__name__)

RECONNECT_DELAY_SECONDS = 5
MAX_RECONNECT_DELAY_SECONDS = 60


async def run() -> None:
    symbols = [coin["symbol"] for coin in settings.coins]
    producer = build_market_data_producer()

    reconnect_delay = RECONNECT_DELAY_SECONDS
    message_count = 0

    while True:
        try:
            async for message in stream_klines(symbols):
                publish_market_message(producer, message.to_dict())
                message_count += 1

                if message_count % 50 == 0:
                    logger.info(f"Published {message_count} messages so far "
                                f"(latest: {message.symbol} close={message.close})")

                reconnect_delay = RECONNECT_DELAY_SECONDS

        except (websockets.exceptions.ConnectionClosed, OSError) as e:
            logger.warning(
                f"WebSocket disconnected ({e}). "
                f"Reconnecting in {reconnect_delay}s..."
            )
            await asyncio.sleep(reconnect_delay)
            reconnect_delay = min(reconnect_delay * 2, MAX_RECONNECT_DELAY_SECONDS)

        except KeyboardInterrupt:
            logger.info("Shutting down market ingestion (Ctrl+C received).")
            break

        except Exception as e:
            logger.error(f"Unexpected error: {e}. Reconnecting in {reconnect_delay}s...")
            await asyncio.sleep(reconnect_delay)
            reconnect_delay = min(reconnect_delay * 2, MAX_RECONNECT_DELAY_SECONDS)

    producer.flush()
    producer.close()
    logger.info(f"Producer closed. Total messages published: {message_count}")


if __name__ == "__main__":
    try:
        asyncio.run(run())
    except KeyboardInterrupt:
        logger.info("Ingestion stopped by user.")