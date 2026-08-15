"""
Main entry point for sentiment ingestion — runs News RSS and GDELT pollers concurrently.
Run with: python -m ingestion.sentiment.run_sentiment_ingestion
"""

import asyncio
import sys

from config.logging_config import get_logger
from ingestion.sentiment.news_rss_poller import poll_news_rss_forever
from ingestion.sentiment.gdelt_poller import poll_gdelt_forever

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

logger = get_logger(__name__)


async def run() -> None:
    logger.info("Starting sentiment ingestion: News RSS + GDELT")
    await asyncio.gather(
        poll_news_rss_forever(),
        poll_gdelt_forever(),
    )


if __name__ == "__main__":
    try:
        asyncio.run(run())
    except KeyboardInterrupt:
        logger.info("Sentiment ingestion stopped by user.")