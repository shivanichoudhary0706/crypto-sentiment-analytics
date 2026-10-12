"""
Main entry point for sentiment ingestion.

Runs the News RSS poller, the GDELT poller, or both, as chosen by --source.
Each poller owns its own Kafka producer and dedup set, so they can safely run
in separate processes (e.g. two terminals) without interfering.

Usage:
    python -m ingestion.sentiment.run_sentiment_ingestion                 # both (default)
    python -m ingestion.sentiment.run_sentiment_ingestion --source rss    # RSS only
    python -m ingestion.sentiment.run_sentiment_ingestion --source gdelt  # GDELT only

Shortcuts:
    python -m ingestion.sentiment.run_rss
    python -m ingestion.sentiment.run_gdelt
"""

import argparse
import asyncio
import sys
from typing import Awaitable, Callable, Dict, List, Optional

from config.logging_config import get_logger
from ingestion.sentiment.gdelt_poller import poll_gdelt_forever
from ingestion.sentiment.news_rss_poller import poll_news_rss_forever

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

logger = get_logger(__name__)

# Registry of available pollers. Adding a new source = one new line here.
POLLERS: Dict[str, Callable[[], Awaitable[None]]] = {
    "rss": poll_news_rss_forever,
    "gdelt": poll_gdelt_forever,
}

SOURCE_CHOICES = ["all", *POLLERS.keys()]
SOURCE_LABELS = {"rss": "News RSS", "gdelt": "GDELT"}


def resolve_sources(source: str) -> List[str]:
    """Translate a --source value into the list of poller keys to run."""
    if source == "all":
        return list(POLLERS.keys())
    if source not in POLLERS:
        raise ValueError(f"Unknown source '{source}'. Choose from: {', '.join(SOURCE_CHOICES)}")
    return [source]


async def run(source: str = "all") -> None:
    selected = resolve_sources(source)
    labels = " + ".join(SOURCE_LABELS.get(s, s) for s in selected)
    logger.info(f"Starting sentiment ingestion: {labels}")
    await asyncio.gather(*(POLLERS[s]() for s in selected))


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run sentiment ingestion pollers.")
    parser.add_argument(
        "--source",
        choices=SOURCE_CHOICES,
        default="all",
        help="Which poller to run (default: all).",
    )
    return parser.parse_args(argv)


def main(source: Optional[str] = None) -> None:
    """CLI entry point. If `source` is given, it overrides command-line parsing."""
    if source is None:
        source = parse_args().source
    try:
        asyncio.run(run(source))
    except KeyboardInterrupt:
        logger.info(f"Sentiment ingestion ({source}) stopped by user.")


if __name__ == "__main__":
    main()