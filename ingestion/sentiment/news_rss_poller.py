"""
Polls crypto news RSS feeds on a schedule, tags each article by coin,
and publishes new (not-yet-seen) articles to Kafka.
"""

import asyncio
from datetime import datetime, timezone

import feedparser
import requests

from config.logging_config import get_logger
from config.settings import settings
from ingestion.sentiment.coin_tagger import tag_coins
from ingestion.sentiment.kafka_producer import build_sentiment_producer, publish_sentiment_message
from ingestion.sentiment.schemas import RawSentimentMessage

logger = get_logger(__name__)

# Some outlets (e.g. CoinDesk) reject requests with no/generic User-Agent.
# A descriptive UA is also just polite practice when polling someone else's server.
REQUEST_HEADERS = {
    "User-Agent": "crypto-sentiment-analytics-thesis/1.0 (M.Tech research project; academic use)"
}


def _parse_published_time(entry) -> str:
    if getattr(entry, "published_parsed", None):
        dt = datetime(*entry.published_parsed[:6], tzinfo=timezone.utc)
        return dt.isoformat()
    return datetime.now(timezone.utc).isoformat()


def _fetch_feed_entries(feed_name: str, feed_url: str) -> list:
    """
    Fetches with a proper User-Agent via requests first (so blocked/HTML
    responses raise clearly), then hands the raw bytes to feedparser —
    rather than letting feedparser make the request itself with no headers.
    """
    response = requests.get(feed_url, headers=REQUEST_HEADERS, timeout=15)
    response.raise_for_status()

    parsed = feedparser.parse(response.content)
    if parsed.bozo and not parsed.entries:
        # bozo=True with zero entries usually means the feed body wasn't
        # valid XML (e.g. an HTML block page). Log and skip this cycle.
        logger.warning(f"Feed '{feed_name}' returned unparseable content: {parsed.bozo_exception}")
        return []
    return parsed.entries


async def poll_news_rss_forever() -> None:
    producer = build_sentiment_producer()
    seen_urls = set()  # in-memory dedup for this run; TimescaleDB dedup comes in Week 4
    message_count = 0

    while True:
        for feed in settings.news_rss_feeds:
            feed_name = feed["name"]
            feed_url = feed["url"]

            try:
                entries = _fetch_feed_entries(feed_name, feed_url)
            except requests.exceptions.RequestException as e:
                logger.error(f"Failed to fetch feed '{feed_name}': {e}")
                continue

            new_in_this_feed = 0
            for entry in entries:
                url = getattr(entry, "link", None)
                if not url or url in seen_urls:
                    continue
                seen_urls.add(url)

                title = getattr(entry, "title", "")
                summary = getattr(entry, "summary", "")
                combined_text = f"{title} {summary}"

                message = RawSentimentMessage(
                    source_type="news_rss",
                    source_name=feed_name,
                    title=title,
                    summary=summary,
                    url=url,
                    published_at=_parse_published_time(entry),
                    ingested_at=datetime.now(timezone.utc).isoformat(),
                    coin_tags=tag_coins(combined_text),
                )
                publish_sentiment_message(producer, message.to_dict())
                message_count += 1
                new_in_this_feed += 1

            if new_in_this_feed:
                logger.info(f"[{feed_name}] published {new_in_this_feed} new articles "
                            f"(total so far: {message_count})")

        await asyncio.sleep(settings.news_poll_interval)