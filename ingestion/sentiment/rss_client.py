"""
RSS news ingestion client.

Fetches configured cryptocurrency news RSS feeds and converts entries
into normalized SentimentMessage objects.

This module does not publish to Kafka. Kafka integration is handled
separately by the Week 3 ingestion runner.
"""

import hashlib
import re
from datetime import datetime, timezone
from typing import Iterable, List, Optional

import feedparser

from config.logging_config import get_logger
from config.settings import settings
from ingestion.sentiment.schemas import SentimentMessage


logger = get_logger(__name__)


COIN_KEYWORDS = {
    "BTC/USDT": (
        "bitcoin",
        "btc",
        "xbt",
    ),
    "ETH/USDT": (
        "ethereum",
        "ether",
        "eth",
    ),
}


def _clean_text(value: Optional[str]) -> str:
    """Normalize whitespace and remove simple HTML tags."""
    if not value:
        return ""

    value = re.sub(r"<[^>]+>", " ", value)
    value = re.sub(r"\s+", " ", value)

    return value.strip()


def _entry_text(entry) -> str:
    """Extract the best available text from an RSS entry."""
    parts = []

    title = _clean_text(entry.get("title"))
    summary = _clean_text(entry.get("summary"))

    if title:
        parts.append(title)

    if summary and summary != title:
        parts.append(summary)

    contents = entry.get("content", [])

    if contents:
        for content in contents:
            value = _clean_text(content.get("value"))

            if value and value not in parts:
                parts.append(value)

    return " ".join(parts).strip()


def _published_at(entry) -> str:
    """Extract an RSS publication timestamp in ISO-8601 UTC format."""
    parsed_time = entry.get("published_parsed") or entry.get("updated_parsed")

    if parsed_time:
        dt = datetime(
            parsed_time.tm_year,
            parsed_time.tm_mon,
            parsed_time.tm_mday,
            parsed_time.tm_hour,
            parsed_time.tm_min,
            parsed_time.tm_sec,
            tzinfo=timezone.utc,
        )

        return dt.isoformat()

    return datetime.now(timezone.utc).isoformat()


def _article_id(source: str, entry) -> str:
    """
    Generate a deterministic article ID.

    Prefer the feed's GUID/id. Fall back to the article URL.
    """
    raw_id = entry.get("id") or entry.get("guid") or entry.get("link") or ""

    raw_value = f"{source}:{raw_id}"

    return hashlib.sha256(raw_value.encode("utf-8")).hexdigest()


def _detect_coin(text: str) -> Optional[str]:
    """
    Detect BTC or ETH relevance using conservative keyword matching.

    Returns None when neither coin is clearly identified.
    """
    normalized = text.lower()

    matches = []

    for symbol, keywords in COIN_KEYWORDS.items():
        if any(
            re.search(rf"\b{re.escape(keyword)}\b", normalized)
            for keyword in keywords
        ):
            matches.append(symbol)

    if len(matches) == 1:
        return matches[0]

    # Ambiguous or unrelated article.
    return None


def parse_feed(feed_url: str) -> List[SentimentMessage]:
    """
    Fetch and parse one RSS feed.

    Returns an empty list when the feed cannot be parsed.
    """
    logger.info(f"Fetching RSS feed: {feed_url}")

    parsed = feedparser.parse(feed_url)

    if getattr(parsed, "bozo", False):
        logger.warning(
            f"RSS feed reported a parsing issue: {feed_url}"
        )

    source = feed_url.split("/")[2]

    collected_at = datetime.now(timezone.utc).isoformat()

    messages = []

    for entry in parsed.entries:
        title = _clean_text(entry.get("title"))

        if not title:
            logger.debug(
                f"Skipping RSS entry without title from {source}"
            )
            continue

        url = entry.get("link", "").strip()

        text = _entry_text(entry)

        message = SentimentMessage(
            source=source,
            article_id=_article_id(source, entry),
            title=title,
            description=_clean_text(entry.get("summary")),
            url=url,
            published_at=_published_at(entry),
            collected_at=collected_at,
            text=text,
            coin=_detect_coin(text),
        )

        messages.append(message)

    logger.info(
        f"Parsed {len(messages)} articles from {source}"
    )

    return messages


def fetch_all_feeds(
    feed_urls: Optional[Iterable[str]] = None,
) -> List[SentimentMessage]:
    """
    Fetch all configured RSS feeds.

    Duplicate article IDs are removed across feeds.
    """
    if feed_urls is None:
        feed_urls = settings.news_rss_feeds

    messages = []
    seen_ids = set()

    for feed_url in feed_urls:
        try:
            feed_messages = parse_feed(feed_url)

            for message in feed_messages:
                if message.article_id in seen_ids:
                    continue

                seen_ids.add(message.article_id)
                messages.append(message)

        except Exception as exc:
            logger.error(
                f"Failed to process RSS feed {feed_url}: {exc}"
            )

    logger.info(
        f"Total unique RSS articles collected: {len(messages)}"
    )

    return messages