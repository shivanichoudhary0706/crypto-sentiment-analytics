"""
Polls GDELT's public 2.0 DOC API on a schedule for crypto-related articles.
No API key required.

Note: GDELT rate-limiting is IP-level and documented to worsen with quick
retries (see https://github.com/alex9smith/gdelt-doc-api/issues/22 and
GDELT's own "Behind The Scenes: API Quotas" post). So this makes exactly
ONE attempt per poll cycle — if rate-limited, it logs and waits for the
next natural 15-minute cycle rather than retrying immediately, which is
the behavior GDELT's own community has found actually clears blocks.
"""

import asyncio
from datetime import datetime, timezone

import requests

from config.logging_config import get_logger
from config.settings import settings
from ingestion.sentiment.coin_tagger import tag_coins
from ingestion.sentiment.kafka_producer import build_sentiment_producer, publish_sentiment_message
from ingestion.sentiment.schemas import RawSentimentMessage

logger = get_logger(__name__)

GDELT_DOC_API_URL = "https://api.gdeltproject.org/api/v2/doc/doc"

# GDELT has been observed filtering non-browser User-Agents before even
# checking quota. A real browser UA string is the documented community fix.
REQUEST_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    )
}


def _fetch_gdelt_articles(query: str, max_records: int) -> list:
    params = {
        "query": query,
        "mode": "artlist",
        "maxrecords": max_records,
        "format": "json",
        "sort": "datedesc",
    }
    try:
        response = requests.get(
            GDELT_DOC_API_URL, params=params, headers=REQUEST_HEADERS, timeout=30
        )
        if response.status_code == 429:
            logger.warning(
                "GDELT rate-limited us. Skipping this cycle — "
                "will try again at the next scheduled poll rather than retrying now "
                "(quick retries are known to prolong GDELT rate-limit blocks)."
            )
            return []

        response.raise_for_status()
        data = response.json()
        return data.get("articles", [])

    except requests.exceptions.RequestException as e:
        logger.error(f"GDELT fetch failed: {e}")
        return []


async def poll_gdelt_forever() -> None:
    producer = build_sentiment_producer()
    seen_urls = set()
    message_count = 0

    query = " OR ".join(settings.gdelt_query_terms)

    while True:
        articles = _fetch_gdelt_articles(query, settings.gdelt_max_records)

        new_count = 0
        for article in articles:
            url = article.get("url")
            if not url or url in seen_urls:
                continue
            seen_urls.add(url)

            title = article.get("title", "")

            message = RawSentimentMessage(
                source_type="gdelt",
                source_name="gdelt",
                title=title,
                summary="",
                url=url,
                published_at=article.get("seendate", datetime.now(timezone.utc).isoformat()),
                ingested_at=datetime.now(timezone.utc).isoformat(),
                coin_tags=tag_coins(title),
            )
            publish_sentiment_message(producer, message.to_dict())
            message_count += 1
            new_count += 1

        if new_count:
            logger.info(f"[gdelt] published {new_count} new articles "
                        f"(total so far: {message_count})")
        elif not articles:
            logger.info("[gdelt] no articles returned this cycle")

        await asyncio.sleep(settings.gdelt_poll_interval)