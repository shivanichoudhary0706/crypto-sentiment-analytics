"""
GDELT 2.0 DOC API client.

Fetches recent cryptocurrency news from the GDELT DOC API and
normalizes results into SentimentMessage objects.

This module does not publish to Kafka. Kafka integration is handled
separately by the Week 3 ingestion runner.
"""

import hashlib
import time
from datetime import datetime, timezone
from typing import List, Optional

import requests

from config.logging_config import get_logger
from ingestion.sentiment.schemas import SentimentMessage


logger = get_logger(__name__)

GDELT_DOC_API_URL = "https://api.gdeltproject.org/api/v2/doc/doc"

REQUEST_TIMEOUT_SECONDS = 30
MAX_RETRIES = 3
RETRY_DELAYS_SECONDS = (5, 15, 30)

USER_AGENT = (
    "crypto-sentiment-analytics/1.0 "
    "(academic research; contact: thesis-project)"
)


def _article_id(url: str) -> str:
    """Generate a deterministic article ID from the article URL."""
    return hashlib.sha256(url.encode("utf-8")).hexdigest()


def _detect_coin(text: str) -> Optional[str]:
    """
    Detect whether an article is related to BTC or ETH.

    Returns None when both coins are mentioned or neither is mentioned.
    """
    normalized = text.lower()

    bitcoin_match = any(
        term in normalized
        for term in ("bitcoin", "btc")
    )

    ethereum_match = any(
        term in normalized
        for term in ("ethereum", "ether", "eth")
    )

    if bitcoin_match and not ethereum_match:
        return "BTC/USDT"

    if ethereum_match and not bitcoin_match:
        return "ETH/USDT"

    return None


def _parse_gdelt_date(value: str) -> str:
    """
    Convert GDELT timestamp YYYYMMDDHHMMSS to ISO-8601 UTC.
    """
    try:
        dt = datetime.strptime(value, "%Y%m%d%H%M%S")
        dt = dt.replace(tzinfo=timezone.utc)
        return dt.isoformat()
    except (TypeError, ValueError):
        return datetime.now(timezone.utc).isoformat()


def _build_query(coin: str) -> str:
    """Build a conservative GDELT query for one supported coin."""
    if coin == "BTC/USDT":
        return "(bitcoin OR btc)"

    if coin == "ETH/USDT":
        return "(ethereum OR ether OR eth)"

    raise ValueError(f"Unsupported coin: {coin}")


def _request_gdelt(params: dict) -> Optional[dict]:
    """
    Request data from GDELT with retry/backoff handling.

    HTTP 429 is treated as a rate-limit response and retried with
    increasing delays.
    """
    headers = {
        "User-Agent": USER_AGENT,
        "Accept": "application/json",
    }

    for attempt in range(MAX_RETRIES + 1):
        try:
            response = requests.get(
                GDELT_DOC_API_URL,
                params=params,
                headers=headers,
                timeout=REQUEST_TIMEOUT_SECONDS,
            )

            if response.status_code == 429:
                if attempt >= MAX_RETRIES:
                    logger.error(
                        "GDELT rate limit persisted after "
                        f"{MAX_RETRIES} retries."
                    )
                    return None

                delay = RETRY_DELAYS_SECONDS[attempt]

                logger.warning(
                    "GDELT returned HTTP 429. "
                    f"Retrying in {delay} seconds "
                    f"(attempt {attempt + 1}/{MAX_RETRIES})."
                )

                time.sleep(delay)
                continue

            response.raise_for_status()

            return response.json()

        except requests.RequestException as exc:
            if attempt >= MAX_RETRIES:
                logger.error(
                    f"GDELT request failed after retries: {exc}"
                )
                return None

            delay = RETRY_DELAYS_SECONDS[attempt]

            logger.warning(
                f"GDELT request error: {exc}. "
                f"Retrying in {delay} seconds "
                f"(attempt {attempt + 1}/{MAX_RETRIES})."
            )

            time.sleep(delay)

        except ValueError as exc:
            logger.error(
                f"GDELT returned invalid JSON: {exc}"
            )
            return None

    return None


def fetch_gdelt(
    coin: str,
    max_records: int = 50,
) -> List[SentimentMessage]:
    """
    Fetch recent GDELT news for one configured coin.

    GDELT requires no API key.
    """
    query = _build_query(coin)

    params = {
        "query": query,
        "mode": "ArtList",
        "format": "json",
        "maxrecords": max_records,
        "sort": "datedesc",
    }

    logger.info(
        f"Fetching GDELT articles for {coin} "
        f"(max_records={max_records})"
    )

    data = _request_gdelt(params)

    if data is None:
        return []

    articles = data.get("articles", [])

    if not isinstance(articles, list):
        logger.warning(
            "GDELT response contained an unexpected "
            "'articles' structure."
        )
        return []

    collected_at = datetime.now(timezone.utc).isoformat()

    messages: List[SentimentMessage] = []

    for article in articles:
        if not isinstance(article, dict):
            continue

        url = (article.get("url") or "").strip()

        if not url:
            continue

        title = (article.get("title") or "").strip()

        if not title:
            continue

        source = (
            article.get("domain")
            or article.get("sourcecountry")
            or "gdelt"
        )

        published_at = _parse_gdelt_date(
            article.get("seendate")
        )

        detected_coin = _detect_coin(title)

        # GDELT query results can still contain articles mentioning
        # multiple cryptocurrencies. We only keep articles that can
        # be confidently assigned to the requested coin.
        if detected_coin != coin:
            continue

        message = SentimentMessage(
            source=f"gdelt:{source}",
            article_id=_article_id(url),
            title=title,
            description="",
            url=url,
            published_at=published_at,
            collected_at=collected_at,
            text=title,
            coin=coin,
        )

        messages.append(message)

    logger.info(
        f"Parsed {len(messages)} GDELT articles for {coin}"
    )

    return messages