"""
Polls GDELT's public 2.0 DOC API on a schedule for crypto-related articles.
No API key required.

REVISED (see decision log): earlier testing suggested GDELT rate-limiting
was a hard, unrecoverable IP-level block, so this poller originally made
exactly ONE attempt per cycle and gave up on any failure. A later batch
test (historical backfill script, same GDELT DOC API) showed that 429s
here are real, standard rate limits that DO recover with backoff — some
cycles needed 3-4 attempts with waits of 60-250s+ before succeeding.

So this poller now retries within a single cycle using exponential backoff
with jitter, but the total retry time is capped (RETRY_BUDGET_SECONDS) so
a bad cycle can never run past the next scheduled poll or starve other
concurrent pollers. If the budget is exhausted, it logs and waits for the
next natural cycle — same fallback behavior as before, just with a real
recovery attempt first.

GDELT sometimes returns rate-limiting as a 200 OK with a plain-text
message instead of an HTTP 429 status, so both cases are detected
explicitly and treated the same way for retry purposes.
"""

import asyncio
import random
import time
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

# --- Retry/backoff tuning -----------------------------------------------
# NOTE: currently hardcoded here rather than config.yaml, to avoid guessing
# at settings.py structure. Fine to promote to config later if wanted —
# consistent with the project's config-driven principle, just not urgent.
MAX_RETRIES_PER_CYCLE = 4
INITIAL_BACKOFF_SECONDS = 45
MAX_BACKOFF_SECONDS = 240
BACKOFF_JITTER_MIN = 5
BACKOFF_JITTER_MAX = 20
# Hard ceiling on total wall-clock time spent retrying in a single cycle.
# Keeps a bad cycle from bleeding into the next scheduled poll.
RETRY_BUDGET_SECONDS = 480  # 8 minutes
REQUEST_TIMEOUT_SECONDS = 30
# --------------------------------------------------------------------------


def _is_rate_limited_text(response_text: str) -> bool:
    return "Please limit requests" in response_text


def _sync_gdelt_request(query: str, max_records: int) -> requests.Response:
    """Blocking call, run off the event loop via asyncio.to_thread."""
    params = {
        "query": query,
        "mode": "artlist",
        "maxrecords": max_records,
        "format": "json",
        "sort": "datedesc",
    }
    return requests.get(
        GDELT_DOC_API_URL,
        params=params,
        headers=REQUEST_HEADERS,
        timeout=REQUEST_TIMEOUT_SECONDS,
    )


def _compute_backoff(attempt: int) -> float:
    base = min(INITIAL_BACKOFF_SECONDS * (2 ** (attempt - 1)), MAX_BACKOFF_SECONDS)
    jitter = random.uniform(BACKOFF_JITTER_MIN, BACKOFF_JITTER_MAX)
    return base + jitter


async def _fetch_gdelt_articles(query: str, max_records: int) -> list:
    cycle_start = time.monotonic()

    for attempt in range(1, MAX_RETRIES_PER_CYCLE + 1):
        elapsed = time.monotonic() - cycle_start
        if elapsed >= RETRY_BUDGET_SECONDS:
            logger.warning(
                f"GDELT retry budget ({RETRY_BUDGET_SECONDS}s) exhausted "
                f"after {attempt - 1} attempt(s). Skipping this cycle."
            )
            return []

        try:
            response = await asyncio.to_thread(_sync_gdelt_request, query, max_records)
        except requests.exceptions.RequestException as e:
            logger.error(f"GDELT fetch failed (attempt {attempt}/{MAX_RETRIES_PER_CYCLE}): {e}")
            if attempt == MAX_RETRIES_PER_CYCLE:
                return []
            await asyncio.sleep(min(_compute_backoff(attempt), RETRY_BUDGET_SECONDS - elapsed))
            continue

        # Rate-limited: either a genuine 429 or the 200-with-plaintext quirk.
        if response.status_code == 429 or _is_rate_limited_text(response.text):
            reason = "429" if response.status_code == 429 else "200-with-plaintext-notice"
            if attempt == MAX_RETRIES_PER_CYCLE:
                logger.warning(
                    f"GDELT still rate-limited ({reason}) after "
                    f"{MAX_RETRIES_PER_CYCLE} attempts. Skipping this cycle."
                )
                return []
            wait_time = _compute_backoff(attempt)
            remaining_budget = RETRY_BUDGET_SECONDS - (time.monotonic() - cycle_start)
            if remaining_budget <= 0:
                logger.warning(
                    f"GDELT rate-limited ({reason}) and retry budget exhausted. "
                    "Skipping this cycle."
                )
                return []
            wait_time = min(wait_time, remaining_budget)
            logger.info(
                f"GDELT rate-limited ({reason}), attempt {attempt}/{MAX_RETRIES_PER_CYCLE}. "
                f"Backing off {wait_time:.1f}s..."
            )
            await asyncio.sleep(wait_time)
            continue

        # Genuinely empty body — separate failure mode, not worth retrying
        # within budget since it hasn't historically been transient.
        if not response.text.strip():
            logger.warning("GDELT returned an empty response. Skipping this cycle.")
            return []

        try:
            response.raise_for_status()
            data = response.json()
        except (requests.exceptions.HTTPError, ValueError) as e:
            logger.error(f"GDELT response error (attempt {attempt}/{MAX_RETRIES_PER_CYCLE}): {e}")
            if attempt == MAX_RETRIES_PER_CYCLE:
                return []
            await asyncio.sleep(min(_compute_backoff(attempt), RETRY_BUDGET_SECONDS - elapsed))
            continue

        articles = data.get("articles", [])
        if attempt > 1:
            logger.info(f"GDELT succeeded on attempt {attempt}/{MAX_RETRIES_PER_CYCLE}.")
        return articles

    return []


async def poll_gdelt_forever() -> None:
    producer = build_sentiment_producer()
    seen_urls = set()
    message_count = 0

    query = " OR ".join(settings.gdelt_query_terms)

    while True:
        articles = await _fetch_gdelt_articles(query, settings.gdelt_max_records)

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