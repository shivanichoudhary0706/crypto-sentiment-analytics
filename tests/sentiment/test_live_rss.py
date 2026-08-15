"""
Live RSS connectivity test.

This test intentionally reaches the configured RSS feeds.
It verifies that the feeds are reachable and produce parseable entries.
"""

from ingestion.sentiment.rss_client import fetch_all_feeds


def test_live_rss_feeds():
    messages = fetch_all_feeds()

    print(f"\nCollected {len(messages)} unique RSS articles")

    assert len(messages) > 0

    for message in messages[:5]:
        print(
            f"\nSource: {message.source}"
            f"\nTitle: {message.title}"
            f"\nURL: {message.url}"
            f"\nPublished: {message.published_at}"
            f"\nCoin: {message.coin}"
        )