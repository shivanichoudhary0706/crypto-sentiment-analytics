"""
Live connectivity test for the GDELT 2.0 DOC API.

This test makes a real API request.
"""

from ingestion.sentiment.gdelt_client import fetch_gdelt


def test_live_gdelt():
    articles = fetch_gdelt(
        coin="BTC/USDT",
        max_records=10,
    )

    print(f"\nCollected {len(articles)} GDELT articles")

    for article in articles[:5]:
        print(f"Source: {article.source}")
        print(f"Title: {article.title}")
        print(f"URL: {article.url}")
        print(f"Published: {article.published_at}")
        print(f"Coin: {article.coin}")
        print("-" * 60)

    assert isinstance(articles, list)