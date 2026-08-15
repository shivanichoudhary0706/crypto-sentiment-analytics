"""
Tests for the Week 3 sentiment message schema.
"""

from ingestion.sentiment.schemas import SentimentMessage


def test_sentiment_message_creation():
    message = SentimentMessage(
        source="coindesk",
        article_id="test-001",
        title="Bitcoin market update",
        description="Bitcoin moved higher today.",
        url="https://example.com/article",
        published_at="2026-08-12T10:00:00+00:00",
        collected_at="2026-08-12T10:05:00+00:00",
        text="Bitcoin moved higher today.",
        coin="BTC/USDT",
    )

    assert message.source == "coindesk"
    assert message.article_id == "test-001"
    assert message.title == "Bitcoin market update"
    assert message.coin == "BTC/USDT"


def test_sentiment_message_to_dict():
    message = SentimentMessage(
        source="coindesk",
        article_id="test-002",
        title="Ethereum market update",
        description="Ethereum market news.",
        url="https://example.com/ethereum",
        published_at="2026-08-12T10:00:00+00:00",
        collected_at="2026-08-12T10:05:00+00:00",
        text="Ethereum market news.",
        coin="ETH/USDT",
    )

    result = message.to_dict()

    assert isinstance(result, dict)
    assert result["source"] == "coindesk"
    assert result["article_id"] == "test-002"
    assert result["coin"] == "ETH/USDT"