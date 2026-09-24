"""
Tests for RawSentimentMessage — the schema the live pollers
(news_rss_poller.py, gdelt_poller.py) publish to the raw-sentiment-data topic.

    python -m pytest tests/sentiment/test_schemas.py -v
"""
import json
from dataclasses import fields

from ingestion.sentiment.schemas import RawSentimentMessage

EXPECTED_FIELDS = [
    "source_type", "source_name", "title", "summary", "url",
    "published_at", "ingested_at", "coin_tags",
]


def make_message(**overrides) -> RawSentimentMessage:
    values = {
        "source_type": "news_rss",
        "source_name": "coindesk",
        "title": "Bitcoin market update",
        "summary": "Bitcoin moved higher today.",
        "url": "https://example.com/article",
        "published_at": "2026-08-12T10:00:00+00:00",
        "ingested_at": "2026-08-12T10:05:00+00:00",
        "coin_tags": ["BTC/USDT"],
    }
    values.update(overrides)
    return RawSentimentMessage(**values)


def test_schema_fields_are_stable():
    """Downstream consumers depend on these exact field names. A rename must fail this test."""
    assert [f.name for f in fields(RawSentimentMessage)] == EXPECTED_FIELDS


def test_message_creation():
    msg = make_message()
    assert msg.source_type == "news_rss"
    assert msg.source_name == "coindesk"
    assert msg.coin_tags == ["BTC/USDT"]


def test_to_dict_has_all_fields():
    result = make_message().to_dict()
    assert isinstance(result, dict)
    assert list(result) == EXPECTED_FIELDS
    assert result["url"] == "https://example.com/article"


def test_to_dict_is_json_serialisable():
    """Messages travel through Kafka as JSON — this must never raise."""
    msg = make_message(coin_tags=["BTC/USDT", "ETH/USDT"])
    decoded = json.loads(json.dumps(msg.to_dict()))
    assert decoded["coin_tags"] == ["BTC/USDT", "ETH/USDT"]


def test_empty_coin_tags_allowed():
    """Untagged articles are valid messages (they are stored, just not scored)."""
    msg = make_message(coin_tags=[], source_type="gdelt", source_name="gdelt", summary="")
    assert msg.to_dict()["coin_tags"] == []


def test_to_dict_returns_a_copy():
    """Mutating the dict (e.g. while serialising) must not change the original message."""
    msg = make_message()
    d = msg.to_dict()
    d["coin_tags"].append("ETH/USDT")
    assert msg.coin_tags == ["BTC/USDT"]