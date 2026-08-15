"""
Unit tests for the sentiment Kafka producer.
"""

import json

from ingestion.sentiment.kafka_producer import (
    JSONSerializer,
    StringKeySerializer,
)


def test_json_serializer_serializes_dictionary():
    serializer = JSONSerializer()

    value = {
        "source": "test",
        "article_id": "abc123",
        "title": "Bitcoin rises",
    }

    result = serializer.serialize(
        "raw-sentiment-data",
        value,
    )

    assert isinstance(result, bytes)

    decoded = json.loads(result.decode("utf-8"))

    assert decoded == value


def test_json_serializer_handles_none():
    serializer = JSONSerializer()

    assert (
        serializer.serialize(
            "raw-sentiment-data",
            None,
        )
        is None
    )


def test_string_key_serializer():
    serializer = StringKeySerializer()

    result = serializer.serialize(
        "raw-sentiment-data",
        "article-123",
    )

    assert result == b"article-123"


def test_string_key_serializer_handles_none():
    serializer = StringKeySerializer()

    assert (
        serializer.serialize(
            "raw-sentiment-data",
            None,
        )
        is None
    )