"""
Kafka producer wrapper for publishing raw sentiment data (news + GDELT).
"""

import json

from kafka import KafkaProducer
from kafka.errors import KafkaError
from kafka.serializer import Serializer

from config.logging_config import get_logger
from config.settings import settings

logger = get_logger(__name__)


class JSONSerializer(Serializer):
    def serialize(self, topic, value):
        return json.dumps(value).encode("utf-8") if value is not None else None


class StringKeySerializer(Serializer):
    def serialize(self, topic, key):
        return key.encode("utf-8") if key else None


def build_sentiment_producer() -> KafkaProducer:
    return KafkaProducer(
        bootstrap_servers=settings.kafka_bootstrap_servers,
        value_serializer=JSONSerializer(),
        key_serializer=StringKeySerializer(),
        acks="all",
        retries=5,
        linger_ms=50,
    )


def publish_sentiment_message(producer: KafkaProducer, message_dict: dict) -> None:
    topic = settings.kafka_topics["sentiment_raw"]
    # Key by source_name so all articles from the same outlet land on the same
    # partition (preserves per-source ordering, useful for dedup logic later)
    key = message_dict.get("source_name")
    try:
        producer.send(topic, key=key, value=message_dict)
    except KafkaError as e:
        logger.error(f"Failed to publish sentiment message from {key}: {e}")
        raise