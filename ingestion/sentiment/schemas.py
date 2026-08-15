"""
Message schema for sentiment data published to Kafka's raw-sentiment-data topic.
"""

from dataclasses import dataclass, asdict
from typing import List


@dataclass
class RawSentimentMessage:
    """
    One article/record from a sentiment source (news RSS or GDELT).
    Sentiment scoring itself happens later (Week 5) — this is raw text only.
    """
    source_type: str          # "news_rss" or "gdelt"
    source_name: str          # e.g. "coindesk", "gdelt"
    title: str
    summary: str               # short excerpt/description, may be empty
    url: str
    published_at: str          # ISO 8601 string
    ingested_at: str           # ISO 8601 string, when we pulled it
    coin_tags: List[str]       # e.g. ["BTC/USDT"], can be multiple or empty

    def to_dict(self) -> dict:
        return asdict(self)