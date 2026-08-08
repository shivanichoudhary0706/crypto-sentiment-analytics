"""
Sanity check for config loading. Run directly with:
    python tests/test_settings.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.settings import settings

print("Coins configured:", settings.coins)
print("Kafka bootstrap servers:", settings.kafka_bootstrap_servers)
print("Kafka topics:", settings.kafka_topics)
print("Reddit subreddits:", settings.reddit_subreddits)
print("News RSS feeds:", settings.news_rss_feeds)
print("FinBERT model:", settings.finbert_model_name)
print("TimescaleDB URL (password masked):",
      settings.timescaledb_url.replace(settings.timescaledb_password or "", "****"))
print("\nConfig loaded successfully.")