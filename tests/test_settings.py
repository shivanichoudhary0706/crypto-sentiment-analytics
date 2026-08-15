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
print("News RSS feeds:", settings.news_rss_feeds)
print("GDELT query terms:", settings.gdelt_query_terms)
print("Coin keywords:", settings.coin_keywords)
print("FinBERT model:", settings.finbert_model_name)
print("TimescaleDB URL (password masked):",
      settings.timescaledb_url.replace(settings.timescaledb_password or "", "****"))
print("\nConfig loaded successfully.")