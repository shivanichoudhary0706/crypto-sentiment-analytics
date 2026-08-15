"""
Centralized configuration loader.

Merges non-secret settings from config.yaml with secrets from .env,
so the rest of the codebase never reads os.environ or opens config.yaml directly —
it just imports `settings` from here.
"""

import os
from pathlib import Path

import yaml
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_YAML_PATH = PROJECT_ROOT / "config" / "config.yaml"
ENV_PATH = PROJECT_ROOT / ".env"

load_dotenv(dotenv_path=ENV_PATH)


def _load_yaml_config() -> dict:
    if not CONFIG_YAML_PATH.exists():
        raise FileNotFoundError(
            f"config.yaml not found at {CONFIG_YAML_PATH}. "
            "Copy/create it before running any pipeline component."
        )
    with open(CONFIG_YAML_PATH, "r") as f:
        return yaml.safe_load(f)


class Settings:
    def __init__(self):
        yaml_cfg = _load_yaml_config()

        # --- Non-secret settings (from config.yaml) ---
        self.coins = yaml_cfg["coins"]
        self.kafka_topics = yaml_cfg["kafka"]["topics"]
        self.kafka_consumer_group = yaml_cfg["kafka"]["consumer_group"]

        self.news_rss_feeds = yaml_cfg["sentiment_sources"]["news_rss"]["feeds"]
        self.news_poll_interval = yaml_cfg["sentiment_sources"]["news_rss"]["poll_interval_seconds"]

        self.gdelt_query_terms = yaml_cfg["sentiment_sources"]["gdelt"]["query_terms"]
        self.gdelt_poll_interval = yaml_cfg["sentiment_sources"]["gdelt"]["poll_interval_seconds"]
        self.gdelt_max_records = yaml_cfg["sentiment_sources"]["gdelt"]["max_records"]

        self.coin_keywords = yaml_cfg["coin_keywords"]

        self.finbert_model_name = yaml_cfg["sentiment_models"]["finbert_model_name"]
        self.sentiment_batch_size = yaml_cfg["sentiment_models"]["batch_size"]
        self.max_lag_minutes = yaml_cfg["lag_detection"]["max_lag_minutes"]
        self.resample_frequency = yaml_cfg["lag_detection"]["resample_frequency"]
        self.significance_level = yaml_cfg["lag_detection"]["significance_level"]
        self.sentiment_positive_threshold = yaml_cfg["signals"]["sentiment_positive_threshold"]
        self.sentiment_negative_threshold = yaml_cfg["signals"]["sentiment_negative_threshold"]
        self.log_level = yaml_cfg["logging"]["level"]
        self.log_dir = PROJECT_ROOT / yaml_cfg["logging"]["log_dir"]

        # --- Secrets (from .env / environment) ---
        self.exchange_name = os.getenv("EXCHANGE_NAME")
        self.exchange_ws_url = os.getenv("EXCHANGE_WS_URL")
        self.exchange_api_key = os.getenv("EXCHANGE_API_KEY")
        self.exchange_api_secret = os.getenv("EXCHANGE_API_SECRET")

        self.kafka_bootstrap_servers = os.getenv(
            "KAFKA_BOOTSTRAP_SERVERS", yaml_cfg["kafka"]["bootstrap_servers"]
        )

        self.timescaledb_host = os.getenv("TIMESCALEDB_HOST")
        self.timescaledb_port = os.getenv("TIMESCALEDB_PORT")
        self.timescaledb_db = os.getenv("TIMESCALEDB_DB")
        self.timescaledb_user = os.getenv("TIMESCALEDB_USER")
        self.timescaledb_password = os.getenv("TIMESCALEDB_PASSWORD")

        self.api_host = os.getenv("API_HOST", "0.0.0.0")
        self.api_port = int(os.getenv("API_PORT", "8000"))

    @property
    def timescaledb_url(self) -> str:
        return (
            f"postgresql://{self.timescaledb_user}:{self.timescaledb_password}"
            f"@{self.timescaledb_host}:{self.timescaledb_port}/{self.timescaledb_db}"
        )


settings = Settings()