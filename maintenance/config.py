"""
Configuration for tools/health_check.py.

Same rules as the other modules:
  * Coins       -> top-level `coins:` list in config/config.yaml
  * Parameters  -> the `maintenance:` section (validated here, so a typo fails loudly)
  * Other jobs' settings (features, signal_model, ...) are read from their own sections
`config.settings` is imported lazily so the pure helpers and their tests need no config file.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from lag_detection.config import to_utc

_REQUIRED_KEYS = ("history_start", "min_articles_per_day", "news_backfill_coins",
                  "retag_source_types", "kafka_max_lag", "stale_minutes",
                  "feature_lock_wait_seconds")
_STALE_KEYS = ("market", "news", "scoring", "signals")

CONTAINERS = ("crypto-kafka", "crypto-kafka-ui", "crypto-timescaledb")
OPTIONAL_CONTAINERS = ("crypto-adminer",)


@dataclass(frozen=True)
class MaintenanceConfig:
    history_start: datetime
    market_symbols: tuple[str, ...]          # 'BTCUSDT' style (market_data.symbol)
    min_articles_per_day: int
    news_backfill_coins: tuple[str, ...]     # 'BTC/USDT' / 'MARKET' (raw_sentiment.coin_tags)
    retag_source_types: tuple[str, ...]
    kafka_max_lag: int
    stale_minutes: dict
    feature_lock_wait_seconds: int
    auto_news_backfill_days: int = 3         # news gaps this recent are filled without --with-news


def parse_maintenance_config(section: dict | None, coins_cfg: list[dict]) -> MaintenanceConfig:
    """Pure function: raw YAML dicts -> validated MaintenanceConfig."""
    if not section:
        raise KeyError("config.yaml has no `maintenance:` section")
    missing = [k for k in _REQUIRED_KEYS if k not in section]
    if missing:
        raise KeyError(f"config.yaml -> maintenance is missing required key(s): {missing}")
    stale = dict(section["stale_minutes"] or {})
    missing_stale = [k for k in _STALE_KEYS if k not in stale]
    if missing_stale:
        raise KeyError(f"maintenance.stale_minutes is missing {missing_stale}")

    symbols = tuple(c["symbol"].replace("/", "").upper() for c in (coins_cfg or []))
    if not symbols:
        raise ValueError("config.yaml -> `coins:` list is empty")

    cfg = MaintenanceConfig(
        history_start=to_utc(section["history_start"]),
        market_symbols=symbols,
        min_articles_per_day=int(section["min_articles_per_day"]),
        news_backfill_coins=tuple(section["news_backfill_coins"] or ()),
        retag_source_types=tuple(section["retag_source_types"] or ()),
        kafka_max_lag=int(section["kafka_max_lag"]),
        stale_minutes={k: float(stale[k]) for k in _STALE_KEYS},
        feature_lock_wait_seconds=int(section["feature_lock_wait_seconds"]),
        auto_news_backfill_days=int(section.get("auto_news_backfill_days", 3)),
    )
    if cfg.min_articles_per_day < 1:
        raise ValueError("maintenance.min_articles_per_day must be >= 1")
    if cfg.auto_news_backfill_days < 0:
        raise ValueError("maintenance.auto_news_backfill_days must be >= 0 (0 = only with --with-news)")
    if any(v <= 0 for v in cfg.stale_minutes.values()):
        raise ValueError("maintenance.stale_minutes values must be > 0")
    return cfg


def load_config() -> MaintenanceConfig:
    from config.settings import settings  # lazy: see module docstring
    return parse_maintenance_config(settings.maintenance, settings.coins)
