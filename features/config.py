"""
Configuration for the feature job.

Single source of truth (same rules as lag_detection/config.py):
  * Coins       -> top-level `coins:` list in config/config.yaml
  * Parameters  -> the `features:` section (all keys REQUIRED, no silent defaults)
  * DB login    -> TIMESCALEDB_* secrets via config.settings
  * Date window -> not configured; incremental watermark, or --start/--end on the CLI

`config.settings` is imported lazily so the pure transform code and its tests
need neither a config file nor a database.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import timedelta

# Frequency -> bar length in seconds. market_data holds 1-minute candles.
FREQ_TO_SECONDS = {"15min": 900, "1h": 3600, "4h": 14400, "1d": 86400}
FREQ_TO_PANDAS = {"15min": "15min", "1h": "1h", "4h": "4h", "1d": "1D"}

_REQUIRED_KEYS = ("frequencies", "rolling_short_bars", "rolling_long_bars",
                  "incremental_lookback_bars", "table")


@dataclass(frozen=True)
class FeatureConfig:
    coins: tuple[str, ...]              # 'BTC/USDT' style (sentiment_scores.coin)
    frequencies: tuple[str, ...]
    rolling_short_bars: int
    rolling_long_bars: int
    incremental_lookback_bars: int      # bars re-computed on every incremental run
    table: str

    @property
    def symbol_to_coin(self) -> dict[str, str]:
        """'BTCUSDT' (market_data.symbol) -> 'BTC/USDT' (sentiment_scores.coin)."""
        return {coin_to_symbol(c): c for c in self.coins}

    @property
    def max_freq_seconds(self) -> int:
        return max(FREQ_TO_SECONDS[f] for f in self.frequencies)

    def history_needed(self) -> timedelta:
        """History to read BEFORE the first bar written, so rolling windows and the
        first log return of an incremental run are computed from real data."""
        return timedelta(seconds=(self.rolling_long_bars + 2) * self.max_freq_seconds)

    def lookback(self, freq: str) -> timedelta:
        return timedelta(seconds=self.incremental_lookback_bars * FREQ_TO_SECONDS[freq])


def coin_to_symbol(coin: str) -> str:
    return coin.replace("/", "").upper()


def parse_feature_config(section: dict | None, coins_cfg: list[dict]) -> FeatureConfig:
    """Pure function: raw YAML dicts -> validated FeatureConfig."""
    if not section:
        raise KeyError("config.yaml has no `features:` section")
    missing = [k for k in _REQUIRED_KEYS if k not in section]
    if missing:
        raise KeyError(f"config.yaml -> features is missing required key(s): {missing}")

    coins = tuple(c["symbol"] for c in (coins_cfg or []))
    if not coins:
        raise ValueError("config.yaml -> `coins:` list is empty")

    freqs = tuple(section["frequencies"])
    bad = [f for f in freqs if f not in FREQ_TO_SECONDS]
    if bad or not freqs:
        raise ValueError(f"features.frequencies: unsupported/empty {bad}; use {list(FREQ_TO_SECONDS)}")

    table = str(section["table"])
    # The table name is interpolated as an identifier, so it must be a plain name.
    if not re.fullmatch(r"[a-z_][a-z0-9_]{0,62}", table):
        raise ValueError(f"features.table must be a plain lowercase SQL identifier, got {table!r}")

    cfg = FeatureConfig(
        coins=coins,
        frequencies=freqs,
        rolling_short_bars=int(section["rolling_short_bars"]),
        rolling_long_bars=int(section["rolling_long_bars"]),
        incremental_lookback_bars=int(section["incremental_lookback_bars"]),
        table=table,
    )
    if not 2 <= cfg.rolling_short_bars < cfg.rolling_long_bars:
        raise ValueError("features: need 2 <= rolling_short_bars < rolling_long_bars")
    if cfg.incremental_lookback_bars < 1:
        raise ValueError("features.incremental_lookback_bars must be >= 1")
    return cfg


def load_config() -> FeatureConfig:
    from config.settings import settings  # lazy: see module docstring
    return parse_feature_config(settings.features, settings.coins)