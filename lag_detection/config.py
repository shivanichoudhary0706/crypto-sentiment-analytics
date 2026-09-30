"""
Lag-detection configuration.

Single source of truth:
  * Coins        -> the top-level `coins:` list in config/config.yaml
                    (the same list ingestion uses — add a coin there, it is analysed here).
  * Parameters   -> the `lag_detection:` section, read via config.settings.
  * DB login     -> TIMESCALEDB_* secrets, read via config.settings.
  * Date window  -> NOT configured. Auto-detected from market_data per coin,
                    optionally pinned with --start/--end on the command line.

All `lag_detection:` keys are REQUIRED. No silent defaults: a missing key is a
loud error, because silent defaults are how a stale config goes unnoticed.
`config.settings` is imported lazily so the pure statistics code (and its
tests) does not need a config file or a database.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

# Whitelists: column names cannot be SQL parameters, so we only ever
# interpolate values from these tuples (via psycopg2.sql.Identifier).
ALLOWED_SENTIMENT_COLUMNS = ("finbert_compound", "vader_compound")
ALLOWED_FILL_METHODS = ("zero", "ffill")
ALLOWED_TARGETS = ("log_return", "abs_return")

FREQ_TO_PG_INTERVAL = {"15min": "15 minutes", "1h": "1 hour", "4h": "4 hours", "1d": "1 day"}
FREQ_TO_PANDAS = {"15min": "15min", "1h": "1h", "4h": "4h", "1d": "1D"}
CANDLES_PER_BAR = {"15min": 15, "1h": 60, "4h": 240, "1d": 1440}  # market_data = 1-minute candles

_REQUIRED_KEYS = ("frequencies", "max_lag", "sentiment_column", "fill_method", "target",
                  "significance", "min_coverage_warn", "output_dir", "rolling")
_REQUIRED_ROLLING_KEYS = ("enabled", "window_bars", "step_bars", "min_coverage")


@dataclass(frozen=True)
class RollingConfig:
    enabled: bool
    window_bars: int
    step_bars: int
    min_coverage: float


@dataclass(frozen=True)
class DBConfig:
    host: str
    port: int
    dbname: str
    user: str
    password: str


@dataclass(frozen=True)
class LagConfig:
    coins: tuple[str, ...]
    frequencies: tuple[str, ...]
    max_lag: int                      # in bars, not minutes
    sentiment_column: str
    fill_method: str
    target: str
    significance: float
    min_coverage_warn: float
    output_dir: Path
    rolling: RollingConfig
    start: datetime | None = None     # None = auto-detect from data
    end: datetime | None = None       # None = auto-detect from data (exclusive)


def to_utc(value: Any) -> datetime | None:
    """Accept None, date, datetime or ISO string; return tz-aware UTC datetime."""
    if value is None:
        return None
    if isinstance(value, datetime):
        dt = value
    elif isinstance(value, date):
        dt = datetime(value.year, value.month, value.day)
    else:
        dt = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    return dt.replace(tzinfo=timezone.utc) if dt.tzinfo is None else dt.astimezone(timezone.utc)


def _require(section: dict, keys: tuple[str, ...], where: str) -> None:
    missing = [k for k in keys if k not in section]
    if missing:
        raise KeyError(f"config.yaml -> {where} is missing required key(s): {missing}")


def parse_lag_config(section: dict | None, coins_cfg: list[dict], project_root: Path) -> LagConfig:
    """Pure function: raw YAML dicts -> validated LagConfig (unit-testable)."""
    if not section:
        raise KeyError("config.yaml has no `lag_detection:` section")
    _require(section, _REQUIRED_KEYS, "lag_detection")
    roll = section["rolling"] or {}
    _require(roll, _REQUIRED_ROLLING_KEYS, "lag_detection.rolling")

    coins = tuple(c["symbol"] for c in (coins_cfg or []))
    if not coins:
        raise ValueError("config.yaml -> `coins:` list is empty")

    cfg = LagConfig(
        coins=coins,
        frequencies=tuple(section["frequencies"]),
        max_lag=int(section["max_lag"]),
        sentiment_column=str(section["sentiment_column"]),
        fill_method=str(section["fill_method"]),
        target=str(section["target"]),
        significance=float(section["significance"]),
        min_coverage_warn=float(section["min_coverage_warn"]),
        output_dir=project_root / str(section["output_dir"]),
        rolling=RollingConfig(
            enabled=bool(roll["enabled"]),
            window_bars=int(roll["window_bars"]),
            step_bars=int(roll["step_bars"]),
            min_coverage=float(roll["min_coverage"]),
        ),
    )
    validate(cfg)
    return cfg


def validate(cfg: LagConfig) -> None:
    if cfg.sentiment_column not in ALLOWED_SENTIMENT_COLUMNS:
        raise ValueError(f"sentiment_column must be one of {ALLOWED_SENTIMENT_COLUMNS}")
    if cfg.fill_method not in ALLOWED_FILL_METHODS:
        raise ValueError(f"fill_method must be one of {ALLOWED_FILL_METHODS}")
    if cfg.target not in ALLOWED_TARGETS:
        raise ValueError(f"target must be one of {ALLOWED_TARGETS}")
    bad = [f for f in cfg.frequencies if f not in FREQ_TO_PG_INTERVAL]
    if bad or not cfg.frequencies:
        raise ValueError(f"Unsupported/empty frequencies {bad}; use {list(FREQ_TO_PG_INTERVAL)}")
    if cfg.max_lag < 1:
        raise ValueError("max_lag must be >= 1")
    if not 0 < cfg.significance < 1:
        raise ValueError("significance must be in (0, 1)")
    if not 1 <= cfg.rolling.step_bars <= cfg.rolling.window_bars:
        raise ValueError("rolling.step_bars must be between 1 and rolling.window_bars")
    if cfg.start and cfg.end and cfg.end <= cfg.start:
        raise ValueError("end must be after start")


def load_config() -> LagConfig:
    from config.settings import PROJECT_ROOT, settings  # lazy: see module docstring
    return parse_lag_config(settings.lag_detection, settings.coins, PROJECT_ROOT)


_REQUIRED_EVENT_STUDY_KEYS = ("cause_coins", "price_targets", "sentiment_column", "threshold",
                              "cluster_window_hours", "horizon_hours", "n_baseline_samples",
                              "n_bootstrap", "min_events", "significance", "onset_min_run",
                              "min_effect_car", "output_dir")


@dataclass(frozen=True)
class EventStudyConfig:
    cause_coins: tuple[str, ...]      # sentiment SOURCE coins (e.g. BTC/USDT, ETH/USDT)
    price_targets: tuple[str, ...]    # price coins to test, incl. spillover targets (e.g. + LTC/USDT)
    sentiment_column: str
    threshold: float                 # |score| >= threshold to count as a strong article
    cluster_window_hours: float
    horizon_hours: int
    n_baseline_samples: int
    n_bootstrap: int
    min_events: int                  # skip a (cause, target) pair with fewer events than this
    significance: float
    onset_min_run: int               # consecutive significant hours required to call it "onset"
    min_effect_car: float            # |mean CAR| floor -- filters noise-level "significant" hours
    output_dir: Path
    start: datetime | None = None
    end: datetime | None = None


def parse_event_study_config(section: dict | None, project_root: Path) -> EventStudyConfig:
    """Pure function: raw YAML dict -> validated EventStudyConfig (unit-testable)."""
    if not section:
        raise KeyError("config.yaml has no `event_study:` section")
    _require(section, _REQUIRED_EVENT_STUDY_KEYS, "event_study")
    cfg = EventStudyConfig(
        cause_coins=tuple(section["cause_coins"]),
        price_targets=tuple(section["price_targets"]),
        sentiment_column=str(section["sentiment_column"]),
        threshold=float(section["threshold"]),
        cluster_window_hours=float(section["cluster_window_hours"]),
        horizon_hours=int(section["horizon_hours"]),
        n_baseline_samples=int(section["n_baseline_samples"]),
        n_bootstrap=int(section["n_bootstrap"]),
        min_events=int(section["min_events"]),
        significance=float(section["significance"]),
        onset_min_run=int(section["onset_min_run"]),
        min_effect_car=float(section["min_effect_car"]),
        output_dir=project_root / str(section["output_dir"]),
    )
    validate_event_study(cfg)
    return cfg


def validate_event_study(cfg: EventStudyConfig) -> None:
    if cfg.sentiment_column not in ALLOWED_SENTIMENT_COLUMNS:
        raise ValueError(f"event_study.sentiment_column must be one of {ALLOWED_SENTIMENT_COLUMNS}")
    if not cfg.cause_coins:
        raise ValueError("event_study.cause_coins is empty")
    if not cfg.price_targets:
        raise ValueError("event_study.price_targets is empty")
    if cfg.threshold <= 0:
        raise ValueError("event_study.threshold must be > 0 (compared against |score|)")
    if cfg.cluster_window_hours <= 0:
        raise ValueError("event_study.cluster_window_hours must be > 0")
    if cfg.horizon_hours < 1:
        raise ValueError("event_study.horizon_hours must be >= 1")
    if cfg.min_events < 1:
        raise ValueError("event_study.min_events must be >= 1")
    if not 0 < cfg.significance < 1:
        raise ValueError("event_study.significance must be in (0, 1)")
    if cfg.onset_min_run < 1:
        raise ValueError("event_study.onset_min_run must be >= 1")
    if cfg.min_effect_car < 0:
        raise ValueError("event_study.min_effect_car must be >= 0")
    if cfg.start and cfg.end and cfg.end <= cfg.start:
        raise ValueError("event_study: end must be after start")


def load_event_study_config() -> EventStudyConfig:
    from config.settings import PROJECT_ROOT, settings  # lazy: see module docstring
    return parse_event_study_config(settings.event_study, PROJECT_ROOT)


def load_db_config() -> DBConfig:
    from config.settings import settings
    values = {
        "TIMESCALEDB_HOST": settings.timescaledb_host,
        "TIMESCALEDB_PORT": settings.timescaledb_port,
        "TIMESCALEDB_DB": settings.timescaledb_db,
        "TIMESCALEDB_USER": settings.timescaledb_user,
        "TIMESCALEDB_PASSWORD": settings.timescaledb_password,
    }
    missing = [k for k, v in values.items() if not v]
    if missing:
        raise EnvironmentError(f".env is missing: {missing}")
    return DBConfig(
        host=settings.timescaledb_host,
        port=int(settings.timescaledb_port),
        dbname=settings.timescaledb_db,
        user=settings.timescaledb_user,
        password=settings.timescaledb_password,
    )