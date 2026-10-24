"""Unit tests for the health-check planning logic. No Kafka, DB or network needed."""
from datetime import date, datetime, timedelta, timezone

import pytest

from maintenance.config import parse_maintenance_config
from maintenance.ranges import coalesce, day_ranges, market_gaps, missing_minutes, tags_changed

T0 = datetime(2026, 6, 1, tzinfo=timezone.utc)
M = timedelta(minutes=1)
GRACE = timedelta(minutes=5)


def test_no_candles_means_everything_missing():
    now = T0 + timedelta(days=1)
    assert market_gaps(T0, now, None, None, [], GRACE) == [(T0, now)]


def test_leading_internal_and_trailing_gaps():
    first, last = T0 + timedelta(days=2), T0 + timedelta(days=5)
    now = last + timedelta(hours=3)
    internal = [(T0 + timedelta(days=3), T0 + timedelta(days=3, hours=1))]
    gaps = market_gaps(T0, now, first, last, internal, GRACE)
    assert gaps == [(T0, first), internal[0], (last + M, now)]
    assert missing_minutes(gaps) == 2 * 24 * 60 + 60 + 179


def test_live_stream_lag_is_not_a_gap():
    last = T0 + timedelta(days=1)
    gaps = market_gaps(T0, last + timedelta(minutes=3), T0, last, [], GRACE)
    assert gaps == []


def test_coalesce_joins_nearby_gaps_only():
    a = (T0, T0 + 10 * M)
    b = (T0 + 20 * M, T0 + 30 * M)           # 10 min after a -> joined
    c = (T0 + timedelta(days=2), T0 + timedelta(days=2) + M)   # far -> separate
    assert coalesce([c, b, a], timedelta(minutes=1000)) == [(T0, T0 + 30 * M), c]


def test_day_ranges_groups_consecutive_days():
    days = [date(2026, 6, 3), date(2026, 6, 1), date(2026, 6, 2), date(2026, 6, 9)]
    assert day_ranges(days) == [(date(2026, 6, 1), date(2026, 6, 4)),
                                (date(2026, 6, 9), date(2026, 6, 10))]


def test_tags_changed_ignores_order():
    assert not tags_changed(["ETH/USDT", "BTC/USDT"], ["BTC/USDT", "ETH/USDT"])
    assert tags_changed([], ["MARKET"])
    assert tags_changed(None, ["MARKET"])
    assert tags_changed(["MARKET"], ["ETH/USDT"])


SECTION = {"history_start": "2026-06-01", "min_articles_per_day": 1,
           "news_backfill_coins": ["BTC/USDT", "MARKET"], "retag_source_types": ["news_rss"],
           "kafka_max_lag": 1000, "feature_lock_wait_seconds": 60,
           "stale_minutes": {"market": 5, "news": 60, "scoring": 60, "signals": 90}}
COINS = [{"symbol": "BTC/USDT"}, {"symbol": "LINK/USDT"}]


def test_config_parses():
    cfg = parse_maintenance_config(SECTION, COINS)
    assert cfg.market_symbols == ("BTCUSDT", "LINKUSDT")
    assert cfg.history_start == T0


def test_config_rejects_missing_keys():
    with pytest.raises(KeyError, match="kafka_max_lag"):
        parse_maintenance_config({k: v for k, v in SECTION.items() if k != "kafka_max_lag"}, COINS)
    with pytest.raises(KeyError, match="stale_minutes"):
        parse_maintenance_config({**SECTION, "stale_minutes": {"market": 5}}, COINS)


def test_ago_handles_never():
    from maintenance.checks import ago
    assert ago(None) == "never"
    assert ago(timedelta(minutes=5)) == "5m ago"
    assert ago(timedelta(hours=3, minutes=2)) == "3h 2m ago"


def test_recent_day_ranges_clips_old_gaps():
    from maintenance.ranges import recent_day_ranges
    runs = [(date(2026, 6, 1), date(2026, 7, 1)),        # old -> dropped
            (date(2026, 10, 19), date(2026, 10, 23)),    # straddles -> clipped
            (date(2026, 10, 23), date(2026, 10, 24))]    # recent -> kept
    assert recent_day_ranges(runs, date(2026, 10, 21)) == [
        (date(2026, 10, 21), date(2026, 10, 23)), (date(2026, 10, 23), date(2026, 10, 24))]


def test_auto_news_days_defaults_and_validates():
    assert parse_maintenance_config(SECTION, COINS).auto_news_backfill_days == 3
    assert parse_maintenance_config({**SECTION, "auto_news_backfill_days": 0}, COINS).auto_news_backfill_days == 0
    with pytest.raises(ValueError):
        parse_maintenance_config({**SECTION, "auto_news_backfill_days": -1}, COINS)
