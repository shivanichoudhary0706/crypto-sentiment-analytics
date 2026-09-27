"""Unit tests for market_consumer validation. No Kafka or DB needed."""

import json
from datetime import timezone

import pytest

from streaming.market_consumer import MalformedMessageError, _decode, _to_row

START_MS = 1790330400000  # a minute boundary in Sep 2026
VALID = {
    "kline_start_time": START_MS,
    "kline_close_time": START_MS + 59_999,
    "symbol": "BTC/USDT",
    "exchange": "binance",
    "open": 65000.0, "high": 65100.0, "low": 64900.0, "close": 65050.0,
    "volume": 12.5,
    "is_closed": True,
}


def _raw(obj) -> bytes:
    return json.dumps(obj).encode("utf-8")


# ---- happy path -------------------------------------------------------------

def test_valid_candle_becomes_row():
    row = _to_row(_decode(_raw(VALID)))
    assert row["symbol"] == "BTC/USDT"
    assert row["close"] == 65050.0
    assert row["time"].tzinfo == timezone.utc
    assert row["kline_close_time"] > row["time"]


def test_string_prices_are_accepted():
    row = _to_row({**VALID, "open": "65000.00", "close": "65050.00"})
    assert row["open"] == 65000.0


def test_zero_volume_is_valid():
    assert _to_row({**VALID, "volume": 0})["volume"] == 0.0


# ---- malformed / corrupt data ------------------------------------------------

def test_not_json_is_malformed():
    with pytest.raises(MalformedMessageError):
        _decode(b"hello this is not json")


def test_missing_fields_is_malformed():
    with pytest.raises(MalformedMessageError, match="missing required field"):
        _to_row({"symbol": "BTC/USDT"})


@pytest.mark.parametrize("field,value", [
    ("close", 0), ("low", -1), ("open", "abc"), ("high", float("nan")),
    ("close", float("inf")), ("open", True),
])
def test_bad_prices_are_rejected(field, value):
    with pytest.raises(MalformedMessageError):
        _to_row({**VALID, field: value})


def test_negative_volume_is_rejected():
    with pytest.raises(MalformedMessageError, match="volume"):
        _to_row({**VALID, "volume": -5})


def test_high_below_low_is_rejected():
    with pytest.raises(MalformedMessageError, match="inconsistent OHLC"):
        _to_row({**VALID, "high": 64800.0})


def test_close_time_before_start_is_rejected():
    with pytest.raises(MalformedMessageError, match="not after start"):
        _to_row({**VALID, "kline_close_time": START_MS - 1})