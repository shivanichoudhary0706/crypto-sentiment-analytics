"""Unit tests for sentiment_consumer validation (poison-message handling).
No Kafka or database needed: tests the pure parsing functions directly."""

import json
from datetime import timezone

import pytest

from streaming.sentiment_consumer import MalformedMessageError, _decode, _to_row

VALID = {
    "source_type": "news_rss",
    "source_name": "cryptoslate",
    "title": "Bitcoin rallies",
    "summary": "BTC up 5%",
    "url": "https://example.com/a",
    "published_at": "2026-09-25T10:00:00Z",
    "coin_tags": ["BTC/USDT"],
}


def _raw(obj) -> bytes:
    return json.dumps(obj).encode("utf-8")


# ---- happy path -------------------------------------------------------------

def test_valid_message_becomes_row():
    row = _to_row(_decode(_raw(VALID)))
    assert row["url"] == VALID["url"]
    assert row["coin_tags"] == ["BTC/USDT"]
    assert row["time"].tzinfo is not None


def test_naive_timestamp_is_treated_as_utc():
    row = _to_row({**VALID, "published_at": "2026-09-25 10:00:00"})
    assert row["time"].tzinfo == timezone.utc


def test_missing_summary_defaults_to_empty_string():
    msg = {k: v for k, v in VALID.items() if k != "summary"}
    assert _to_row(msg)["summary"] == ""


# ---- poison messages: must raise MalformedMessageError, never crash -----------

def test_not_json_is_malformed():            # manual Test B, message 1
    with pytest.raises(MalformedMessageError):
        _decode(b"hello this is not json")


def test_missing_fields_is_malformed():      # manual Test B, message 2
    with pytest.raises(MalformedMessageError, match="missing required field"):
        _to_row(_decode(_raw({"title": "missing fields test"})))


def test_empty_value_is_malformed():
    with pytest.raises(MalformedMessageError):
        _decode(None)


def test_json_array_is_malformed():
    with pytest.raises(MalformedMessageError):
        _decode(b"[1, 2, 3]")


def test_invalid_utf8_is_malformed():
    with pytest.raises(MalformedMessageError):
        _decode(b"\xff\xfe\xfa")


def test_unparseable_date_is_malformed_not_now():
    """Research rule: never fall back to 'now' (would bias lag estimates)."""
    with pytest.raises(MalformedMessageError, match="published_at"):
        _to_row({**VALID, "published_at": "not a date"})


def test_coin_tags_must_be_a_list():
    with pytest.raises(MalformedMessageError, match="coin_tags"):
        _to_row({**VALID, "coin_tags": "BTC/USDT"})