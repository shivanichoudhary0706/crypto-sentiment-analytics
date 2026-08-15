"""
Unit tests for the RSS ingestion client.
"""

import time
from types import SimpleNamespace

from ingestion.sentiment.rss_client import (
    _article_id,
    _clean_text,
    _detect_coin,
    _entry_text,
    _published_at,
)


def test_clean_text_removes_html_and_extra_whitespace():
    value = "<p>Bitcoin&nbsp; is <b>rising</b>.</p>"

    result = _clean_text(value)

    assert "Bitcoin" in result
    assert "<p>" not in result
    assert "<b>" not in result


def test_detect_coin_bitcoin():
    result = _detect_coin(
        "Bitcoin BTC investors are watching the market."
    )

    assert result == "BTC/USDT"


def test_detect_coin_ethereum():
    result = _detect_coin(
        "Ethereum ETH network activity increased today."
    )

    assert result == "ETH/USDT"


def test_detect_coin_ambiguous_article():
    result = _detect_coin(
        "Bitcoin and Ethereum both moved higher."
    )

    assert result is None


def test_detect_coin_unrelated_article():
    result = _detect_coin(
        "Traditional financial markets opened higher today."
    )

    assert result is None


def test_entry_text_combines_title_and_summary():
    entry = SimpleNamespace(
        title="Bitcoin rises",
        summary="BTC gained during the session.",
    )

    result = _entry_text(entry.__dict__)

    assert "Bitcoin rises" in result
    assert "BTC gained during the session." in result


def test_article_id_is_deterministic():
    entry = {
        "id": "article-123",
        "link": "https://example.com/article",
    }

    first = _article_id("example.com", entry)
    second = _article_id("example.com", entry)

    assert first == second
    assert len(first) == 64


def test_published_at_uses_feed_timestamp():
    entry = {
        "published_parsed": time.struct_time(
            (
                2026,
                8,
                12,
                10,
                30,
                0,
                2,
                224,
                0,
            )
        )
    }

    result = _published_at(entry)

    assert result.startswith("2026-08-12T10:30:00")
    assert result.endswith("+00:00")