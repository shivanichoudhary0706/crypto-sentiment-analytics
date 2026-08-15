"""
Unit tests for the GDELT 2.0 DOC API client.
"""

from ingestion.sentiment.gdelt_client import (
    _build_query,
    _detect_coin,
    _parse_gdelt_date,
)


def test_build_bitcoin_query():
    result = _build_query("BTC/USDT")

    assert "bitcoin" in result
    assert "btc" in result


def test_build_ethereum_query():
    result = _build_query("ETH/USDT")

    assert "ethereum" in result
    assert "ether" in result


def test_build_query_rejects_unknown_coin():
    try:
        _build_query("DOGE/USDT")
        assert False
    except ValueError:
        pass


def test_detect_bitcoin():
    result = _detect_coin(
        "Bitcoin price rises as BTC demand increases."
    )

    assert result == "BTC/USDT"


def test_detect_ethereum():
    result = _detect_coin(
        "Ethereum network activity pushes ETH higher."
    )

    assert result == "ETH/USDT"


def test_detect_ambiguous_article():
    result = _detect_coin(
        "Bitcoin and Ethereum both gain today."
    )

    assert result is None


def test_parse_gdelt_date():
    result = _parse_gdelt_date("20260812153000")

    assert result == "2026-08-12T15:30:00+00:00"


def test_parse_invalid_gdelt_date():
    result = _parse_gdelt_date("invalid")

    assert result.endswith("+00:00")