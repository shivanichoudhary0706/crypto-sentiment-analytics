"""Unit tests for coin_tagger. No Kafka or DB needed."""
from ingestion.sentiment.coin_tagger import MARKET_TAG, tag_coins

KW = {"BTC/USDT": ["bitcoin", "btc"], "ETH/USDT": ["ethereum", "eth", "ether"]}


def test_btc_only():
    assert tag_coins("Bitcoin breaks $100k", KW) == ["BTC/USDT"]


def test_eth_only_is_not_market():
    # Regression: the old tagger returned inside the loop after checking BTC only,
    # so ETH-only articles were tagged MARKET.
    assert tag_coins("Ethereum gas fees fall", KW) == ["ETH/USDT"]


def test_both_coins_tagged():
    # Regression: the old tagger dropped the second coin.
    assert tag_coins("BTC and ETH rally together", KW) == ["BTC/USDT", "ETH/USDT"]


def test_no_coin_is_market():
    assert tag_coins("SEC publishes new crypto guidance", KW) == [MARKET_TAG]


def test_whole_words_only():
    # "eth" must not match inside other words.
    assert tag_coins("Something happened whether or not the method works", KW) == [MARKET_TAG]


def test_case_and_possessive():
    assert tag_coins("ETH's upgrade and BITCOIN's halving", KW) == ["BTC/USDT", "ETH/USDT"]


def test_empty_text():
    assert tag_coins("", KW) == [MARKET_TAG]
    assert tag_coins(None, KW) == [MARKET_TAG]
