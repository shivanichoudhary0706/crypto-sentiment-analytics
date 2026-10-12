from datetime import date

from batch.export_to_parquet import partition_path, safe_value


def test_safe_value_flattens_slash():
    assert safe_value("BTC/USDT") == "BTC-USDT"
    assert safe_value("BTCUSDT") == "BTCUSDT"


def test_partition_paths_are_hive_style():
    p = partition_path("sentiment_scores", "BTC/USDT", date(2026, 6, 1))
    assert p.parts[-3:] == ("coin_key=BTC-USDT", "date=2026-06-01", "part-0.parquet")
    m = partition_path("market_data", "ETHUSDT", date(2026, 6, 1))
    assert m.parts[-3] == "symbol=ETHUSDT"