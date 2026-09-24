"""
Week 6 tests — synthetic data with a KNOWN planted lag.
If the pipeline cannot recover a lag we planted, its answer on real data is meaningless.

    python -m pytest tests/test_lag_detection.py -v
"""
import numpy as np
import pandas as pd
import pytest

from lag_detection.rolling import rolling_lag_scan
from lag_detection.series_builder import align_series
from lag_detection.stats_tests import (
    cross_correlation, granger_test, make_stationary, peak_lag, stationarity_test,
)

TRUE_LAG = 3


def synthetic(n=1000, true_lag=TRUE_LAG, beta=0.6, seed=42):
    """sentiment ~ AR(1); return_t = beta * sentiment_{t-lag} + noise."""
    rng = np.random.default_rng(seed)
    x = np.zeros(n)
    e = rng.normal(size=n)
    for t in range(1, n):
        x[t] = 0.3 * x[t - 1] + e[t]
    y = rng.normal(size=n)
    y[true_lag:] += beta * x[:-true_lag]
    idx = pd.date_range("2026-01-01", periods=n, freq="1h", tz="UTC")
    return pd.DataFrame({"sentiment": x, "log_return": y, "n_articles": 1}, index=idx)


def test_ccf_recovers_planted_lag():
    df = synthetic()
    pk = peak_lag(cross_correlation(df["sentiment"], df["log_return"], 12), "lead")
    assert pk["lag"] == TRUE_LAG and pk["significant_bonf"]


def test_granger_forward_significant_reverse_not():
    df = synthetic()
    fwd, _ = granger_test(df, "sentiment", "log_return", 12)
    rev, _ = granger_test(df, "log_return", "sentiment", 12)
    assert fwd["significant_at_chosen"] and fwd["p_value_at_chosen"] < 0.001
    assert fwd["chosen_lag"] >= TRUE_LAG          # model must reach back far enough
    assert not rev["significant_at_chosen"]


def test_no_relationship_gives_no_signal():
    rng = np.random.default_rng(7)
    idx = pd.date_range("2026-01-01", periods=1000, freq="1h", tz="UTC")
    df = pd.DataFrame({"sentiment": rng.normal(size=1000), "log_return": rng.normal(size=1000)}, index=idx)
    fwd, _ = granger_test(df, "sentiment", "log_return", 12)
    assert not fwd["significant_at_chosen"]


def test_random_walk_is_differenced():
    rng = np.random.default_rng(1)
    idx = pd.date_range("2026-01-01", periods=500, freq="1h", tz="UTC")
    df = pd.DataFrame({"price": rng.normal(size=500).cumsum()}, index=idx)
    assert stationarity_test(df["price"])["verdict"] != "stationary"
    _, report = make_stationary(df, ["price"])
    assert report["price"]["transform"] == "diff1"


def test_constant_sentiment_raises():
    idx = pd.date_range("2026-01-01", periods=100, freq="1h", tz="UTC")
    df = pd.DataFrame({"sentiment": 0.0}, index=idx)
    with pytest.raises(ValueError):
        make_stationary(df, ["sentiment"])


def test_align_series_fill_and_returns():
    idx = pd.date_range("2026-01-01", periods=4, freq="1h", tz="UTC")
    market = pd.DataFrame({"close": [100.0, 110.0, 99.0, 99.0], "n_candles": [60, 60, 60, 60]}, index=idx)
    sent = pd.DataFrame({"sent_mean": [0.5], "sent_sum": [1.0], "n_articles": [2]}, index=idx[[2]])
    df, diag = align_series(market, sent, "1h", "zero")
    assert len(df) == 3                                   # first bar has no return
    assert df["log_return"].iloc[0] == pytest.approx(np.log(1.1))
    assert list(df["sentiment"]) == [0.0, 0.5, 0.0]
    assert diag["coverage"] == pytest.approx(1 / 3)
    df_ff, _ = align_series(market, sent, "1h", "ffill")
    assert list(df_ff["sentiment"]) == [0.0, 0.5, 0.5]


def test_align_drops_incomplete_last_bar():
    idx = pd.date_range("2026-01-01", periods=3, freq="1h", tz="UTC")
    market = pd.DataFrame({"close": [100.0, 101.0, 102.0], "n_candles": [60, 60, 12]}, index=idx)
    df, _ = align_series(market, pd.DataFrame(), "1h")
    assert len(df) == 1


def test_rolling_finds_stable_lag():
    table, summary = rolling_lag_scan(synthetic(), "sentiment", "log_return",
                                      window=168, step=84, max_lag=12)
    assert summary["n_windows_ok"] >= 5
    assert summary["ccf_peak_lag_mode"] == TRUE_LAG


# --------------------------------------------------------------------------- #
# Config: coins come from `coins:`, no silent defaults, no dates in config
# --------------------------------------------------------------------------- #
from datetime import datetime, timezone  # noqa: E402
from pathlib import Path  # noqa: E402

from lag_detection.config import parse_lag_config  # noqa: E402
from lag_detection.series_builder import resolve_window  # noqa: E402

SECTION = {
    "frequencies": ["1h", "4h"], "max_lag": 24, "sentiment_column": "finbert_compound",
    "fill_method": "zero", "target": "log_return", "significance": 0.05,
    "min_coverage_warn": 0.2, "output_dir": "reports/lag",
    "rolling": {"enabled": True, "window_bars": 168, "step_bars": 24, "min_coverage": 0.2},
}
COINS = [{"symbol": "BTC/USDT", "exchange": "binance"}, {"symbol": "SOL/USDT", "exchange": "binance"}]


def test_coins_come_from_top_level_list():
    cfg = parse_lag_config(SECTION, COINS, Path("."))
    assert cfg.coins == ("BTC/USDT", "SOL/USDT")
    assert cfg.start is None and cfg.end is None          # window not configured


def test_missing_key_is_loud():
    broken = {k: v for k, v in SECTION.items() if k != "max_lag"}
    with pytest.raises(KeyError, match="max_lag"):
        parse_lag_config(broken, COINS, Path("."))


def test_missing_section_is_loud():
    with pytest.raises(KeyError):
        parse_lag_config({}, COINS, Path("."))


class _FakeCursor:
    def __init__(self, row): self.row = row
    def __enter__(self): return self
    def __exit__(self, *a): return False
    def execute(self, *a, **k): pass
    def fetchone(self): return self.row


class _FakeConn:
    def __init__(self, row): self.row = row
    def cursor(self): return _FakeCursor(self.row)


def test_window_auto_detected_from_data():
    first = datetime(2026, 8, 12, tzinfo=timezone.utc)
    last = datetime(2026, 9, 22, tzinfo=timezone.utc)
    start, end, source = resolve_window(_FakeConn((first, last)), "BTC/USDT", None, None)
    assert (start, source) == (first, "auto")
    assert end == datetime(2026, 9, 22, 0, 1, tzinfo=timezone.utc)   # exclusive, last candle included


def test_window_can_be_pinned():
    first = datetime(2026, 8, 12, tzinfo=timezone.utc)
    last = datetime(2026, 9, 22, tzinfo=timezone.utc)
    pin_s, pin_e = datetime(2026, 8, 20, tzinfo=timezone.utc), datetime(2026, 9, 1, tzinfo=timezone.utc)
    start, end, source = resolve_window(_FakeConn((first, last)), "BTC/USDT", pin_s, pin_e)
    assert (start, end, source) == (pin_s, pin_e, "pinned")


def test_window_no_data_raises():
    with pytest.raises(ValueError, match="No market_data"):
        resolve_window(_FakeConn((None, None)), "BTC/USDT", None, None)