"""Tests for per-frequency signal config (signal_model.per_freq)."""
import copy

import pytest

from signals.config import parse_signal_config

BASE = {
    "targets": ["SOL/USDT", "LTC/USDT", "LINK/USDT"],
    "input_coins": ["BTC/USDT", "ETH/USDT"],
    "freq": "1h",
    "horizon_bars": 1,
    "buy_threshold": 0.55,
    "sell_threshold": 0.45,
    "train_window_bars": 720,
    "min_train_bars": 240,
    "model": {"max_iter": 200, "learning_rate": 0.05, "max_depth": 3,
              "min_samples_leaf": 20, "l2_regularization": 1.0, "random_state": 42},
    "table": "signals",
    "backtest": {"test_days": 30},
    "per_freq": {"4h": {"train_window_bars": 360, "min_train_bars": 120}},
}


def section(**per_freq_4h):
    s = copy.deepcopy(BASE)
    if per_freq_4h:
        s["per_freq"]["4h"] = per_freq_4h
    return s


def test_default_freq_is_unchanged():
    cfg = parse_signal_config(section())
    assert cfg.freq == "1h" and cfg.train_window_bars == 720 and cfg.min_train_bars == 240


def test_explicit_default_freq_is_unchanged():
    assert parse_signal_config(section(), "1h") == parse_signal_config(section())


def test_4h_overrides_bar_counts_and_inherits_the_rest():
    cfg = parse_signal_config(section(), "4h")
    assert cfg.freq == "4h"
    assert (cfg.train_window_bars, cfg.min_train_bars) == (360, 120)
    assert (cfg.buy_threshold, cfg.sell_threshold) == (0.55, 0.45)
    assert cfg.targets == ("SOL/USDT", "LTC/USDT", "LINK/USDT")
    assert cfg.bar.total_seconds() == 4 * 3600


def test_4h_can_override_thresholds_and_one_model_param():
    cfg = parse_signal_config(section(train_window_bars=360, min_train_bars=120,
                                      buy_threshold=0.6, model={"max_depth": 4}), "4h")
    assert cfg.buy_threshold == 0.6
    assert cfg.model["max_depth"] == 4 and cfg.model["max_iter"] == 200  # merged, not replaced


def test_unconfigured_freq_is_an_error():
    with pytest.raises(KeyError, match="per_freq"):
        parse_signal_config(section(), "1d")


def test_per_freq_must_set_bar_counts():
    with pytest.raises(KeyError, match="min_train_bars"):
        parse_signal_config(section(train_window_bars=360), "4h")


def test_per_freq_cannot_change_coins_or_table():
    with pytest.raises(ValueError, match="table"):
        parse_signal_config(section(train_window_bars=360, min_train_bars=120, table="x"), "4h")


def test_per_freq_values_are_still_validated():
    with pytest.raises(ValueError, match="min_train_bars"):
        parse_signal_config(section(train_window_bars=100, min_train_bars=120), "4h")