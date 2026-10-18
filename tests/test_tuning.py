"""Tests for the choose-then-confirm tuning helpers (signals/run_tuning.py)."""
from __future__ import annotations

import pandas as pd
import pytest

from signals.config import parse_signal_config
from signals.run_tuning import candidate_config, parse_tuning_config, select_best
from tests.test_signals import _CFG

BASE = parse_signal_config({**_CFG, "train_window_bars": 2160, "min_train_bars": 500})


def test_candidate_keeps_the_same_history_in_days():
    c4 = candidate_config(BASE, {"freq": "4h", "buy_threshold": 0.58, "sell_threshold": 0.42})
    assert (c4.freq, c4.buy_threshold, c4.sell_threshold) == ("4h", 0.58, 0.42)
    assert c4.train_window_bars == 540          # 90 days of 4h bars (= 2160 hourly bars)
    assert c4.min_train_bars == 125
    assert c4.targets == BASE.targets and c4.model == BASE.model
    c1 = candidate_config(BASE, {"freq": "1h", "buy_threshold": 0.55, "sell_threshold": 0.45})
    assert c1.train_window_bars == 2160


def test_candidate_with_invalid_thresholds_is_rejected():
    with pytest.raises(ValueError):
        candidate_config(BASE, {"freq": "1h", "buy_threshold": 0.45, "sell_threshold": 0.55})


def test_select_best_uses_price_news_return_then_fewer_trades():
    res = pd.DataFrame([
        {"candidate": 0, "variant": "price+news", "strategy_return": 0.05, "trades": 300},
        {"candidate": 0, "variant": "price_only", "strategy_return": 0.50, "trades": 10},   # ignored
        {"candidate": 1, "variant": "price+news", "strategy_return": 0.08, "trades": 200},
        {"candidate": 2, "variant": "price+news", "strategy_return": 0.08, "trades": 90},
    ])
    assert select_best(res) == 2


def test_tuning_config_validation():
    good = {"choose_days": 30, "confirm_days": 30, "output_dir": "reports/tuning",
            "candidates": [{"freq": "1h", "buy_threshold": 0.55, "sell_threshold": 0.45},
                           {"freq": "4h", "buy_threshold": 0.58, "sell_threshold": 0.42}]}
    assert len(parse_tuning_config(good).candidates) == 2
    with pytest.raises(ValueError):
        parse_tuning_config({**good, "candidates": good["candidates"][:1]})
    with pytest.raises(ValueError):
        parse_tuning_config({**good, "candidates": [{"freq": "1h"}, {"freq": "4h"}]})
    with pytest.raises(KeyError):
        parse_tuning_config({k: v for k, v in good.items() if k != "confirm_days"})