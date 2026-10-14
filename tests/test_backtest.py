"""
Tests for the walk-forward backtest (signals/backtest.py). No database needed.

The key guarantees:
  * no look-ahead: changing prices AFTER a moment never changes predictions made BEFORE it,
  * the news-vs-price comparison detects news value when (and only when) it exists,
  * BUY/SELL/HOLD -> position -> P&L (with fees) is computed exactly as documented.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from signals.backtest import evaluate, strategy_returns, walk_forward
from signals.config import parse_backtest_config
from signals.dataset import build_model_frame
from tests.test_signals import INPUTS, T0, TARGETS, _CFG, make_features

H = pd.Timedelta(hours=1)
KW = dict(bar=H, horizon=H, retrain_every_bars=24, train_window_bars=2000,
          model_params=_CFG["model"], buy_threshold=0.55, sell_threshold=0.45)


def _frame(n=900, seed=1, planted=True):
    return build_model_frame(make_features(n, seed=seed, planted=planted), TARGETS, INPUTS, 1)


def test_no_look_ahead_future_prices_do_not_change_past_predictions():
    frame, names = _frame(600)
    start, end = T0 + 400 * H, T0 + 590 * H
    cut = T0 + 496 * H                                     # a retrain-block boundary (400 + 4*24)
    base = walk_forward(frame, names, test_start=start, test_end=end, **KW)

    tampered = frame.copy()
    after = tampered["bucket"] > cut
    tampered.loc[after, "close"] *= np.random.default_rng(9).uniform(0.5, 1.5, after.sum())
    # labels derived from close must be recomputed exactly as build_model_frame would
    tampered = tampered.sort_values(["coin", "bucket"])
    nxt = tampered.groupby("coin")["close"].shift(-1)
    tampered["label"] = (nxt > tampered["close"]).astype(float).where(nxt.notna())
    changed = walk_forward(tampered, names, test_start=start, test_end=end, **KW)

    key = ["variant", "coin", "bucket"]
    a = base[base["bucket"] < cut].sort_values(key).reset_index(drop=True)
    b = changed[changed["bucket"] < cut].sort_values(key).reset_index(drop=True)
    assert len(a) > 0
    np.testing.assert_allclose(a["prob_up"], b["prob_up"])


def test_news_variant_wins_when_news_really_predicts():
    frame, names = _frame(900, seed=1, planted=True)
    preds = walk_forward(frame, names, test_start=T0 + 600 * H, test_end=T0 + 899 * H, **KW)
    _, m = evaluate(preds, fee_bps=10)
    allrow = m[m["coin"].str.startswith("ALL")].set_index("variant")
    assert allrow.loc["price+news", "accuracy"] > 0.65
    assert allrow.loc["price_only", "accuracy"] < 0.6
    assert allrow.loc["price+news", "auc"] > allrow.loc["price_only", "auc"] + 0.1
    assert allrow.loc["price+news", "buy_hit_rate"] > 0.65


def test_on_pure_noise_neither_variant_has_skill():
    frame, names = _frame(900, seed=4, planted=False)
    preds = walk_forward(frame, names, test_start=T0 + 600 * H, test_end=T0 + 899 * H, **KW)
    _, m = evaluate(preds, fee_bps=10)
    for r in m[m["coin"].str.startswith("ALL")].itertuples():
        assert 0.4 < r.auc < 0.6


def test_both_variants_score_identical_bars():
    frame, names = _frame(500)
    preds = walk_forward(frame, names, test_start=T0 + 400 * H, test_end=T0 + 499 * H, **KW)
    counts = preds.groupby("variant").size()
    assert counts["price+news"] == counts["price_only"] > 0
    assert set(preds["variant"]) == {"price+news", "price_only"}


def test_strategy_positions_fees_and_returns():
    p = pd.DataFrame({
        "bucket": pd.date_range(T0, periods=6, freq="1h"),
        "action": ["HOLD", "BUY", "HOLD", "SELL", "HOLD", "BUY"],
        "next_return": [0.10, 0.02, -0.01, 0.05, 0.03, 0.01],
    })
    t = strategy_returns(p, fee_bps=10)
    assert t["position"].tolist() == [0, 1, 1, 0, 0, 1]      # HOLD keeps the previous position
    assert t["trade"].tolist() == [False, True, False, True, False, True]
    expected = [0.0, 0.02 - 0.001, -0.01, -0.001, 0.0, 0.01 - 0.001]
    np.testing.assert_allclose(t["strategy_return"], expected)


def test_evaluate_reports_every_coin_and_a_portfolio_row():
    frame, names = _frame(500)
    preds = walk_forward(frame, names, test_start=T0 + 400 * H, test_end=T0 + 499 * H, **KW)
    traded, m = evaluate(preds, fee_bps=10)
    assert set(m["coin"]) == {"SOL/USDT", "LTC/USDT", "ALL (equal weight)"}
    assert len(m) == 3 * 2                                   # (2 coins + ALL) x 2 variants
    assert m["coin"].iloc[-1].startswith("ALL")              # portfolio rows printed last
    assert (m["buy"] + m["sell"] + m["hold"] == m["bars"]).iloc[:4].all()
    assert {"position", "strategy_return", "fee"} <= set(traded.columns)


@pytest.mark.parametrize("patch", [{"test_days": 0}, {"retrain_every_bars": 0}, {"fee_bps": -1}])
def test_backtest_config_validation(patch):
    good = {"test_days": 60, "retrain_every_bars": 24, "fee_bps": 10, "output_dir": "reports/backtest"}
    assert parse_backtest_config(good).fee_bps == 10
    with pytest.raises(ValueError):
        parse_backtest_config({**good, **patch})