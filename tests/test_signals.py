"""
Tests for the signal layer (signals/). No database needed.

The key guarantees:
  * the label at bar t is the move AFTER bar t (no look-ahead in features),
  * a model predicting at bar T never trains on a label that needs a price after T,
  * the pipeline learns a real pattern when one exists (planted-signal test),
  * probabilities map to BUY / HOLD / SELL exactly as configured.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from signals.config import parse_signal_config
from signals.dataset import build_model_frame, training_rows
from signals.model import BUY, HOLD, SELL, predict_up, to_action, train
from signals.run_signals import generate_signals

T0 = pd.Timestamp("2026-08-01 00:00", tz="UTC")
TARGETS = ("SOL/USDT", "LTC/USDT")
INPUTS = ("BTC/USDT",)

_CFG = {"targets": list(TARGETS), "input_coins": list(INPUTS), "freq": "1h", "horizon_bars": 1,
        "buy_threshold": 0.55, "sell_threshold": 0.45, "train_window_bars": 2000,
        "min_train_bars": 200, "table": "signals",
        "model": {"max_iter": 100, "learning_rate": 0.1, "max_depth": 3, "min_samples_leaf": 20,
                  "l2_regularization": 1.0, "random_state": 0}}


def make_features(n_bars: int, seed: int = 0, planted: bool = False) -> pd.DataFrame:
    """Synthetic features_bar rows (1h) for BTC + targets.

    planted=True: a positive btc_news_score at bar t makes every target rise at
    bar t+1, and a negative one makes it fall -- a pattern the model must find.
    """
    rng = np.random.default_rng(seed)
    buckets = pd.date_range(T0, periods=n_bars, freq="1h")
    news = rng.choice([-0.8, 0.0, 0.8], size=n_bars, p=[0.25, 0.5, 0.25])
    rows = []
    for i, coin in enumerate(("BTC/USDT", *TARGETS)):
        ret = rng.normal(0, 0.01, n_bars)
        if planted and coin in TARGETS:
            ret[1:] = np.sign(news[:-1]) * np.abs(ret[1:]) + (news[:-1] == 0) * ret[1:]
        close = 100 * (i + 1) * np.exp(np.cumsum(ret))
        rows.append(pd.DataFrame({
            "bucket": buckets, "coin": coin, "close": close, "volume": rng.uniform(1, 2, n_bars),
            "log_return": np.r_[np.nan, np.diff(np.log(close))],
            "realized_vol": np.abs(ret), "ret_vol_short": 0.01, "ret_vol_long": 0.01,
            "is_complete": True,
            "btc_news_score": news, "btc_news_count": (news != 0).astype(int),
            "btc_news_roll": pd.Series(news).rolling(6, min_periods=1).mean().to_numpy(),
        }))
    return pd.concat(rows, ignore_index=True)


# ---------------------------------------------------------------- dataset
def test_label_is_next_bar_move_and_features_are_not_from_the_future():
    feats = make_features(50)
    frame, names = build_model_frame(feats, TARGETS, INPUTS, horizon_bars=1)
    sol = frame[frame["coin"] == "SOL/USDT"].reset_index(drop=True)
    src = feats[feats["coin"] == "SOL/USDT"].reset_index(drop=True)
    # label_t = 1 if close_{t+1} > close_t
    expected = (src["close"].shift(-1) > src["close"]).astype(float).iloc[:-1]
    np.testing.assert_array_equal(sol["label"].iloc[:-1].to_numpy(), expected.to_numpy())
    assert pd.isna(sol["label"].iloc[-1])                      # future unknown for newest bar
    # 'ret' at t is the move DURING bar t (already happened), lag1 is bar t-1
    np.testing.assert_allclose(sol["ret"].iloc[5], src["log_return"].iloc[5])
    np.testing.assert_allclose(sol["ret_lag1"].iloc[5], src["log_return"].iloc[4])
    assert {"btc_ret", "btc_ret_lag1", "btc_rv", "btc_news_score", "coin_sol", "coin_ltc"} <= set(names)
    assert "future_return" not in names and "label" not in names and "close" not in names


def test_training_rows_never_see_labels_from_after_the_prediction_bar():
    frame, _ = build_model_frame(make_features(100), TARGETS, INPUTS, horizon_bars=3)
    predict_at = T0 + pd.Timedelta(hours=60)
    rows = training_rows(frame, predict_at, pd.Timedelta(hours=3), train_window_bars=1000)
    # label of row t uses close at t+3, which must be <= predict_at
    assert rows["bucket"].max() == predict_at - pd.Timedelta(hours=3)
    assert rows["label"].notna().all()


def test_training_window_keeps_latest_rows_per_coin():
    frame, _ = build_model_frame(make_features(300), TARGETS, INPUTS, horizon_bars=1)
    rows = training_rows(frame, T0 + pd.Timedelta(hours=299), pd.Timedelta(hours=1), 100)
    assert rows.groupby("coin").size().tolist() == [100, 100]
    # predicting at bar 299 -> newest usable label is bar 298 -> 100 rows = bars 199..298
    assert rows["bucket"].min() == T0 + pd.Timedelta(hours=199)


def test_price_gap_rows_are_not_labelled():
    feats = make_features(40)
    feats.loc[(feats["coin"] == "SOL/USDT") & (feats["bucket"] == T0 + pd.Timedelta(hours=10)), "close"] = np.nan
    frame, _ = build_model_frame(feats, TARGETS, INPUTS, horizon_bars=1)
    sol = frame[frame["coin"] == "SOL/USDT"].set_index("bucket")
    assert pd.isna(sol.loc[T0 + pd.Timedelta(hours=9), "label"])    # its future is the gap
    assert pd.isna(sol.loc[T0 + pd.Timedelta(hours=10), "label"])   # it is the gap


# ---------------------------------------------------------------- model
def test_model_learns_a_planted_pattern():
    """If news really predicts the next move, out-of-sample accuracy must be high."""
    frame, names = build_model_frame(make_features(1500, seed=1, planted=True), TARGETS, INPUTS, 1)
    split = T0 + pd.Timedelta(hours=1000)
    tr = training_rows(frame, split, pd.Timedelta(hours=1), 5000)
    te = frame[(frame["bucket"] > split) & frame["label"].notna()]
    model = train(tr, names, _CFG["model"])
    acc = ((predict_up(model, te) > 0.5) == (te["label"] == 1)).mean()
    assert acc > 0.65


def test_model_on_pure_noise_is_near_coin_flip():
    frame, names = build_model_frame(make_features(1500, seed=2), TARGETS, INPUTS, 1)
    split = T0 + pd.Timedelta(hours=1000)
    model = train(training_rows(frame, split, pd.Timedelta(hours=1), 5000), names, _CFG["model"])
    te = frame[(frame["bucket"] > split) & frame["label"].notna()]
    acc = ((predict_up(model, te) > 0.5) == (te["label"] == 1)).mean()
    assert 0.4 < acc < 0.6


def test_one_class_history_falls_back_to_base_rate():
    frame, names = build_model_frame(make_features(60), TARGETS, INPUTS, 1)
    rows = frame[frame["label"].notna()].copy()
    rows["label"] = 1.0
    model = train(rows, names, _CFG["model"])
    assert model.estimator is None
    assert (predict_up(model, rows.head(3)) == 1.0).all()


def test_to_action_thresholds():
    acts = to_action([0.70, 0.55, 0.549, 0.50, 0.451, 0.45, 0.10], 0.55, 0.45)
    assert list(acts) == [BUY, BUY, HOLD, HOLD, HOLD, SELL, SELL]


# ---------------------------------------------------------------- end to end (no DB)
def test_generate_signals_one_row_per_target_for_newest_bar():
    cfg = parse_signal_config(_CFG)
    signals, info = generate_signals(make_features(400, seed=3, planted=True), cfg)
    assert sorted(signals["coin"]) == sorted(TARGETS)
    last = T0 + pd.Timedelta(hours=399)
    assert (signals["bucket"] == last).all()
    assert (signals["valid_from"] == last + pd.Timedelta(hours=1)).all()
    assert signals["action"].isin([BUY, HOLD, SELL]).all()
    assert signals["prob_up"].between(0, 1).all()
    assert info["n_train"] == 2 * 399          # bars 0..398 per coin (bar 399's label is unknown)
    assert info["skipped"] == {}


def test_generate_signals_skips_coins_without_enough_history():
    cfg = parse_signal_config({**_CFG, "min_train_bars": 500})
    signals, info = generate_signals(make_features(300), cfg)
    assert signals.empty
    assert set(info["skipped"]) == set(TARGETS)


# ---------------------------------------------------------------- config
@pytest.mark.parametrize("patch", [
    {"buy_threshold": 0.45},                       # buy must be above 0.5
    {"sell_threshold": 0.6},
    {"freq": "2h"},
    {"targets": ["BTC/USDT"]},                     # overlaps an input coin
    {"min_train_bars": 5000},                      # more than the window
    {"table": "signals; drop table x"},
])
def test_config_rejects_bad_values(patch):
    with pytest.raises(ValueError):
        parse_signal_config({**_CFG, **patch})


def test_config_rejects_missing_keys():
    with pytest.raises(KeyError):
        parse_signal_config({k: v for k, v in _CFG.items() if k != "targets"})
    with pytest.raises(KeyError):
        parse_signal_config({**_CFG, "model": {"max_iter": 10}})