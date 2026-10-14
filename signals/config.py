"""
Configuration for the signal model: the `signal_model:` section of config.yaml.
All keys are REQUIRED (no silent defaults). `config.settings` is imported lazily
so the pure dataset/model code and its tests need no config file or database.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

FREQ_SECONDS = {"15min": 900, "1h": 3600, "4h": 14400, "1d": 86400}
FREQS = tuple(FREQ_SECONDS)

_REQUIRED_KEYS = ("targets", "input_coins", "freq", "horizon_bars", "buy_threshold",
                  "sell_threshold", "train_window_bars", "min_train_bars", "model", "table")
_REQUIRED_MODEL_KEYS = ("max_iter", "learning_rate", "max_depth", "min_samples_leaf",
                        "l2_regularization", "random_state")


@dataclass(frozen=True)
class SignalConfig:
    targets: tuple[str, ...]        # coins we emit BUY/HOLD/SELL for, e.g. SOL/USDT
    input_coins: tuple[str, ...]    # coins whose price moves are model inputs, e.g. BTC/USDT
    freq: str                       # bar size the model works on, e.g. "1h"
    horizon_bars: int               # predict the move over the next N bars
    buy_threshold: float            # prob_up >= this -> BUY
    sell_threshold: float           # prob_up <= this -> SELL; in between -> HOLD
    train_window_bars: int          # train on the most recent N labelled bars per coin
    min_train_bars: int             # refuse to emit signals with less history than this
    model: dict                     # HistGradientBoostingClassifier parameters
    table: str

    @property
    def bar(self):
        import pandas as pd
        return pd.Timedelta(seconds=FREQ_SECONDS[self.freq])

    @property
    def horizon(self):
        return self.bar * self.horizon_bars


def parse_signal_config(section: dict | None) -> SignalConfig:
    if not section:
        raise KeyError("config.yaml has no `signal_model:` section")
    missing = [k for k in _REQUIRED_KEYS if k not in section]
    if missing:
        raise KeyError(f"config.yaml -> signal_model is missing required key(s): {missing}")
    model = section["model"] or {}
    missing = [k for k in _REQUIRED_MODEL_KEYS if k not in model]
    if missing:
        raise KeyError(f"config.yaml -> signal_model.model is missing required key(s): {missing}")

    cfg = SignalConfig(
        targets=tuple(section["targets"]),
        input_coins=tuple(section["input_coins"]),
        freq=str(section["freq"]),
        horizon_bars=int(section["horizon_bars"]),
        buy_threshold=float(section["buy_threshold"]),
        sell_threshold=float(section["sell_threshold"]),
        train_window_bars=int(section["train_window_bars"]),
        min_train_bars=int(section["min_train_bars"]),
        model=dict(model),
        table=str(section["table"]),
    )
    if not cfg.targets:
        raise ValueError("signal_model.targets is empty")
    if set(cfg.targets) & set(cfg.input_coins):
        raise ValueError("signal_model: a coin cannot be both a target and an input coin")
    if cfg.freq not in FREQS:
        raise ValueError(f"signal_model.freq must be one of {FREQS}")
    if cfg.horizon_bars < 1:
        raise ValueError("signal_model.horizon_bars must be >= 1")
    if not 0 < cfg.sell_threshold < 0.5 < cfg.buy_threshold < 1:
        raise ValueError("signal_model: need 0 < sell_threshold < 0.5 < buy_threshold < 1")
    if not 0 < cfg.min_train_bars <= cfg.train_window_bars:
        raise ValueError("signal_model: need 0 < min_train_bars <= train_window_bars")
    if not re.fullmatch(r"[a-z_][a-z0-9_]{0,62}", cfg.table):
        raise ValueError(f"signal_model.table must be a plain lowercase SQL identifier, got {cfg.table!r}")
    return cfg

@dataclass(frozen=True)
class BacktestConfig:
    test_days: int              # replay the most recent N days
    retrain_every_bars: int     # retrain the model every N bars during the replay (24 = daily)
    fee_bps: float              # cost per position change, in basis points (10 = 0.1%)
    output_dir: str


_REQUIRED_BACKTEST_KEYS = ("test_days", "retrain_every_bars", "fee_bps", "output_dir")


def parse_backtest_config(section: dict | None) -> BacktestConfig:
    if not section:
        raise KeyError("config.yaml has no `signal_model.backtest:` section")
    missing = [k for k in _REQUIRED_BACKTEST_KEYS if k not in section]
    if missing:
        raise KeyError(f"config.yaml -> signal_model.backtest is missing required key(s): {missing}")
    cfg = BacktestConfig(test_days=int(section["test_days"]),
                         retrain_every_bars=int(section["retrain_every_bars"]),
                         fee_bps=float(section["fee_bps"]),
                         output_dir=str(section["output_dir"]))
    if cfg.test_days < 1 or cfg.retrain_every_bars < 1:
        raise ValueError("signal_model.backtest: test_days and retrain_every_bars must be >= 1")
    if not 0 <= cfg.fee_bps < 1000:
        raise ValueError("signal_model.backtest.fee_bps must be between 0 and 1000")
    return cfg


def load_backtest_config() -> BacktestConfig:
    from config.settings import settings
    return parse_backtest_config((settings.signal_model or {}).get("backtest"))

def load_config() -> SignalConfig:
    from config.settings import settings  # lazy: see module docstring
    return parse_signal_config(settings.signal_model)