"""
Walk-forward backtest of the signal model. Pure pandas/sklearn, no DB.

How it replays history without cheating:
  * The test period is split into blocks of `retrain_every_bars` (e.g. 24 = one day).
  * At the start of each block the model is retrained using ONLY labels that were
    already known at that moment (same rule as live: training_rows()).
  * That model then predicts every bar in the block. Nothing from inside or after
    the block is ever used for training.
So every prediction below is one the live system could genuinely have made.

Two variants are run on identical bars, which isolates what news adds:
  * "price+news" -- every feature (the live model)
  * "price_only" -- the same model with all *_news_* columns removed

Trading rule used to turn signals into money (long-only spot, one coin each):
  * BUY  -> hold the coin for the next bar
  * SELL -> hold cash for the next bar
  * HOLD -> keep whatever position we already have
  * every change of position pays `fee_bps` (Binance spot: 10 bps = 0.1%)
The portfolio splits money equally across the target coins.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from signals.dataset import news_columns, training_rows
from signals.model import BUY, SELL, predict_up, to_action, train

VARIANTS = ("price+news", "price_only")


def walk_forward(frame: pd.DataFrame, feature_names: list[str], *, test_start, test_end,
                 bar: pd.Timedelta, horizon: pd.Timedelta, retrain_every_bars: int,
                 train_window_bars: int, model_params: dict, buy_threshold: float,
                 sell_threshold: float) -> pd.DataFrame:
    """Predictions for every priced bar in [test_start, test_end), both variants.

    Returns one row per (variant, coin, bucket): prob_up, action, label,
    next_return (simple return of the coin over the next bar, for P&L).
    """
    frame = frame.sort_values(["coin", "bucket"]).copy()
    nxt = frame.groupby("coin")["close"].shift(-1)
    frame["next_return"] = nxt / frame["close"] - 1.0

    no_news = [f for f in feature_names if f not in set(news_columns(feature_names))]
    variant_features = {"price+news": feature_names, "price_only": no_news}

    test = frame[(frame["bucket"] >= test_start) & (frame["bucket"] < test_end)
                 & frame["close"].notna() & frame["next_return"].notna()]
    out = []
    block_start = pd.Timestamp(test_start)
    step = bar * retrain_every_bars
    while block_start < test_end:
        block = test[(test["bucket"] >= block_start) & (test["bucket"] < block_start + step)]
        if not block.empty:
            rows = training_rows(frame, block_start, horizon, train_window_bars)
            for variant, names in variant_features.items():
                model = train(rows, names, model_params)
                prob = predict_up(model, block)
                out.append(pd.DataFrame({
                    "variant": variant, "coin": block["coin"].to_numpy(),
                    "bucket": block["bucket"].to_numpy(), "prob_up": prob,
                    "action": to_action(prob, buy_threshold, sell_threshold),
                    "label": block["label"].to_numpy(),
                    "next_return": block["next_return"].to_numpy(),
                    "n_train": model.n_train,
                }))
        block_start += step
    if not out:
        return pd.DataFrame(columns=["variant", "coin", "bucket", "prob_up", "action", "label",
                                     "next_return", "n_train"])
    return pd.concat(out, ignore_index=True)


def strategy_returns(preds: pd.DataFrame, fee_bps: float) -> pd.DataFrame:
    """Per-bar returns of the long-only rule for ONE variant and ONE coin (sorted by bucket)."""
    p = preds.sort_values("bucket").copy()
    pos = pd.Series(np.where(p["action"] == BUY, 1.0, np.where(p["action"] == SELL, 0.0, np.nan)),
                    index=p.index)
    p["position"] = pos.ffill().fillna(0.0)               # HOLD keeps the previous position
    trades = p["position"].diff().abs().fillna(p["position"].abs())
    p["fee"] = trades * fee_bps / 10_000
    p["trade"] = trades > 0
    p["strategy_return"] = p["position"] * p["next_return"] - p["fee"]
    return p


def _max_drawdown(returns: pd.Series) -> float:
    equity = (1 + returns).cumprod()
    return float((equity / equity.cummax() - 1).min()) if len(equity) else 0.0


def _metrics(p: pd.DataFrame) -> dict:
    """Prediction quality + trading result for one (variant, coin) or the portfolio."""
    labelled = p[p["label"].notna()]
    y = labelled["label"].astype(int)
    up_pred = labelled["prob_up"] > 0.5
    buys, sells = labelled[labelled["action"] == BUY], labelled[labelled["action"] == SELL]
    auc = float(roc_auc_score(y, labelled["prob_up"])) if y.nunique() == 2 else float("nan")
    return {
        "bars": int(len(p)),
        "buy": int((p["action"] == BUY).sum()),
        "sell": int((p["action"] == SELL).sum()),
        "hold": int((p["action"] == "HOLD").sum()),
        "accuracy": float((up_pred == (y == 1)).mean()) if len(y) else float("nan"),
        "baseline_accuracy": float(max(y.mean(), 1 - y.mean())) if len(y) else float("nan"),
        "buy_hit_rate": float((buys["label"] == 1).mean()) if len(buys) else float("nan"),
        "sell_hit_rate": float((sells["label"] == 0).mean()) if len(sells) else float("nan"),
        "auc": auc,
        "strategy_return": float((1 + p["strategy_return"]).prod() - 1),
        "buy_hold_return": float((1 + p["next_return"]).prod() - 1),
        "trades": int(p["trade"].sum()),
        "time_in_market": float(p["position"].mean()),
        "max_drawdown": _max_drawdown(p["strategy_return"]),
        "buy_hold_max_drawdown": _max_drawdown(p["next_return"]),
    }


def evaluate(preds: pd.DataFrame, fee_bps: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(per-bar trading table, metrics table with one row per variant x coin + ALL)."""
    traded, rows = [], []
    for (variant, coin), g in preds.groupby(["variant", "coin"], sort=True):
        t = strategy_returns(g, fee_bps)
        traded.append(t)
        rows.append({"variant": variant, "coin": coin, **_metrics(t)})
    traded = pd.concat(traded, ignore_index=True) if traded else preds.iloc[0:0]

    # Equal-weight portfolio: each bar, average the coins' returns.
    for variant, g in traded.groupby("variant", sort=True):
        port = g.groupby("bucket").agg(
            strategy_return=("strategy_return", "mean"), next_return=("next_return", "mean"),
            position=("position", "mean"), trade=("trade", "sum")).reset_index()
        m = _metrics(g)                                     # prediction metrics over all coins
        pm = _metrics(port.assign(label=np.nan, prob_up=np.nan, action="HOLD"))
        for k in ("strategy_return", "buy_hold_return", "max_drawdown", "buy_hold_max_drawdown",
                  "time_in_market"):
            m[k] = pm[k]
        m["trades"] = int(g["trade"].sum())
        rows.append({"variant": variant, "coin": "ALL (equal weight)", **m})
    metrics = pd.DataFrame(rows)
    if not metrics.empty:
        metrics["_all"] = metrics["coin"].str.startswith("ALL")          # portfolio rows last
        metrics["_v"] = metrics["variant"].map({v: i for i, v in enumerate(VARIANTS)})
        metrics = metrics.sort_values(["_all", "coin", "_v"]).drop(columns=["_all", "_v"])
    return traded, metrics.reset_index(drop=True)