"""
features_bar rows -> model matrix. Pure pandas, no DB. Unit-testable.

One row per (target coin, bar t). Everything in a row is known at the CLOSE of
bar t; the label looks forward:

    label_t = 1 if ln(close_{t+h} / close_t) > 0 else 0      (h = horizon_bars)

so a model trained on these rows answers "will this coin be higher h bars from
now?", which is exactly the question a live BUY / SELL decision needs.

Feature groups (all at bar t or earlier):
  own price   : log_return and its lags 1-3, realized_vol, ret_vol_short/long,
                volume relative to its 24-bar average
  input coins : e.g. BTC and ETH log_return (t and t-1) and realized_vol
  news        : every <stream>_news_score / _news_count / _news_roll column
  calendar    : hour of day (sin/cos)
  coin id     : one-hot, so one pooled model serves all target coins
"""
from __future__ import annotations

import numpy as np
import pandas as pd

OWN_LAGS = (1, 2, 3)
VOLUME_WINDOW = 24
NEWS_SUFFIXES = ("_news_score", "_news_count", "_news_roll")

# Columns the dataset needs from features_bar (news columns are discovered).
BASE_INPUT_COLUMNS = ("bucket", "coin", "close", "volume", "log_return", "realized_vol",
                      "ret_vol_short", "ret_vol_long", "is_complete")


def coin_key(coin: str) -> str:
    """'BTC/USDT' -> 'btc' (used as a column prefix)."""
    return coin.split("/")[0].lower()


def news_columns(columns) -> list[str]:
    return sorted(c for c in columns if c.endswith(NEWS_SUFFIXES))


def _own_features(g: pd.DataFrame) -> pd.DataFrame:
    """Per-coin features on that coin's regular bar grid (sorted by bucket)."""
    out = pd.DataFrame(index=g.index)
    out["ret"] = g["log_return"]
    for k in OWN_LAGS:
        out[f"ret_lag{k}"] = g["log_return"].shift(k)
    out["realized_vol"] = g["realized_vol"]
    out["ret_vol_short"] = g["ret_vol_short"]
    out["ret_vol_long"] = g["ret_vol_long"]
    avg_vol = g["volume"].rolling(VOLUME_WINDOW, min_periods=1).mean()
    out["volume_ratio"] = (g["volume"] / avg_vol).where(avg_vol > 0)
    return out


def _input_coin_features(features: pd.DataFrame, input_coins) -> pd.DataFrame:
    """Wide frame indexed by bucket: <key>_ret, <key>_ret_lag1, <key>_rv per input coin."""
    frames = []
    for coin in input_coins:
        g = features[features["coin"] == coin].sort_values("bucket").set_index("bucket")
        if g.empty:
            continue
        k = coin_key(coin)
        frames.append(pd.DataFrame({
            f"{k}_ret": g["log_return"],
            f"{k}_ret_lag1": g["log_return"].shift(1),
            f"{k}_rv": g["realized_vol"],
        }))
    return pd.concat(frames, axis=1) if frames else pd.DataFrame()


def build_model_frame(features: pd.DataFrame, targets, input_coins,
                      horizon_bars: int) -> tuple[pd.DataFrame, list[str]]:
    """features_bar rows (one freq) -> (frame, feature_names).

    The frame has columns: bucket, coin, close, is_complete, future_return, label,
    plus every feature in feature_names. label is NaN where the future is unknown
    (the newest bars, or a price gap) -- those rows can be predicted, not trained on.
    """
    features = features.copy()
    features["bucket"] = pd.to_datetime(features["bucket"], utc=True)
    news_cols = news_columns(features.columns)
    inputs = _input_coin_features(features, input_coins)
    coin_dummies = [f"coin_{coin_key(c)}" for c in sorted(targets)]

    parts = []
    for coin in targets:
        g = features[features["coin"] == coin].sort_values("bucket").reset_index(drop=True)
        if g.empty:
            continue
        own = _own_features(g)
        frame = pd.concat([g[["bucket", "coin", "close", "is_complete"]], own, g[news_cols]], axis=1)

        future = np.log(g["close"].shift(-horizon_bars) / g["close"])
        frame["future_return"] = future
        frame["label"] = (future > 0).astype(float).where(future.notna())

        if not inputs.empty:
            frame = frame.join(inputs, on="bucket")
        hour = frame["bucket"].dt.hour
        frame["hour_sin"] = np.sin(2 * np.pi * hour / 24)
        frame["hour_cos"] = np.cos(2 * np.pi * hour / 24)
        for d in coin_dummies:
            frame[d] = float(d == f"coin_{coin_key(coin)}")
        parts.append(frame)

    if not parts:
        return pd.DataFrame(), []
    frame = pd.concat(parts, ignore_index=True)

    input_cols = [c for c in inputs.columns] if not inputs.empty else []
    feature_names = (["ret", *[f"ret_lag{k}" for k in OWN_LAGS], "realized_vol", "ret_vol_short",
                      "ret_vol_long", "volume_ratio"]
                     + input_cols + news_cols + ["hour_sin", "hour_cos"] + coin_dummies)
    return frame, feature_names


def training_rows(frame: pd.DataFrame, predict_bucket, horizon: pd.Timedelta,
                  train_window_bars: int) -> pd.DataFrame:
    """Labelled rows a model may learn from when predicting at `predict_bucket`.

    Row t's label uses close_{t+h}. That price is only known once bar t+h has
    closed, so a model predicting at bar T may only learn from rows with
    t + h <= T. Anything later would let tomorrow's price leak into today's model.
    Keeps the most recent `train_window_bars` such rows per coin.
    """
    cutoff = pd.Timestamp(predict_bucket) - horizon
    rows = frame[(frame["bucket"] <= cutoff) & frame["label"].notna() & frame["close"].notna()]
    return (rows.sort_values("bucket")
                .groupby("coin", group_keys=False)
                .tail(train_window_bars))