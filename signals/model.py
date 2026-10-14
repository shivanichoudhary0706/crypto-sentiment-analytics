"""
Model training, prediction and the probability -> BUY/HOLD/SELL rule. Pure, no DB.

Gradient-boosted trees (scikit-learn HistGradientBoostingClassifier):
  * handles missing values natively (a price gap or a coin with no data yet),
  * captures non-linear effects (e.g. news matters only when volatility is high),
  * trains in about a second on months of hourly bars, so the live job simply
    retrains on every pass -- no saved model files to version or go stale.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

BUY, HOLD, SELL = "BUY", "HOLD", "SELL"


@dataclass
class TrainedModel:
    estimator: HistGradientBoostingClassifier | None   # None -> constant model (one class only)
    feature_names: list[str]
    n_train: int
    base_rate: float           # share of "up" labels in the training data
    train_end: pd.Timestamp    # newest training row's bucket


def train(rows: pd.DataFrame, feature_names: list[str], params: dict) -> TrainedModel:
    y = rows["label"].astype(int).to_numpy()
    base_rate = float(y.mean()) if len(y) else 0.5
    train_end = rows["bucket"].max() if len(rows) else pd.NaT
    if len(np.unique(y)) < 2:
        # Degenerate history (all up or all down): fall back to the base rate.
        return TrainedModel(None, feature_names, len(y), base_rate, train_end)
    est = HistGradientBoostingClassifier(**params)
    est.fit(rows[feature_names].to_numpy(dtype=float), y)
    return TrainedModel(est, feature_names, len(y), base_rate, train_end)


def predict_up(model: TrainedModel, rows: pd.DataFrame) -> np.ndarray:
    """Probability that the price is higher `horizon_bars` from now."""
    if model.estimator is None:
        return np.full(len(rows), model.base_rate)
    proba = model.estimator.predict_proba(rows[model.feature_names].to_numpy(dtype=float))
    up_col = list(model.estimator.classes_).index(1)
    return proba[:, up_col]


def to_action(prob_up, buy_threshold: float, sell_threshold: float):
    """prob_up >= buy -> BUY; <= sell -> SELL; otherwise HOLD (not confident)."""
    p = np.asarray(prob_up, dtype=float)
    return np.where(p >= buy_threshold, BUY, np.where(p <= sell_threshold, SELL, HOLD))