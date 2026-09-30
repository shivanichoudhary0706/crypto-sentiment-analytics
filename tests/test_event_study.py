"""
Week 7 tests — synthetic data with a KNOWN planted price reaction.
If the pipeline cannot recover a reaction we planted, its answer on real
news events is meaningless. Mirrors the synthetic-recovery pattern in
tests/test_lag_detection.py.

    python -m pytest tests/test_event_study.py -v
"""
import numpy as np
import pandas as pd
import pytest

from lag_detection.event_study import (
    baseline_car_samples, cluster_events, event_car, horizon_significance,
    onset_lag, per_event_optimal_exit,
)

TRUE_ONSET = 3       # planted: price starts moving 3 hours after a positive event
JUMP_SIZE = 0.05      # planted: +5% cumulative move once it kicks in


# --------------------------------------------------------------------
# cluster_events
# --------------------------------------------------------------------

def test_cluster_events_merges_close_articles():
    articles = pd.DataFrame({
        "time": pd.to_datetime([
            "2026-08-01 10:00", "2026-08-01 10:45",  # 45min apart -> same cluster
            "2026-08-02 09:00",                       # >1h later -> new cluster
        ], utc=True),
        "url": ["a", "b", "c"],
        "score": [0.7, 0.8, -0.9],
    })
    events = cluster_events(articles, cluster_window_hours=1.0)
    assert len(events) == 2
    assert events.iloc[0]["n_articles"] == 2
    assert events.iloc[0]["direction"] == "positive"
    assert events.iloc[1]["direction"] == "negative"


def test_cluster_events_empty_input():
    empty = pd.DataFrame(columns=["time", "url", "score"])
    events = cluster_events(empty, cluster_window_hours=1.0)
    assert events.empty
    assert list(events.columns) == ["event_time", "direction", "magnitude", "n_articles"]


# --------------------------------------------------------------------
# event_car
# --------------------------------------------------------------------

def _flat_then_jump_series(n=100, jump_at=50, jump_size=JUMP_SIZE, freq="1h"):
    """Price is flat, then jumps once at `jump_at` and stays there."""
    idx = pd.date_range("2026-08-01", periods=n, freq=freq, tz="UTC")
    prices = np.full(n, 100.0)
    prices[jump_at:] = 100.0 * (1 + jump_size)
    return pd.Series(prices, index=idx)


def test_event_car_recovers_planted_jump():
    close = _flat_then_jump_series()
    event_time = close.index[50 - TRUE_ONSET]  # event happens TRUE_ONSET hours before the jump
    car = event_car(close, event_time, horizon_hours=10)
    # before onset: ~0; at/after onset: ~ln(1+JUMP_SIZE)
    assert abs(car[TRUE_ONSET - 2]) < 1e-9
    assert abs(car[TRUE_ONSET] - np.log(1 + JUMP_SIZE)) < 1e-9


def test_event_car_nan_past_end_of_series():
    close = _flat_then_jump_series(n=10)
    event_time = close.index[8]
    car = event_car(close, event_time, horizon_hours=10)
    assert np.isnan(car[-1])  # horizon runs past available data


def test_event_car_nan_when_event_before_series_start():
    close = _flat_then_jump_series()
    car = event_car(close, close.index[0] - pd.Timedelta(hours=5), horizon_hours=5)
    assert np.all(np.isnan(car))


# --------------------------------------------------------------------
# horizon_significance + onset_lag: full recovery test
# --------------------------------------------------------------------

def test_onset_lag_recovers_planted_onset():
    """Plant many positive events, each followed by a real jump TRUE_ONSET
    hours later; noise events elsewhere. The pipeline should flag onset
    at (or within 1 hour of) TRUE_ONSET, not before it."""
    rng = np.random.default_rng(0)
    n = 24 * 40  # 40 days hourly
    idx = pd.date_range("2026-08-01", periods=n, freq="1h", tz="UTC")
    log_returns = rng.normal(0, 0.002, size=n)  # ambient noise

    event_positions = np.arange(20, n - 30, 30)  # one event every 30h
    for pos in event_positions:
        log_returns[pos + TRUE_ONSET] += np.log(1 + JUMP_SIZE)  # the planted reaction

    close = pd.Series(100.0 * np.exp(np.cumsum(log_returns)), index=idx)
    events = pd.DataFrame({
        "event_time": idx[event_positions],
        "direction": "positive",
        "magnitude": 0.8,
        "n_articles": 1,
    })

    horizon = 15
    event_cars = np.stack([event_car(close, t, horizon) for t in events["event_time"]])
    baseline = baseline_car_samples(close, horizon, n_samples=1000, rng=rng)
    # min_effect_car filters noise-level "significant" hours (tiny |mean_car|
    # can still get p<0.05 from bootstrap sampling variance with few events) --
    # this is the fix for a real off-by-one found while building this test:
    # without it, a fluke-significant hour right before the true onset merges
    # into one continuous run and reports onset a full hour early.
    sig_table = horizon_significance(event_cars, baseline, alpha=0.05, n_bootstrap=500, rng=rng,
                                     min_effect_car=0.01)

    onset = onset_lag(sig_table, min_run=2)
    assert onset is not None
    assert TRUE_ONSET <= onset <= TRUE_ONSET + 2  # allow 1-2h slack from bootstrap noise

    exits = per_event_optimal_exit(event_cars, ["positive"] * len(events))
    # every event's own best exit should be at/after the jump, i.e. not "sell before news works"
    assert (exits["optimal_exit_hour"] >= TRUE_ONSET).mean() > 0.8


def test_horizon_significance_no_effect_mostly_not_significant():
    """Events with NO planted reaction (pure noise) should mostly fail to
    reach significance -- guards against a test that always says 'yes'."""
    rng = np.random.default_rng(1)
    n = 24 * 40
    idx = pd.date_range("2026-08-01", periods=n, freq="1h", tz="UTC")
    log_returns = rng.normal(0, 0.002, size=n)
    close = pd.Series(100.0 * np.exp(np.cumsum(log_returns)), index=idx)

    event_positions = rng.integers(20, n - 30, size=15)
    horizon = 15
    event_cars = np.stack([event_car(close, idx[p], horizon) for p in event_positions])
    baseline = baseline_car_samples(close, horizon, n_samples=1000, rng=rng)
    sig_table = horizon_significance(event_cars, baseline, alpha=0.05, n_bootstrap=500, rng=rng)

    assert sig_table["significant_fdr"].fillna(False).mean() < 0.3  # mostly not significant


# --------------------------------------------------------------------
# per_event_optimal_exit
# --------------------------------------------------------------------

def test_per_event_optimal_exit_positive_is_argmax():
    car = np.array([0.01, 0.02, 0.05, 0.03, 0.01])
    df = per_event_optimal_exit(np.array([car]), ["positive"])
    assert df.iloc[0]["optimal_exit_hour"] == 3  # 1-indexed, argmax at index 2


def test_per_event_optimal_exit_negative_is_argmin():
    car = np.array([-0.01, -0.04, -0.02, -0.05, -0.03])
    df = per_event_optimal_exit(np.array([car]), ["negative"])
    assert df.iloc[0]["optimal_exit_hour"] == 4  # argmin at index 3


def test_per_event_optimal_exit_skips_all_nan():
    car = np.full(5, np.nan)
    df = per_event_optimal_exit(np.array([car]), ["positive"])
    assert df.empty