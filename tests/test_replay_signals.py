"""Tests for signals.replay_signals -- the look-ahead guard and skip logic.

generate_signals is replaced by a fake that records what it was given, so these
tests check the replay loop itself, not the model.
"""
from types import SimpleNamespace

import pandas as pd

from signals import replay_signals as rs

CFG = SimpleNamespace(bar=pd.Timedelta(hours=1), train_window_bars=10, horizon_bars=1,
                      targets=("LTC/USDT", "LINK/USDT"))
BUCKETS = pd.date_range("2026-10-20", periods=60, freq="1h", tz="UTC")


def _features():
    rows = [{"bucket": b, "coin": c, "close": float(i)}
            for i, b in enumerate(BUCKETS) for c in CFG.targets]
    return pd.DataFrame(rows)


def _fake_generate(seen):
    def fake(hist, cfg):
        t = hist["bucket"].max()
        seen.append(t)
        return pd.DataFrame({
            "bucket": [t, t], "symbol": ["LTCUSDT", "LINKUSDT"],
            "prob_up": [0.6, 0.4], "action": ["BUY", "SELL"], "model_version": ["hgb", "hgb"],
        }), {"skipped": {}}
    return fake


def test_model_never_sees_rows_after_the_bar(monkeypatch):
    seen = []
    monkeypatch.setattr(rs, "generate_signals", _fake_generate(seen))
    targets = list(BUCKETS[40:45])
    rs.replay(_features(), CFG, targets, skip=set())
    assert seen == targets  # newest row passed in is exactly the bar being scored


def test_replayed_rows_are_tagged(monkeypatch):
    monkeypatch.setattr(rs, "generate_signals", _fake_generate([]))
    out = rs.replay(_features(), CFG, list(BUCKETS[40:43]), skip=set())
    assert len(out) == 6
    assert out["model_version"].str.startswith(rs.REPLAY_PREFIX).all()


def test_existing_signals_are_not_rewritten(monkeypatch):
    monkeypatch.setattr(rs, "generate_signals", _fake_generate([]))
    t = BUCKETS[40]
    out = rs.replay(_features(), CFG, [t], skip={("LTCUSDT", t)})
    assert list(out["symbol"]) == ["LINKUSDT"]


def test_fallback_to_older_bar_is_dropped(monkeypatch):
    def stale(hist, cfg):
        t = hist["bucket"].max() - pd.Timedelta(hours=1)  # coin's latest bar is older than t
        return pd.DataFrame({"bucket": [t], "symbol": ["LTCUSDT"], "prob_up": [0.5],
                             "action": ["HOLD"], "model_version": ["hgb"]}), {"skipped": {}}
    monkeypatch.setattr(rs, "generate_signals", stale)
    assert rs.replay(_features(), CFG, [BUCKETS[40]], skip=set()).empty