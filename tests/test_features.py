"""
Tests for the feature layer (features/).

The central test is PARITY: on the same synthetic data, features_bar must
reproduce lag_detection/series_builder.align_series() — close, log_return,
zero-filled sentiment and article counts. That is the evidence that the table
signals/backtesting read from computes exactly what the Week 6 statistics were
validated on. A future PySpark engine must pass the same test.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest

from features import transforms as tf
from features.config import parse_feature_config
from features.run_feature_job import plan_run, parse_args
from lag_detection.series_builder import align_series

UTC = timezone.utc
T0 = datetime(2026, 9, 1, 0, 0, tzinfo=UTC)
SYMBOL_TO_COIN = {"BTCUSDT": "BTC/USDT", "ETHUSDT": "ETH/USDT"}


# ---------------------------------------------------------------- synthetic data
def make_candles(symbol: str, base: float, hours: int, seed: int,
                 gap_hours: tuple[int, ...] = (), partial_hour: int | None = None,
                 trailing_minutes: int = 0) -> pd.DataFrame:
    """1-minute candles. `gap_hours` are fully missing, `partial_hour` keeps only
    its first 30 candles, `trailing_minutes` adds an incomplete final bar."""
    rng = np.random.default_rng(seed)
    rows, price = [], base
    for m in range(hours * 60 + trailing_minutes):
        h = m // 60
        price *= float(np.exp(rng.normal(0, 0.001)))
        if h in gap_hours or (partial_hour is not None and h == partial_hour and m % 60 >= 30):
            continue
        rows.append((T0 + timedelta(minutes=m), symbol, price, price * 1.001, price * 0.999,
                     price, float(rng.uniform(1, 5))))
    df = pd.DataFrame(rows, columns=list(tf.CANDLE_COLUMNS))
    df["time"] = pd.to_datetime(df["time"], utc=True)
    return df


def make_scores(coin: str, hours: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(hours * 2):
        if rng.random() < 0.4:
            t = T0 + timedelta(minutes=30 * i + int(rng.integers(0, 29)))
            fb = float(rng.uniform(-1, 1))
            label = "positive" if fb > 0.3 else ("negative" if fb < -0.3 else "neutral")
            rows.append((t, f"https://news/{coin}/{i}", coin, fb, float(rng.uniform(-1, 1)), label))
    # The same story arriving again later from another source -> must be ignored.
    first = rows[0]
    rows.append((first[0] + timedelta(hours=3), first[1], coin, 0.99, 0.99, "positive"))
    return scores_df(rows)


def scores_df(rows: list[tuple]) -> pd.DataFrame:
    df = pd.DataFrame(rows, columns=list(tf.SCORE_COLUMNS))
    df["time"] = pd.to_datetime(df["time"], utc=True)
    return df


def build(candles, scores, freq="1h", short=3, long=6) -> pd.DataFrame:
    return tf.build_features(candles, scores, (freq,), short, long, SYMBOL_TO_COIN)


def one(df: pd.DataFrame, symbol="BTCUSDT") -> pd.DataFrame:
    return df[df.symbol == symbol].set_index("bucket").sort_index()


# ---------------------------------------------------------------- pandas reference
def series_builder_reference(candles, scores, symbol, coin, freq) -> pd.DataFrame:
    """What series_builder's SQL returns, fed through its align_series()."""
    c = candles[candles.symbol == symbol].set_index("time").sort_index()
    market = c.groupby(c.index.floor(freq)).agg(close=("close", "last"), n_candles=("close", "size"))
    market.index.name = "bucket"
    s = scores[scores.coin == coin].sort_values("time").drop_duplicates("url", keep="first")
    s = s.set_index("time")
    sent = s.groupby(s.index.floor(freq)).agg(
        sent_mean=("finbert_compound", "mean"), sent_sum=("finbert_compound", "sum"),
        n_articles=("finbert_compound", "size"))
    sent.index.name = "bucket"
    df, _ = align_series(market, sent, freq, fill_method="zero")
    return df


# ---------------------------------------------------------------- transform tests
@pytest.mark.parametrize("freq", ["1h", "4h"])
def test_parity_with_series_builder(freq):
    hours = 48
    candles = pd.concat([
        make_candles("BTCUSDT", 60000, hours, 1, gap_hours=(10, 11), partial_hour=20, trailing_minutes=25),
        make_candles("ETHUSDT", 2500, hours, 2, gap_hours=(30,)),
    ], ignore_index=True)
    scores = pd.concat([make_scores("BTC/USDT", hours, 3), make_scores("ETH/USDT", hours, 4)],
                       ignore_index=True)
    got = build(candles, scores, freq)

    for symbol, coin in SYMBOL_TO_COIN.items():
        ref = series_builder_reference(candles, scores, symbol, coin, freq)
        mine = one(got, symbol).loc[ref.index.min():]   # series_builder drops the first bar
        assert list(mine.index) == list(ref.index), f"{symbol} {freq}: bar grids differ"
        np.testing.assert_allclose(mine["close"], ref["close"], rtol=1e-12, equal_nan=True)
        np.testing.assert_allclose(mine["log_return"], ref["log_return"], atol=1e-12, equal_nan=True)
        np.testing.assert_allclose(mine["finbert_score"], ref["sentiment"], atol=1e-12)
        np.testing.assert_array_equal(mine["n_articles"].to_numpy(), ref["n_articles"].to_numpy())
        assert (mine["coin"] == coin).all() and (mine["freq"] == freq).all()


def test_incomplete_last_bar_dropped_but_inner_partial_kept():
    got = one(build(make_candles("BTCUSDT", 60000, 6, 5, partial_hour=2, trailing_minutes=20), None))
    assert got.index.max() == pd.Timestamp(T0 + timedelta(hours=5))   # 20-minute hour 6 dropped
    partial = got.loc[pd.Timestamp(T0 + timedelta(hours=2))]
    assert partial["n_candles"] == 30 and not partial["is_complete"]
    assert got.drop(index=pd.Timestamp(T0 + timedelta(hours=2)))["is_complete"].all()


def test_gap_bars_are_rows_with_null_price():
    got = one(build(make_candles("BTCUSDT", 60000, 6, 6, gap_hours=(3,)), None))
    gap = pd.Timestamp(T0 + timedelta(hours=3))
    assert len(got) == 6                                   # regular grid, nothing glued together
    assert got.loc[gap, "n_candles"] == 0 and pd.isna(got.loc[gap, "close"])
    assert pd.isna(got.loc[gap + pd.Timedelta(hours=1), "log_return"])   # unknown across a gap
    # The first candle after the gap must not create a fake 1-minute return.
    assert got["realized_vol"].notna().sum() == 5


def test_duplicate_url_counted_once_keeping_earliest():
    t = T0 + timedelta(minutes=10)
    scores = scores_df([(t, "u1", "BTC/USDT", 0.5, 0.1, "positive"),
                        (t + timedelta(hours=1), "u1", "BTC/USDT", -0.9, -0.9, "negative")])
    got = one(build(make_candles("BTCUSDT", 60000, 3, 7), scores))
    assert got["n_articles"].sum() == 1
    assert got.loc[pd.Timestamp(T0), "finbert_score"] == pytest.approx(0.5)
    assert got.loc[pd.Timestamp(T0), "n_positive"] == 1


def test_no_news_means_zero_score_but_null_mean():
    got = one(build(make_candles("BTCUSDT", 60000, 3, 8), None))
    assert (got["finbert_score"] == 0).all() and got["finbert_mean"].isna().all()
    assert got["finbert_zscore"].isna().all()               # zero variance -> undefined, not inf


def test_rolling_features_use_only_past_and_current_bars():
    scores = scores_df([(T0 + timedelta(hours=h, minutes=5), f"u{h}", "BTC/USDT", v, 0.0, "neutral")
                        for h, v in [(0, 0.3), (1, 0.6), (2, 0.9)]])
    got = one(build(make_candles("BTCUSDT", 60000, 8, 9), scores, short=3, long=6))
    assert got.iloc[2]["finbert_roll_short"] == pytest.approx(0.6)        # mean(0.3, 0.6, 0.9)
    assert got.iloc[2]["articles_roll_short"] == 3
    assert got.iloc[1]["finbert_roll_short"] == pytest.approx(0.45)       # bar 2's news not leaked into bar 1
    assert got.iloc[5]["finbert_roll_short"] == pytest.approx(0.0)
    assert got.iloc[3]["finbert_roll_long"] == pytest.approx(1.8 / 4)     # partial window at the start


def test_realized_vol_matches_manual_calculation():
    candles = make_candles("BTCUSDT", 60000, 2, 10)
    got = one(build(candles, None))
    c = candles.set_index("time")["close"]
    r = np.log(c).diff()
    hour1 = r[(r.index >= T0 + timedelta(hours=1)) & (r.index < T0 + timedelta(hours=2))]
    assert got.iloc[1]["realized_vol"] == pytest.approx(np.sqrt((hour1 ** 2).sum()))


def test_output_columns_and_empty_inputs():
    out = tf.build_features(make_candles("BTCUSDT", 60000, 5, 11), None, ("1h", "4h"), 2, 4, SYMBOL_TO_COIN)
    assert tuple(out.columns) == tf.FEATURE_COLUMNS
    # 5 hours of data: five complete 1h bars; one complete 4h bar (04:00-08:00 is still forming).
    assert (out["freq"] == "1h").sum() == 5 and (out["freq"] == "4h").sum() == 1
    empty = tf.build_features(make_candles("BTCUSDT", 1, 0, 0), None, ("1h",), 2, 4, SYMBOL_TO_COIN)
    assert empty.empty and tuple(empty.columns) == tf.FEATURE_COLUMNS


# ---------------------------------------------------------------- config + run plan
_GOOD = {"frequencies": ["1h", "4h"], "rolling_short_bars": 6, "rolling_long_bars": 24,
         "incremental_lookback_bars": 6, "table": "features_bar",
         "news_streams": {"btc": "BTC/USDT", "mkt": "MARKET"}}
_COINS = [{"symbol": "BTC/USDT"}, {"symbol": "ETH/USDT"}]


def test_config_parses():
    cfg = parse_feature_config(_GOOD, _COINS)
    assert cfg.symbol_to_coin == SYMBOL_TO_COIN
    assert cfg.history_needed() == timedelta(hours=4 * 26)


@pytest.mark.parametrize("patch", [{"frequencies": ["2h"]}, {"rolling_short_bars": 24},
                                   {"incremental_lookback_bars": 0},
                                   {"table": "features_bar; DROP TABLE x"}])
def test_config_rejects_bad_values(patch):
    with pytest.raises(ValueError):
        parse_feature_config({**_GOOD, **patch}, _COINS)


def test_config_rejects_missing_key():
    with pytest.raises(KeyError):
        parse_feature_config({k: v for k, v in _GOOD.items() if k != "table"}, _COINS)


def test_plan_full_when_table_empty_or_forced():
    cfg = parse_feature_config(_GOOD, _COINS)
    assert plan_run(cfg, {}, False).mode == "full"
    assert plan_run(cfg, {"1h": T0}, False).mode == "full"            # a frequency was never built
    wm = {"1h": T0, "4h": T0}
    assert plan_run(cfg, wm, True) == plan_run(cfg, {}, False)


def test_plan_incremental_rewinds_lookback_and_reads_history():
    cfg = parse_feature_config(_GOOD, _COINS)
    plan = plan_run(cfg, {"1h": T0, "4h": T0}, False)
    assert plan.mode == "incremental"
    assert plan.write_from == T0 - timedelta(hours=24)                # 6 bars of the 4h frequency
    assert plan.read_from == plan.write_from - cfg.history_needed()
    assert plan.read_end is None


def test_plan_pinned_window():
    cfg = parse_feature_config(_GOOD, _COINS)
    end = T0 + timedelta(days=10)
    plan = plan_run(cfg, {"1h": T0, "4h": T0}, False, start=T0, end=end)
    assert (plan.mode, plan.write_from, plan.read_end) == ("pinned", T0, end)
    assert plan.read_from == T0 - cfg.history_needed()


def test_cli_validation():
    assert parse_args(["--every", "300"]).every == 300
    with pytest.raises(SystemExit):
        parse_args(["--every", "5"])
    with pytest.raises(SystemExit):
        parse_args(["--every", "300", "--start", "2026-06-01"])



# ---------------------------------------------------------------- news streams
STREAMS = {"btc": "BTC/USDT", "mkt": "MARKET"}


def test_news_streams_reach_price_only_coins():
    """SOL has no news of its own, but its rows must carry BTC and MARKET news."""
    candles = pd.concat([make_candles("BTCUSDT", 60000, 4, 20), make_candles("SOLUSDT", 150, 4, 21)],
                        ignore_index=True)
    scores = scores_df([
        (T0 + timedelta(minutes=5), "b1", "BTC/USDT", 0.8, 0.5, "positive"),
        (T0 + timedelta(minutes=40), "b2", "BTC/USDT", 0.4, 0.1, "positive"),
        (T0 + timedelta(hours=2, minutes=1), "m1", "MARKET", -0.6, -0.2, "negative"),
    ])
    s2c = {**SYMBOL_TO_COIN, "SOLUSDT": "SOL/USDT"}
    out = tf.build_features(candles, scores, ("1h",), 3, 6, s2c, STREAMS)
    assert tuple(out.columns) == tf.feature_columns(("btc", "mkt"))
    sol = one(out, "SOLUSDT")
    assert (sol["n_articles"] == 0).all() and (sol["finbert_score"] == 0).all()   # no own news
    assert sol.iloc[0]["btc_news_score"] == pytest.approx(0.6)                    # mean(0.8, 0.4)
    assert sol.iloc[0]["btc_news_count"] == 2
    assert sol.iloc[2]["mkt_news_score"] == pytest.approx(-0.6)
    assert sol.iloc[1]["mkt_news_count"] == 0 and sol.iloc[1]["mkt_news_score"] == 0.0
    assert sol.iloc[1]["btc_news_roll"] == pytest.approx(0.3)                     # mean(0.6, 0.0)
    # The same stream values appear on the BTC rows (stream = shared market context).
    btc = one(out, "BTCUSDT")
    np.testing.assert_allclose(btc["btc_news_score"], sol["btc_news_score"])
    np.testing.assert_allclose(btc["btc_news_score"], btc["finbert_score"])        # BTC's own news


def test_no_streams_means_base_columns_only():
    out = tf.build_features(make_candles("BTCUSDT", 60000, 3, 22), None, ("1h",), 2, 4, SYMBOL_TO_COIN)
    assert tuple(out.columns) == tf.FEATURE_COLUMNS        