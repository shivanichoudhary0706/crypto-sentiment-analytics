"""
Pure pandas transforms — no DB, no config files. Unit-testable.

Alignment convention (identical to lag_detection/series_builder.py; enforced by
tests/test_features.py, and to be matched by any future PySpark engine):
  * A bar labelled t covers [t, t + freq), buckets aligned to the Unix epoch (UTC).
  * close_t         = close of the last 1-minute candle inside bar t.
  * log_return_t    = ln(close_t / close_{t-1})     -> price move DURING bar t.
  * finbert_score_t = mean FinBERT compound of articles PUBLISHED during bar t,
                      0.0 when none was published (no news = neutral mood).
  * The time grid is REGULAR: a missing bar stays as a row with null price, so a
    shift of k rows always means k * freq of wall-clock time.
  * Rolling windows are row-based over that regular grid and include bar t itself.
    A signal for bar t+1 may therefore use features of bar t — never bar t+1.

Inputs
  candles: time (tz-aware UTC), symbol, open, high, low, close, volume
  scores:  time (tz-aware UTC = publish time), url, coin,
           finbert_compound, vader_compound, finbert_label
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from features.config import FREQ_TO_PANDAS, FREQ_TO_SECONDS

MINUTE = pd.Timedelta(minutes=1)

# Column order of the features_bar table. Shared by transform, DDL and upsert so
# they cannot drift apart silently. A future Spark engine must emit exactly this.
FEATURE_COLUMNS = (
    "bucket", "symbol", "coin", "freq",
    "open", "high", "low", "close", "volume", "n_candles", "is_complete",
    "log_return", "abs_return", "realized_vol",
    "n_articles", "n_positive", "n_negative",
    "finbert_mean", "vader_mean", "finbert_score", "vader_score",
    "finbert_roll_short", "finbert_roll_long", "finbert_zscore",
    "articles_roll_short", "articles_roll_long",
    "ret_vol_short", "ret_vol_long",
)

CANDLE_COLUMNS = ("time", "symbol", "open", "high", "low", "close", "volume")
SCORE_COLUMNS = ("time", "url", "coin", "finbert_compound", "vader_compound", "finbert_label")


def _empty_scores() -> pd.DataFrame:
    df = pd.DataFrame(columns=list(SCORE_COLUMNS))
    df["time"] = pd.to_datetime(df["time"], utc=True)
    return df


def with_minute_returns(candles: pd.DataFrame) -> pd.DataFrame:
    """Add r1m = 1-minute log return; null when the previous candle is not exactly
    one minute earlier (a gap must not be counted as a single 1-minute move)."""
    c = candles.sort_values(["symbol", "time"]).copy()
    g = c.groupby("symbol", sort=False)
    contiguous = (c["time"] - g["time"].shift()) == MINUTE
    r = np.log(c["close"] / g["close"].shift())
    c["r1m"] = r.where(contiguous)
    return c


def market_bars(candles: pd.DataFrame, freq: str) -> pd.DataFrame:
    """1-minute candles (with r1m) -> one OHLCV row per (symbol, bucket)."""
    c = candles.assign(bucket=candles["time"].dt.floor(FREQ_TO_PANDAS[freq]),
                       r1m_sq=candles["r1m"] ** 2)
    bars = c.groupby(["symbol", "bucket"], sort=True).agg(
        open=("open", "first"), high=("high", "max"), low=("low", "min"),
        close=("close", "last"), volume=("volume", "sum"),
        n_candles=("close", "size"),
        rv_sum=("r1m_sq", lambda s: s.sum(min_count=1)),
    )
    # Realized volatility of the bar: sqrt(sum of squared 1-minute log returns).
    bars["realized_vol"] = np.sqrt(bars.pop("rv_sum"))
    return bars.reset_index()


def dedup_articles(scores: pd.DataFrame) -> pd.DataFrame:
    """The same story via RSS *and* CryptoCompare -> keep the earliest copy per
    coin (mirrors DISTINCT ON (url) in series_builder)."""
    return (scores.sort_values("time", kind="stable")
                  .drop_duplicates(["coin", "url"], keep="first"))


def sentiment_bars(scores: pd.DataFrame, freq: str) -> pd.DataFrame:
    """Deduplicated article scores -> one row per (symbol, bucket)."""
    cols = ["symbol", "bucket", "n_articles", "n_positive", "n_negative", "finbert_mean", "vader_mean"]
    if scores.empty:
        return pd.DataFrame(columns=cols)
    s = dedup_articles(scores)
    s = s.assign(
        symbol=s["coin"].str.replace("/", "", regex=False).str.upper(),
        bucket=s["time"].dt.floor(FREQ_TO_PANDAS[freq]),
        is_pos=(s["finbert_label"] == "positive").astype(int),
        is_neg=(s["finbert_label"] == "negative").astype(int),
        finbert_compound=s["finbert_compound"].astype(float),
        vader_compound=s["vader_compound"].astype(float),
    )
    out = s.groupby(["symbol", "bucket"]).agg(
        n_articles=("url", "size"), n_positive=("is_pos", "sum"), n_negative=("is_neg", "sum"),
        finbert_mean=("finbert_compound", "mean"), vader_mean=("vader_compound", "mean"),
    )
    return out.reset_index()[cols]


def assemble_symbol(mbars: pd.DataFrame, sbars: pd.DataFrame, freq: str,
                    rolling_short: int, rolling_long: int) -> pd.DataFrame:
    """Price + sentiment bars for ONE symbol -> regular grid with derived features."""
    rule = FREQ_TO_PANDAS[freq]
    expected = FREQ_TO_SECONDS[freq] // 60

    m = mbars.set_index("bucket").drop(columns="symbol")
    grid = pd.date_range(m.index.min(), m.index.max(), freq=rule, name="bucket")
    df = m.reindex(grid)
    df["n_candles"] = df["n_candles"].fillna(0).astype(int)

    # The last bar may still be forming — drop it if incomplete (a later run writes
    # it). Earlier partial bars are KEPT and flagged through is_complete.
    if df["n_candles"].iloc[-1] < expected:
        df = df.iloc[:-1]
    if df.empty:
        return df

    s = sbars.set_index("bucket").drop(columns="symbol").reindex(df.index)
    for col in ("n_articles", "n_positive", "n_negative"):
            df[col] = pd.to_numeric(s[col]).fillna(0).astype(int)
    df["finbert_mean"] = s["finbert_mean"].astype(float)
    df["vader_mean"] = s["vader_mean"].astype(float)

    df["is_complete"] = df["n_candles"] == expected
    df["log_return"] = np.log(df["close"]).diff()
    df["abs_return"] = df["log_return"].abs()
    df["finbert_score"] = df["finbert_mean"].fillna(0.0)
    df["vader_score"] = df["vader_mean"].fillna(0.0)

    short = dict(window=rolling_short, min_periods=1)
    long = dict(window=rolling_long, min_periods=1)
    df["finbert_roll_short"] = df["finbert_score"].rolling(**short).mean()
    df["finbert_roll_long"] = df["finbert_score"].rolling(**long).mean()
    std_long = df["finbert_score"].rolling(**long).std()
    df["finbert_zscore"] = ((df["finbert_score"] - df["finbert_roll_long"]) / std_long).where(std_long > 0)
    df["articles_roll_short"] = df["n_articles"].rolling(**short).sum().astype(int)
    df["articles_roll_long"] = df["n_articles"].rolling(**long).sum().astype(int)
    df["ret_vol_short"] = df["log_return"].rolling(**short).std()
    df["ret_vol_long"] = df["log_return"].rolling(**long).std()
    return df.reset_index()


def build_features(candles: pd.DataFrame, scores: pd.DataFrame | None,
                   frequencies: tuple[str, ...], rolling_short: int, rolling_long: int,
                   symbol_to_coin: dict[str, str]) -> pd.DataFrame:
    """Full transform for all symbols and frequencies -> rows in FEATURE_COLUMNS order."""
    if scores is None or scores.empty:
        scores = _empty_scores()
    if candles.empty:
        return pd.DataFrame(columns=list(FEATURE_COLUMNS))

    candles = with_minute_returns(candles)
    frames = []
    for freq in frequencies:
        mb = market_bars(candles, freq)
        sb = sentiment_bars(scores, freq)
        for symbol, m_sym in mb.groupby("symbol", sort=True):
            out = assemble_symbol(m_sym, sb[sb["symbol"] == symbol], freq, rolling_short, rolling_long)
            if out.empty:
                continue
            out["symbol"] = symbol
            out["coin"] = symbol_to_coin.get(symbol, symbol)
            out["freq"] = freq
            frames.append(out[list(FEATURE_COLUMNS)])
    if not frames:
        return pd.DataFrame(columns=list(FEATURE_COLUMNS))
    return pd.concat(frames, ignore_index=True)