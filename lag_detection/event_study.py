"""
Week 7 — event-study module.

Week 6 asks: does the hourly SENTIMENT SERIES help predict the RETURN
SERIES, on average, across the whole window? (continuous, CCF/Granger)

This module asks a different question: does ONE strongly positive/negative
news EVENT produce a measurable price reaction, and if so, how many hours
until the reaction shows up (onset) and how many hours until it is spent
(the effective hold/exit window)? A single article gets averaged away in
an hourly mean; isolating it here is the point.

It also answers a second question for free, because the machinery is the
same either way: does a cause_coin's news move a DIFFERENT coin's price?
    cause_coin == price_symbol  -> "own-coin" reaction (BTC news -> BTC price)
    cause_coin != price_symbol  -> "spillover" reaction (BTC news -> LTC price)
This is how the small-cap coins enter the thesis: they need no news
coverage of their own, only clean Binance price history, because the
CAUSE is always a big-coin (or later, general-market) sentiment event.

Method (standard finance "event study" design, adapted for 24/7 crypto):
  1. Pull strongly-scored articles for cause_coin (|score| >= threshold).
  2. Cluster same-story duplicates published close together into one
     discrete EVENT (event_time, direction, magnitude, n_articles).
  3. For each event, compute the cumulative log return of price_symbol at
     every hour 1..H after the event, against the price level AT the
     event -> the event's CAR (cumulative abnormal return) curve.
  4. Average CAR separately for positive- and negative-direction events.
  5. At every horizon h, ask: is the mean event CAR unusual compared to
     the mean CAR of a same-sized random sample of ordinary (non-event)
     times? Answered by bootstrap -> one p-value per horizon, FDR-corrected
     across all horizons (reuses the same statsmodels FDR machinery as
     Week 6's Granger scan, so there is one place, not two, to fix a bug).
  6. onset_hour = first horizon where FDR-significant and stays so for
     `onset_min_run` consecutive hours (avoids calling one noisy hour the
     answer). Per-event optimal exit = argmax (positive) / argmin
     (negative) of THAT event's own realised curve; report the
     DISTRIBUTION across events (median, IQR), not just one number,
     because a single average hides how much events vary.

DOCUMENTED LIMITATION (state this in the thesis methodology chapter):
per-event optimal exit as built here is the realised, hindsight-optimal
argmax/argmin — useful for characterising the distribution of effective
timeframes, but not itself a real-time rule. A sharper version for
negative events ("hour by which X% of the eventual decline has already
happened") is noted as a future refinement, not built in this pass.
Whatever hold-duration rule is derived from this distribution must be
fit on freeze v1 (in-sample) and checked against freeze v2 (out-of-
sample) before it goes anywhere near the signal module — otherwise it is
just curve-fit to six-to-twelve weeks of data.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from psycopg2 import sql
from statsmodels.stats.multitest import multipletests

from lag_detection.config import ALLOWED_SENTIMENT_COLUMNS
from lag_detection.series_builder import fetch_market_bars, resolve_window

logger = logging.getLogger(__name__)

# DISTINCT ON (url): mirrors series_builder._SENTIMENT_SQL — the same story
# can arrive via RSS *and* CryptoCompare backfill; keep the earliest timestamp
# so a duplicate cannot count as two articles / shift the event time later.
_STRONG_ARTICLES_SQL = sql.SQL("""
WITH dedup AS (
    SELECT DISTINCT ON (url) time, url, {col}::double precision AS score
    FROM sentiment_scores
    WHERE coin = %(coin)s
      AND time >= %(start)s
      AND time <  %(end)s
      AND {col} IS NOT NULL
      AND abs({col}) >= %(threshold)s
    ORDER BY url, time
)
SELECT time, url, score FROM dedup ORDER BY time;
""")


# --------------------------------------------------------------------------
# DB-touching helpers (thin; the logic they call is pure and unit-tested)
# --------------------------------------------------------------------------

def fetch_strong_articles(conn, coin: str, start: datetime, end: datetime,
                          column: str, threshold: float) -> pd.DataFrame:
    """Articles for `coin` with |score| >= threshold, deduped by url."""
    if column not in ALLOWED_SENTIMENT_COLUMNS:
        raise ValueError(f"Unsupported sentiment column: {column}")
    query = _STRONG_ARTICLES_SQL.format(col=sql.Identifier(column))
    params = {"coin": coin, "start": start, "end": end, "threshold": threshold}
    with conn.cursor() as cur:
        cur.execute(query, params)
        rows = cur.fetchall()
    df = pd.DataFrame(rows, columns=["time", "url", "score"])
    if df.empty:
        return df
    df["time"] = pd.to_datetime(df["time"], utc=True)
    df["score"] = df["score"].astype(float)
    return df.sort_values("time").reset_index(drop=True)


def fetch_hourly_close(conn, price_symbol: str, start: datetime, end: datetime) -> pd.Series:
    """Hourly close price for price_symbol (may differ from the sentiment's
    cause_coin — this is what makes spillover analysis possible). Reuses
    the exact same market_data aggregation Week 6 uses (fetch_market_bars),
    just called with a different coin."""
    bars = fetch_market_bars(conn, price_symbol, "1h", start, end)
    return bars["close"]


# --------------------------------------------------------------------------
# Pure functions — no DB, unit-testable in isolation
# --------------------------------------------------------------------------

def cluster_events(articles: pd.DataFrame, cluster_window_hours: float) -> pd.DataFrame:
    """Group same-story duplicates published within `cluster_window_hours`
    of the PREVIOUS article into one event (rolling merge: a steady drip of
    coverage on one story chains into a single event; this is intentional —
    it is one news event even if five outlets cover it over several hours).

    articles: columns [time, url, score], already sorted by time.
    Returns one row per event: [event_time, direction, magnitude, n_articles].
        event_time = time of the FIRST article in the cluster (news breaks then)
        direction  = 'positive' / 'negative', sign of the cluster's mean score
        magnitude  = mean |score| in the cluster
    """
    cols = ["event_time", "direction", "magnitude", "n_articles"]
    if articles.empty:
        return pd.DataFrame(columns=cols)

    window = pd.Timedelta(hours=cluster_window_hours)
    events: list[dict] = []
    cluster_start = articles.iloc[0]["time"]
    cluster_scores = [articles.iloc[0]["score"]]

    for i in range(1, len(articles)):
        t, score = articles.iloc[i]["time"], articles.iloc[i]["score"]
        if t - articles.iloc[i - 1]["time"] <= window:
            cluster_scores.append(score)
        else:
            events.append(_close_cluster(cluster_start, cluster_scores))
            cluster_start, cluster_scores = t, [score]
    events.append(_close_cluster(cluster_start, cluster_scores))
    return pd.DataFrame(events, columns=cols)


def _close_cluster(start_time, scores: list[float]) -> dict:
    mean_score = float(np.mean(scores))
    return {
        "event_time": start_time,
        "direction": "positive" if mean_score > 0 else "negative",
        "magnitude": float(np.mean(np.abs(scores))),
        "n_articles": len(scores),
    }


def event_car(close: pd.Series, event_time: pd.Timestamp, horizon_hours: int) -> np.ndarray:
    """Cumulative log return at each hour 1..horizon_hours after event_time,
    against the close price at (or just before) event_time. NaN where price
    data is missing — a gap, or the event is too close to the end of the
    available window for the full horizon."""
    out = np.full(horizon_hours, np.nan)
    pos = close.index.searchsorted(event_time, side="right") - 1
    if pos < 0:
        return out
    base_price = close.iloc[pos]
    if pd.isna(base_price) or base_price <= 0:
        return out
    for h in range(1, horizon_hours + 1):
        j = pos + h
        if j >= len(close):
            break
        price_h = close.iloc[j]
        if pd.notna(price_h) and price_h > 0:
            out[h - 1] = float(np.log(price_h / base_price))
    return out


def baseline_car_samples(close: pd.Series, horizon_hours: int, n_samples: int,
                         rng: np.random.Generator) -> np.ndarray:
    """n_samples x horizon_hours matrix of CAR from RANDOM (non-event) start
    bars — the 'nothing happened' null distribution each horizon is tested
    against."""
    n = len(close)
    if n <= horizon_hours + 1:
        return np.empty((0, horizon_hours))
    starts = rng.integers(0, n - horizon_hours - 1, size=n_samples)
    out = np.full((n_samples, horizon_hours), np.nan)
    for i, pos in enumerate(starts):
        base_price = close.iloc[pos]
        if pd.isna(base_price) or base_price <= 0:
            continue
        for h in range(1, horizon_hours + 1):
            price_h = close.iloc[pos + h]
            if pd.notna(price_h) and price_h > 0:
                out[i, h - 1] = float(np.log(price_h / base_price))
    return out


def horizon_significance(event_cars: np.ndarray, baseline: np.ndarray, alpha: float,
                         n_bootstrap: int = 2000,
                         rng: np.random.Generator | None = None,
                         min_effect_car: float = 0.0) -> pd.DataFrame:
    """For each horizon h: is the MEAN event CAR unusual vs. the mean CAR of
    a same-sized random sample of ordinary times? Bootstrap the baseline
    mean (draw len(events) baseline rows with replacement, repeat
    n_bootstrap times) and locate the real mean in that distribution ->
    empirical two-sided p-value. FDR-correct across all horizons tested
    together (same statsmodels call Week 6's Granger scan uses).

    `significant_fdr` is the pure statistical flag (p_fdr < alpha). It is
    kept in the output for transparency, but `significant` additionally
    requires |mean_car| >= min_effect_car -- STATISTICAL significance can
    trip on a noise-level move when n_events is small (a few dozen events
    is not a lot of statistical power), so `significant` (economic +
    statistical) is the flag onset_lag should be driven from. Set
    min_effect_car=0.0 to disable this filter and get pure p<alpha
    behaviour.
    """
    rng = rng or np.random.default_rng(7)
    horizon_hours = event_cars.shape[1]
    records = []
    for h in range(horizon_hours):
        obs = event_cars[:, h]
        obs = obs[~np.isnan(obs)]
        base = baseline[:, h]
        base = base[~np.isnan(base)]
        if len(obs) == 0 or len(base) < 10:
            records.append({"horizon_h": h + 1, "n_events": int(len(obs)),
                            "mean_car": np.nan, "p_value": np.nan})
            continue
        mean_car = float(np.mean(obs))
        boot_means = np.fromiter(
            (np.mean(rng.choice(base, size=len(obs), replace=True)) for _ in range(n_bootstrap)),
            dtype=float, count=n_bootstrap,
        )
        p = float(min(1.0, 2 * min((boot_means >= mean_car).mean(), (boot_means <= mean_car).mean())))
        records.append({"horizon_h": h + 1, "n_events": int(len(obs)),
                        "mean_car": mean_car, "p_value": p})

    table = pd.DataFrame(records)
    valid = table["p_value"].notna()
    reject = np.zeros(len(table), dtype=bool)
    p_fdr = np.full(len(table), np.nan)
    if valid.sum() > 0:
        rej, pf, *_ = multipletests(table.loc[valid, "p_value"], alpha=alpha, method="fdr_bh")
        reject[valid.to_numpy()] = rej
        p_fdr[valid.to_numpy()] = pf
    table["p_fdr"] = p_fdr
    table["significant_fdr"] = reject
    table["significant"] = reject & (table["mean_car"].abs() >= min_effect_car)
    return table


def onset_lag(sig_table: pd.DataFrame, min_run: int = 2) -> int | None:
    """First horizon where `significant` (statistical AND economic, see
    horizon_significance) is True and STAYS True for at least `min_run`
    consecutive horizons — guards against calling one noisy hour the
    answer. None if no such run exists."""
    flags = sig_table["significant"].fillna(False).to_numpy()
    run = 0
    for i, f in enumerate(flags):
        run = run + 1 if f else 0
        if run >= min_run:
            return int(sig_table["horizon_h"].iloc[i - min_run + 1])
    return None


def per_event_optimal_exit(event_cars: np.ndarray, directions: list[str]) -> pd.DataFrame:
    """Per event, hindsight-optimal exit hour: argmax cumulative return for a
    POSITIVE event (best time to have sold for max gain), argmin for a
    NEGATIVE event (best time to have sold to cap loss). See module
    docstring's documented limitation before treating this as a live rule."""
    rows = []
    for car, direction in zip(event_cars, directions):
        if np.all(np.isnan(car)):
            continue
        h = (np.nanargmax(car) if direction == "positive" else np.nanargmin(car)) + 1
        rows.append({"direction": direction, "optimal_exit_hour": int(h),
                    "car_at_optimal": float(car[h - 1])})
    return pd.DataFrame(rows, columns=["direction", "optimal_exit_hour", "car_at_optimal"])


# --------------------------------------------------------------------------
# Orchestration — one (cause_coin, price_symbol) pair, DB in, files out
# --------------------------------------------------------------------------

def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, (pd.Timestamp, datetime)):
        return o.isoformat()
    if isinstance(o, Path):
        return str(o)
    return str(o)


def run_one(conn, cfg, cause_coin: str, price_symbol: str) -> dict | None:
    """Run the full event study for one (cause_coin, price_symbol) pair.
    cause_coin == price_symbol -> own-coin reaction; otherwise -> spillover.
    Returns None (and logs why) if there are fewer than cfg.min_events
    events — too few to say anything reliable, so the pair is skipped
    rather than reported with a misleadingly precise p-value.
    """
    start, end, window_source = resolve_window(conn, price_symbol, cfg.start, cfg.end)

    articles = fetch_strong_articles(conn, cause_coin, start, end, cfg.sentiment_column, cfg.threshold)
    events = cluster_events(articles, cfg.cluster_window_hours)
    if len(events) < cfg.min_events:
        logger.warning("%s -> %s: only %d event(s) (< min_events=%d) — skipped",
                       cause_coin, price_symbol, len(events), cfg.min_events)
        return None

    # Extend the price fetch past `end` so events near the window's edge can
    # still see their full horizon where data happens to exist.
    close = fetch_hourly_close(conn, price_symbol, start, end + timedelta(hours=cfg.horizon_hours))
    event_cars = np.stack([event_car(close, t, cfg.horizon_hours) for t in events["event_time"]])

    rng = np.random.default_rng(7)
    baseline = baseline_car_samples(close, cfg.horizon_hours, cfg.n_baseline_samples, rng)

    prefix = f"{cause_coin.replace('/', '')}_to_{price_symbol.replace('/', '')}_{cfg.sentiment_column}"
    out = cfg.output_dir
    out.mkdir(parents=True, exist_ok=True)
    events.to_csv(out / f"{prefix}_events.csv", index=False)

    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "cause_coin": cause_coin, "price_target": price_symbol,
        "relationship": "own_coin" if cause_coin == price_symbol else "spillover",
        "sentiment_column": cfg.sentiment_column, "threshold": cfg.threshold,
        "cluster_window_hours": cfg.cluster_window_hours, "horizon_hours": cfg.horizon_hours,
        "alpha": cfg.significance,
        "window_source": window_source, "window_start": start.isoformat(), "window_end": end.isoformat(),
        "n_events_total": len(events),
    }
    for direction in ("positive", "negative"):
        mask = (events["direction"] == direction).to_numpy()
        if mask.sum() == 0:
            summary[direction] = None
            continue
        dir_cars = event_cars[mask]
        sig_table = horizon_significance(dir_cars, baseline, cfg.significance,
                                         n_bootstrap=cfg.n_bootstrap, rng=rng,
                                         min_effect_car=cfg.min_effect_car)
        onset = onset_lag(sig_table, cfg.onset_min_run)
        exits = per_event_optimal_exit(dir_cars, [direction] * int(mask.sum()))
        sig_table.to_csv(out / f"{prefix}_{direction}_curve.csv", index=False)
        exits.to_csv(out / f"{prefix}_{direction}_exits.csv", index=False)
        eh = exits["optimal_exit_hour"]
        summary[direction] = {
            "n_events": int(mask.sum()),
            "onset_hour": onset,
            "median_optimal_exit_hour": float(eh.median()) if len(eh) else None,
            "p25_optimal_exit_hour": float(eh.quantile(0.25)) if len(eh) else None,
            "p75_optimal_exit_hour": float(eh.quantile(0.75)) if len(eh) else None,
            "mean_car_at_median_exit": (
                float(sig_table.loc[sig_table["horizon_h"] == round(eh.median()), "mean_car"].iloc[0])
                if len(eh) and round(eh.median()) in set(sig_table["horizon_h"]) else None
            ),
        }

    with open(out / f"{prefix}_summary.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, default=_json_default)

    logger.info("%s -> %s: %d events (%d+/%d-) | pos onset=%s | neg onset=%s",
               cause_coin, price_symbol, len(events),
               int((events["direction"] == "positive").sum()),
               int((events["direction"] == "negative").sum()),
               summary.get("positive", {}).get("onset_hour") if summary.get("positive") else "n/a",
               summary.get("negative", {}).get("onset_hour") if summary.get("negative") else "n/a")
    return summary