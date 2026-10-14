"""
TimescaleDB I/O for the signal job: read features_bar, create and upsert the
signals table. No model logic lives here.
"""
from __future__ import annotations

from datetime import datetime

import pandas as pd
from psycopg2 import sql
from psycopg2.extras import execute_values

from signals.dataset import BASE_INPUT_COLUMNS, NEWS_SUFFIXES

SIGNAL_COLUMNS = ("bucket", "valid_from", "symbol", "coin", "freq", "horizon_bars", "close",
                  "prob_up", "action", "model_version", "n_train")


def ensure_schema(conn, table: str) -> None:
    ddl = sql.SQL("""
CREATE TABLE IF NOT EXISTS {t} (
    bucket        TIMESTAMPTZ      NOT NULL,   -- closed bar the signal was computed from
    valid_from    TIMESTAMPTZ      NOT NULL,   -- start of the bar the signal is for (bucket + freq)
    symbol        TEXT             NOT NULL,
    coin          TEXT             NOT NULL,
    freq          TEXT             NOT NULL,
    horizon_bars  INTEGER          NOT NULL,
    close         DOUBLE PRECISION,            -- price when the signal was issued
    prob_up       DOUBLE PRECISION NOT NULL,   -- model's probability of a rise over the horizon
    action        TEXT             NOT NULL CHECK (action IN ('BUY', 'HOLD', 'SELL')),
    model_version TEXT             NOT NULL,
    n_train       INTEGER          NOT NULL,
    created_at    TIMESTAMPTZ      NOT NULL DEFAULT now(),
    PRIMARY KEY (symbol, freq, bucket)
);
CREATE INDEX IF NOT EXISTS {idx} ON {t} (bucket DESC);
""").format(t=sql.Identifier(table), idx=sql.Identifier(f"{table}_bucket_idx"))
    with conn.cursor() as cur:
        cur.execute(ddl)
    conn.commit()


def _features_columns(conn, features_table: str) -> list[str]:
    with conn.cursor() as cur:
        cur.execute("SELECT column_name FROM information_schema.columns WHERE table_name = %s",
                    (features_table,))
        present = [r[0] for r in cur.fetchall()]
    conn.commit()
    return [c for c in BASE_INPUT_COLUMNS if c in present] + \
           sorted(c for c in present if c.endswith(NEWS_SUFFIXES))


def fetch_features(conn, features_table: str, coins, freq: str,
                   since: datetime | None) -> pd.DataFrame:
    """features_bar rows for the given coins at one frequency (optionally from `since`)."""
    cols = _features_columns(conn, features_table)
    missing = [c for c in BASE_INPUT_COLUMNS if c not in cols]
    if missing:
        raise RuntimeError(f"{features_table} lacks columns {missing}; run the feature job first")
    where = [sql.SQL("freq = %(freq)s"), sql.SQL("coin = ANY(%(coins)s)")]
    params = {"freq": freq, "coins": list(coins)}
    if since is not None:
        where.append(sql.SQL("bucket >= %(since)s"))
        params["since"] = since
    q = sql.SQL("SELECT {cols} FROM {t} WHERE {w} ORDER BY coin, bucket").format(
        cols=sql.SQL(", ").join(sql.Identifier(c) for c in cols),
        t=sql.Identifier(features_table), w=sql.SQL(" AND ").join(where))
    with conn.cursor() as cur:
        cur.execute(q, params)
        rows = cur.fetchall()
    conn.commit()
    df = pd.DataFrame(rows, columns=cols)
    if not df.empty:
        df["bucket"] = pd.to_datetime(df["bucket"], utc=True)
        num = [c for c in cols if c not in ("bucket", "coin", "is_complete")]
        df[num] = df[num].astype(float)
    return df


def upsert_signals(conn, table: str, signals: pd.DataFrame) -> int:
    if signals.empty:
        return 0
    cols = sql.SQL(", ").join(sql.Identifier(c) for c in SIGNAL_COLUMNS)
    updates = sql.SQL(", ").join(
        sql.SQL("{c} = EXCLUDED.{c}").format(c=sql.Identifier(c))
        for c in SIGNAL_COLUMNS if c not in ("symbol", "freq", "bucket"))
    q = sql.SQL("INSERT INTO {t} ({cols}) VALUES %s "
                "ON CONFLICT (symbol, freq, bucket) DO UPDATE SET {u}, created_at = now()").format(
        t=sql.Identifier(table), cols=cols, u=updates)
    records = []
    for r in signals[list(SIGNAL_COLUMNS)].itertuples(index=False, name=None):
        records.append(tuple(None if (isinstance(v, float) and pd.isna(v)) else
                             (v.to_pydatetime() if isinstance(v, pd.Timestamp) else
                              (v.item() if hasattr(v, "item") else v)) for v in r))
    with conn.cursor() as cur:
        execute_values(cur, q.as_string(conn), records)
    conn.commit()
    return len(records)