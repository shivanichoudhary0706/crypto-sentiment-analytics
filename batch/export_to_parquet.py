"""
Export TimescaleDB tables to a Hive-partitioned Parquet lake (batch layer input).

Run from project root:
    python -m batch.export_to_parquet --start 2026-06-01 --end 2026-09-27
    python -m batch.export_to_parquet --start 2026-06-01 --end 2026-09-27 --overwrite

--end is exclusive. Existing day-files are skipped unless --overwrite is given;
re-run with --overwrite after any backfill so updated days are refreshed.
"""
import argparse
import logging
import os
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import psycopg2
import pyarrow as pa
import pyarrow.parquet as pq

log = logging.getLogger("batch.export")

LAKE_ROOT = Path(os.getenv("LAKE_ROOT", "data/lake"))

# partition_col: column the lake is partitioned by
# dir_key:       folder name used in the path (key=value)
# keep_col:      keep the original column inside the file (needed when the
#                raw value is not path-safe, e.g. "BTC/USDT")
TABLES = {
    "market_data": {
        "partition_col": "symbol", "dir_key": "symbol", "keep_col": False,
        "float_cols": ["open", "high", "low", "close", "volume"],
        "sql": ("SELECT time, symbol, open, high, low, close, volume "
                "FROM market_data WHERE time >= %s AND time < %s"),
    },
    "sentiment_scores": {
        "partition_col": "coin", "dir_key": "coin_key", "keep_col": True,
        "float_cols": ["finbert_compound", "vader_compound"],
        "sql": ("SELECT time, url, coin, source_name, finbert_label, "
                "finbert_compound, vader_compound "
                "FROM sentiment_scores WHERE time >= %s AND time < %s"),
    },
}


def safe_value(value: str) -> str:
    """'BTC/USDT' would create a nested folder; partition values must be flat."""
    return str(value).replace("/", "-")


def partition_path(table: str, value: str, day: date) -> Path:
    key = TABLES[table]["dir_key"]
    return (LAKE_ROOT / table / f"{key}={safe_value(value)}"
            / f"date={day.isoformat()}" / "part-0.parquet")


def _connect():
    return psycopg2.connect(
        host=os.getenv("DB_HOST", "localhost"),
        port=int(os.getenv("DB_PORT", "5432")),
        dbname=os.getenv("DB_NAME", "crypto_analytics"),
        user=os.getenv("DB_USER", "crypto_user"),
        password=os.getenv("DB_PASSWORD", os.getenv("TIMESCALEDB_PASSWORD", "")),
    )


def _fetch_day(conn, table: str, day: date) -> pd.DataFrame:
    spec = TABLES[table]
    start = datetime(day.year, day.month, day.day, tzinfo=timezone.utc)
    with conn.cursor() as cur:
        cur.execute(spec["sql"], (start, start + timedelta(days=1)))
        cols = [d[0] for d in cur.description]
        df = pd.DataFrame(cur.fetchall(), columns=cols)
    if df.empty:
        return df
    df["time"] = pd.to_datetime(df["time"], utc=True)
    for c in spec["float_cols"]:            # NUMERIC arrives as Decimal -> float64
        df[c] = pd.to_numeric(df[c], errors="coerce").astype("float64")
    return df.sort_values("time")


def export_day(conn, table: str, day: date, overwrite: bool) -> int:
    spec = TABLES[table]
    df = _fetch_day(conn, table, day)
    written = 0
    for value, part in df.groupby(spec["partition_col"]):
        path = partition_path(table, value, day)
        if path.exists() and not overwrite:
            continue
        if not spec["keep_col"]:
            part = part.drop(columns=[spec["partition_col"]])  # Spark restores it from the path
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + ".tmp")
        pq.write_table(pa.Table.from_pandas(part, preserve_index=False), tmp)
        os.replace(tmp, path)   # atomic swap: a crash never leaves a half-written file
        written += len(part)
    return written


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--start", required=True, type=date.fromisoformat)
    ap.add_argument("--end", required=True, type=date.fromisoformat, help="exclusive")
    ap.add_argument("--tables", default=",".join(TABLES))
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    tables = [t.strip() for t in args.tables.split(",")]
    unknown = set(tables) - set(TABLES)
    if unknown:
        raise SystemExit(f"Unknown tables: {sorted(unknown)}")

    totals = {t: 0 for t in tables}
    conn = _connect()
    try:
        day = args.start
        while day < args.end:
            for t in tables:
                n = export_day(conn, t, day, args.overwrite)
                totals[t] += n
            log.info("%s done: %s", day, {t: totals[t] for t in tables})
            day += timedelta(days=1)
    finally:
        conn.close()
    log.info("Export complete. Rows written: %s (lake: %s)", totals, LAKE_ROOT.resolve())


if __name__ == "__main__":
    main()