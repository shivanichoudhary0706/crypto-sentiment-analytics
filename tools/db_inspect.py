"""Inspect every table in the public schema: columns, row count, sample rows,
and a full CSV export per table.

Run from project root:
    python -m tools.db_inspect
    python -m tools.db_inspect --sample 10
"""
import argparse
import os
from pathlib import Path

import psycopg2
from psycopg2 import sql

# Fill these in (or set as environment variables)
CONN = dict(
    host=os.getenv("DB_HOST", "localhost"),
    port=int(os.getenv("DB_PORT", "5432")),      # check docker-compose host port
    dbname=os.getenv("DB_NAME", "REPLACE_ME"),
    user=os.getenv("DB_USER", "REPLACE_ME"),
    password=os.getenv("DB_PASSWORD", "REPLACE_ME"),
)

EXPORT_DIR = Path("exports")
MAX_CELL = 60  # truncate long text in console output only (CSV is untouched)


def short(value):
    text = str(value)
    return text if len(text) <= MAX_CELL else text[: MAX_CELL - 3] + "..."


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", type=int, default=5, help="rows to print per table")
    args = parser.parse_args()

    EXPORT_DIR.mkdir(exist_ok=True)

    conn = psycopg2.connect(**CONN)
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT table_name FROM information_schema.tables
                WHERE table_schema = 'public' AND table_type = 'BASE TABLE'
                ORDER BY table_name
                """
            )
            tables = [r[0] for r in cur.fetchall()]
            print(f"Found {len(tables)} table(s): {', '.join(tables)}\n")

            for table in tables:
                ident = sql.Identifier(table)

                cur.execute(
                    """
                    SELECT column_name, data_type FROM information_schema.columns
                    WHERE table_schema = 'public' AND table_name = %s
                    ORDER BY ordinal_position
                    """,
                    (table,),
                )
                columns = cur.fetchall()

                cur.execute(sql.SQL("SELECT COUNT(*) FROM {}").format(ident))
                count = cur.fetchone()[0]

                print("=" * 80)
                print(f"TABLE: {table}   ROWS: {count}")
                print("-" * 80)
                for name, dtype in columns:
                    print(f"  {name:<28} {dtype}")

                # Sample rows (newest first if the first column is a time column)
                cur.execute(
                    sql.SQL("SELECT * FROM {} ORDER BY 1 DESC LIMIT %s").format(ident),
                    (args.sample,),
                )
                names = [d[0] for d in cur.description]
                print(f"\n  Sample ({args.sample} rows):")
                for row in cur.fetchall():
                    print("  " + " | ".join(f"{n}={short(v)}" for n, v in zip(names, row)))

                # Full export
                path = EXPORT_DIR / f"{table}.csv"
                copy_sql = sql.SQL(
                    "COPY (SELECT * FROM {} ORDER BY 1) TO STDOUT WITH CSV HEADER"
                ).format(ident).as_string(conn)
                with open(path, "w", newline="", encoding="utf-8") as f:
                    cur.copy_expert(copy_sql, f)
                print(f"\n  -> exported all rows to {path}\n")
    finally:
        conn.close()


if __name__ == "__main__":
    main()