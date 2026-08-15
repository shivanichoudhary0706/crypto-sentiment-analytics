"""
Shared TimescaleDB connection helper for the Kafka consumers.
"""

import psycopg2

from config.settings import settings


def get_db_connection():
    """
    Plain psycopg2 connection (not SQLAlchemy) — consumers need simple,
    explicit INSERT ... ON CONFLICT statements, which psycopg2 handles
    directly without an ORM layer in between. SQLAlchemy is reserved for
    the FastAPI layer in Week 10, where query flexibility matters more.
    """
    return psycopg2.connect(
        host=settings.timescaledb_host,
        port=settings.timescaledb_port,
        dbname=settings.timescaledb_db,
        user=settings.timescaledb_user,
        password=settings.timescaledb_password,
    )