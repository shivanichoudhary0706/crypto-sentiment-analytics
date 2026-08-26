"""
Sanity check for the sentiment scoring pipeline.
Run with: python tests/test_sentiment_scoring.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from streaming.db import get_db_connection

conn = get_db_connection()

with conn.cursor() as cur:
    cur.execute("SELECT count(*) FROM sentiment_scores;")
    count = cur.fetchone()[0]
    print(f"sentiment_scores row count: {count}")
    assert count > 0, "FAIL: No sentiment scores found — run sentiment scoring first."

    cur.execute("SELECT DISTINCT finbert_label FROM sentiment_scores;")
    labels = {row[0] for row in cur.fetchall()}
    print(f"FinBERT labels present: {labels}")
    assert labels.issubset({"positive", "negative", "neutral"}), f"FAIL: unexpected labels {labels}"

    cur.execute("SELECT min(finbert_compound), max(finbert_compound) FROM sentiment_scores;")
    fb_min, fb_max = cur.fetchone()
    print(f"FinBERT compound range: [{fb_min:.3f}, {fb_max:.3f}]")
    assert fb_min >= -1.0 and fb_max <= 1.0, "FAIL: FinBERT compound out of expected range"

    cur.execute("SELECT min(vader_compound), max(vader_compound) FROM sentiment_scores;")
    v_min, v_max = cur.fetchone()
    print(f"VADER compound range: [{v_min:.3f}, {v_max:.3f}]")
    assert v_min >= -1.0 and v_max <= 1.0, "FAIL: VADER compound out of expected range"

    cur.execute("SELECT DISTINCT coin FROM sentiment_scores;")
    coins = {row[0] for row in cur.fetchall()}
    print(f"Coins with scored sentiment: {coins}")

print("\nPASS: Sentiment scoring pipeline verified.")
conn.close()