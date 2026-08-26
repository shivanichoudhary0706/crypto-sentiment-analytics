"""
Polls raw_sentiment for articles not yet in sentiment_scores, scores them
with FinBERT + VADER, writes results. One row per (article, coin) pair —
articles with no coin_tags are skipped since an unattributed sentiment
score has no use in lag detection.

Run with: python -m sentiment.run_sentiment_scoring
"""

import time

from config.logging_config import get_logger
from config.settings import settings
from streaming.db import get_db_connection
from sentiment.text_cleaner import build_scoring_text
from sentiment.finbert_scorer import score_batch as finbert_score_batch
from sentiment.vader_scorer import score_text as vader_score_text

logger = get_logger(__name__)

POLL_INTERVAL_SECONDS = 60
BATCH_SIZE = settings.sentiment_batch_size  # 16, from config.yaml

FETCH_UNSCORED_SQL = """
    SELECT rs.time, rs.url, rs.title, rs.summary, rs.source_name, rs.coin_tags
    FROM raw_sentiment rs
    WHERE rs.coin_tags != '{}'
      AND NOT EXISTS (
          SELECT 1 FROM sentiment_scores ss
          WHERE ss.url = rs.url AND ss.time = rs.time
      )
    ORDER BY rs.time DESC
    LIMIT %(limit)s
"""

INSERT_SQL = """
    INSERT INTO sentiment_scores
        (time, url, coin, source_name, finbert_label, finbert_positive,
         finbert_negative, finbert_neutral, finbert_compound, vader_compound)
    VALUES
        (%(time)s, %(url)s, %(coin)s, %(source_name)s, %(finbert_label)s,
         %(finbert_positive)s, %(finbert_negative)s, %(finbert_neutral)s,
         %(finbert_compound)s, %(vader_compound)s)
    ON CONFLICT (time, url, coin) DO NOTHING
"""


def _fetch_unscored_articles(conn, limit: int):
    with conn.cursor() as cur:
        cur.execute(FETCH_UNSCORED_SQL, {"limit": limit})
        columns = [desc[0] for desc in cur.description]
        return [dict(zip(columns, row)) for row in cur.fetchall()]


def _score_and_store(conn, articles: list) -> int:
    texts = [build_scoring_text(a["title"], a["summary"]) for a in articles]
    finbert_results = finbert_score_batch(texts)

    written = 0
    with conn.cursor() as cur:
        for article, text, fb in zip(articles, texts, finbert_results):
            vader_compound = vader_score_text(text)

            for coin in article["coin_tags"]:
                row = {
                    "time": article["time"],
                    "url": article["url"],
                    "coin": coin,
                    "source_name": article["source_name"],
                    "finbert_label": fb["label"],
                    "finbert_positive": fb["positive"],
                    "finbert_negative": fb["negative"],
                    "finbert_neutral": fb["neutral"],
                    "finbert_compound": fb["compound"],
                    "vader_compound": vader_compound,
                }
                cur.execute(INSERT_SQL, row)
                if cur.rowcount > 0:
                    written += 1
    return written


def run() -> None:
    conn = get_db_connection()
    conn.autocommit = True

    logger.info("Sentiment scoring started. Polling for unscored articles every "
                f"{POLL_INTERVAL_SECONDS}s, batch size {BATCH_SIZE}.")

    total_scored = 0
    try:
        while True:
            articles = _fetch_unscored_articles(conn, BATCH_SIZE)

            if articles:
                written = _score_and_store(conn, articles)
                total_scored += written
                logger.info(f"Scored {len(articles)} articles -> {written} coin-tagged rows "
                            f"written (total so far: {total_scored})")
            else:
                logger.info("No unscored articles found this cycle.")

            time.sleep(POLL_INTERVAL_SECONDS)

    except KeyboardInterrupt:
        logger.info("Sentiment scoring stopped by user.")
    finally:
        conn.close()


if __name__ == "__main__":
    run()