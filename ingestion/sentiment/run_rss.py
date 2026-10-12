"""
Run only the News RSS poller.
Run with: python -m ingestion.sentiment.run_rss
"""

from ingestion.sentiment.run_sentiment_ingestion import main

if __name__ == "__main__":
    main("rss")