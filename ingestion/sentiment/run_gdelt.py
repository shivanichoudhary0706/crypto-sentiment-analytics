"""
Run only the GDELT poller.
Run with: python -m ingestion.sentiment.run_gdelt
"""

from ingestion.sentiment.run_sentiment_ingestion import main

if __name__ == "__main__":
    main("gdelt")