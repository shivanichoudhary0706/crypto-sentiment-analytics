"""
VADER scorer — fast, general-purpose sentiment, used as a second opinion
alongside FinBERT's finance-tuned score.
"""

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

_analyzer = SentimentIntensityAnalyzer()


def score_text(text: str) -> float:
    """
    Returns VADER's compound score: a single float from -1 (most negative)
    to +1 (most positive).
    """
    if not text:
        return 0.0
    return _analyzer.polarity_scores(text)["compound"]