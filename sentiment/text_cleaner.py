"""
Strips HTML from RSS summaries and builds the final text sent to sentiment models.
"""

from bs4 import BeautifulSoup


def clean_text(raw_text: str) -> str:
    """
    Strips HTML tags (RSS summaries often contain raw <p>, <img>, etc.)
    and collapses whitespace. Returns plain text suitable for FinBERT/VADER.
    """
    if not raw_text:
        return ""
    soup = BeautifulSoup(raw_text, "html.parser")
    text = soup.get_text(separator=" ")
    return " ".join(text.split())


def build_scoring_text(title: str, summary: str) -> str:
    """
    Combines title + cleaned summary into the text actually sent to the models.
    Title carries most of the signal in financial news headlines, so it's
    always included even if the summary is empty (GDELT articles have none).
    """
    cleaned_summary = clean_text(summary)
    if cleaned_summary:
        return f"{title}. {cleaned_summary}"
    return title