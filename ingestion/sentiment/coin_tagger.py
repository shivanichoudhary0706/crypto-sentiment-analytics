"""
Shared coin-tagging logic, used by both the News RSS and GDELT pollers.
Keeps "which words mean which coin" config-driven (config.yaml's coin_keywords),
not hardcoded per-source.
"""

from typing import List

from config.settings import settings


def tag_coins(text: str) -> List[str]:
    """
    Case-insensitive keyword match against config.yaml's coin_keywords block.
    Returns the list of coin symbols (e.g. ["BTC/USDT"]) mentioned in the text.
    An article can match multiple coins, or none.
    """
    text_lower = text.lower()
    matched = []
    for symbol, keywords in settings.coin_keywords.items():
        if any(keyword in text_lower for keyword in keywords):
            matched.append(symbol)
    return matched