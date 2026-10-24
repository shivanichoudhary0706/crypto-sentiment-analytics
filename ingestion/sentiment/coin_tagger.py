"""
Shared coin-tagging logic, used by the News RSS and GDELT pollers and by the
re-tagging step of tools/health_check.py.

Keeps "which words mean which coin" config-driven (config.yaml's coin_keywords),
not hardcoded per-source.

Matching is case-insensitive and on WHOLE WORDS: "eth" matches "ETH" and
"ETH's", but not "something", "whether" or "method".
"""

import re
from functools import lru_cache
from typing import Dict, List, Tuple

from config.settings import settings

MARKET_TAG = "MARKET"   # article mentions no configured coin -> market-wide crypto news


@lru_cache(maxsize=8)
def _compile(keyword_items: Tuple[Tuple[str, Tuple[str, ...]], ...]) -> Tuple[Tuple[str, "re.Pattern"], ...]:
    compiled = []
    for symbol, keywords in keyword_items:
        alternatives = "|".join(re.escape(k.lower()) for k in keywords if k)
        if alternatives:
            compiled.append((symbol, re.compile(rf"\b(?:{alternatives})\b")))
    return tuple(compiled)


def tag_coins(text: str, coin_keywords: Dict[str, List[str]] = None) -> List[str]:
    """
    Returns the coin symbols (e.g. ["BTC/USDT", "ETH/USDT"]) mentioned in the text,
    in config order. An article can match several coins; if it matches none it is
    tagged ["MARKET"].

    coin_keywords defaults to config.yaml's coin_keywords (the argument exists for tests).
    """
    keywords = settings.coin_keywords if coin_keywords is None else coin_keywords
    items = tuple((symbol, tuple(kws)) for symbol, kws in keywords.items())
    text_lower = (text or "").lower()
    matched = [symbol for symbol, pattern in _compile(items) if pattern.search(text_lower)]
    return matched or [MARKET_TAG]
