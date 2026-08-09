"""
Message schema for market data published to Kafka's raw-market-data topic.
"""

from dataclasses import dataclass, asdict


@dataclass
class MarketKlineMessage:
    """
    One 1-minute kline (candlestick) update from Binance.

    is_closed=False means this is a live/in-progress candle (updates every
    ~1 second as the market moves); is_closed=True means the 1-minute
    window has finished. Downstream consumers (Week 4 TimescaleDB writer)
    should typically filter to is_closed=True to avoid storing partial candles,
    unless a use case specifically needs live in-progress ticks.
    """
    symbol: str              # e.g. "BTCUSDT"
    exchange: str             # e.g. "binance"
    event_time: int           # epoch ms, when Binance emitted this event
    kline_start_time: int     # epoch ms, start of the 1-min window
    kline_close_time: int     # epoch ms, end of the 1-min window
    open: float
    high: float
    low: float
    close: float
    volume: float
    is_closed: bool           # True = candle finalized, False = still forming

    def to_dict(self) -> dict:
        return asdict(self)