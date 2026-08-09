"""
Async client for Binance's public kline (candlestick) WebSocket stream.
No API key required — this is public market data.
"""

import json
from typing import AsyncGenerator, List

import websockets

from config.logging_config import get_logger
from ingestion.market.schemas import MarketKlineMessage

logger = get_logger(__name__)

BASE_WS_URL = "wss://data-stream.binance.vision"


def _build_stream_url(symbols: List[str]) -> str:
    """
    Binance combined-stream endpoint takes lowercase symbols with no slash,
    e.g. BTC/USDT -> btcusdt@kline_1m
    """
    streams = [f"{s.replace('/', '').lower()}@kline_1m" for s in symbols]
    stream_path = "/".join(streams)
    return f"{BASE_WS_URL}/stream?streams={stream_path}"


def _parse_kline_event(raw_message: dict) -> MarketKlineMessage:
    """
    Combined stream wraps each event as {"stream": "...", "data": {...}}.
    The actual kline payload is under data['k'].
    """
    data = raw_message["data"]
    k = data["k"]

    return MarketKlineMessage(
        symbol=data["s"],
        exchange="binance",
        event_time=data["E"],
        kline_start_time=k["t"],
        kline_close_time=k["T"],
        open=float(k["o"]),
        high=float(k["h"]),
        low=float(k["l"]),
        close=float(k["c"]),
        volume=float(k["v"]),
        is_closed=bool(k["x"]),
    )


async def stream_klines(symbols: List[str]) -> AsyncGenerator[MarketKlineMessage, None]:
    """
    Connects to Binance's public kline WebSocket and yields parsed messages
    indefinitely. Caller is responsible for wrapping this in reconnect logic
    (see run_market_ingestion.py) since a single dropped connection should
    not kill the whole ingestion process.
    """
    url = _build_stream_url(symbols)
    logger.info(f"Connecting to Binance WebSocket: {url}")

    async with websockets.connect(url, ping_interval=20, ping_timeout=20) as ws:
        logger.info(f"Connected. Streaming klines for: {symbols}")
        async for raw in ws:
            try:
                parsed = json.loads(raw)
                message = _parse_kline_event(parsed)
                yield message
            except (KeyError, json.JSONDecodeError) as e:
                logger.warning(f"Skipping malformed message: {e} | raw={raw[:200]}")
                continue