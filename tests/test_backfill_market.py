"""Retry behaviour of tools/backfill_market.fetch_klines. No network: the HTTP session is faked."""
import pytest
import requests

from tools import backfill_market as bm


class FakeResponse:
    def __init__(self, status=200, payload=None, headers=None):
        self.status_code, self._payload, self.headers, self.text = status, payload or [], headers or {}, ""

    def json(self):
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"HTTP {self.status_code}")


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch):
    monkeypatch.setattr(bm.time_module, "sleep", lambda s: None)


def _session(monkeypatch, outcomes):
    calls = []

    def get(url, params=None, timeout=None):
        calls.append(params)
        out = outcomes[len(calls) - 1]
        if isinstance(out, Exception):
            raise out
        return out

    monkeypatch.setattr(bm._SESSION, "get", get)
    return calls


def test_read_timeout_is_retried(monkeypatch):
    # The 2026-10-24 failure: the body read timed out mid-backfill.
    calls = _session(monkeypatch, [requests.exceptions.ConnectionError("Read timed out."),
                                   FakeResponse(payload=[[1]])])
    assert bm.fetch_klines("LINKUSDT", 0, 60000) == [[1]]
    assert len(calls) == 2


def test_rate_limit_is_retried(monkeypatch):
    calls = _session(monkeypatch, [FakeResponse(429, headers={"Retry-After": "1"}), FakeResponse(payload=[])])
    assert bm.fetch_klines("BTCUSDT", 0, 60000) == []
    assert len(calls) == 2


def test_ip_ban_is_never_retried(monkeypatch):
    calls = _session(monkeypatch, [FakeResponse(418)])
    with pytest.raises(bm.BinanceBanError):
        bm.fetch_klines("BTCUSDT", 0, 60000)
    assert len(calls) == 1


def test_gives_up_after_max_attempts(monkeypatch):
    _session(monkeypatch, [requests.exceptions.Timeout()] * bm.MAX_ATTEMPTS)
    with pytest.raises(requests.exceptions.Timeout):
        bm.fetch_klines("BTCUSDT", 0, 60000)


def test_bad_symbol_fails_fast(monkeypatch):
    calls = _session(monkeypatch, [FakeResponse(400)])
    with pytest.raises(requests.HTTPError):
        bm.fetch_klines("NOPEUSDT", 0, 60000)
    assert len(calls) == 1
