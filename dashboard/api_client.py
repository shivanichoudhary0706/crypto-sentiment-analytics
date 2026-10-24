"""Thin HTTP client for the Crypto Signal Desk API, with a short cache."""
import os

import requests
import streamlit as st
from dotenv import load_dotenv

load_dotenv()
API_BASE = os.getenv("API_BASE_URL", "http://127.0.0.1:8000")
TIMEOUT_S = 10


class ApiError(RuntimeError):
    pass


@st.cache_data(ttl=60, show_spinner=False)
def get(path: str, **params):
    """GET an endpoint. Returns parsed JSON, or None on 404 (no data yet)."""
    try:
        r = requests.get(f"{API_BASE}{path}",
                         params={k: v for k, v in params.items() if v is not None},
                         timeout=TIMEOUT_S)
    except requests.RequestException as exc:
        raise ApiError(f"API not reachable at {API_BASE} ({exc.__class__.__name__})") from exc
    if r.status_code == 404:
        return None
    if not r.ok:
        raise ApiError(f"{path} -> HTTP {r.status_code}: {r.text[:200]}")
    return r.json()