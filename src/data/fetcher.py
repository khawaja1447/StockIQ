"""
Price data fetching module for StockIQ.
Handles both stocks (via yfinance) and crypto (via yfinance + CoinGecko/Alternative.me).
All fetch functions are decorated with Streamlit cache to minimise API calls.
"""

from __future__ import annotations

import requests
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from typing import Dict, Optional

# ──────────────────────────────────────────────────────────────────
# Asset Catalogue
# ──────────────────────────────────────────────────────────────────

STOCKS: Dict[str, str] = {
    "AAPL":  "Apple Inc.",
    "TSLA":  "Tesla Inc.",
    "NVDA":  "NVIDIA Corporation",
    "MSFT":  "Microsoft Corporation",
    "AMZN":  "Amazon.com Inc.",
    "GOOGL": "Alphabet Inc.",
    "META":  "Meta Platforms Inc.",
    "SPY":   "S&P 500 ETF",
    "AMD":   "Advanced Micro Devices",
    "NFLX":  "Netflix Inc.",
}

CRYPTO: Dict[str, str] = {
    "BTC-USD":   "Bitcoin",
    "ETH-USD":   "Ethereum",
    "BNB-USD":   "BNB",
    "SOL-USD":   "Solana",
    "ADA-USD":   "Cardano",
    "XRP-USD":   "XRP",
    "DOGE-USD":  "Dogecoin",
    "AVAX-USD":  "Avalanche",
    "MATIC-USD": "Polygon",
    "DOT-USD":   "Polkadot",
}

ALL_ASSETS: Dict[str, str] = {**STOCKS, **CRYPTO}

# Subreddits used for sentiment scraping
REDDIT_SUBS: Dict[str, list[str]] = {
    "BTC-USD":   ["Bitcoin", "CryptoCurrency"],
    "ETH-USD":   ["ethereum", "CryptoCurrency"],
    "BNB-USD":   ["binance", "CryptoCurrency"],
    "SOL-USD":   ["solana", "CryptoCurrency"],
    "ADA-USD":   ["cardano", "CryptoCurrency"],
    "XRP-USD":   ["Ripple", "CryptoCurrency"],
    "DOGE-USD":  ["dogecoin", "CryptoCurrency"],
    "AVAX-USD":  ["Avax", "CryptoCurrency"],
    "MATIC-USD": ["0xPolygon", "CryptoCurrency"],
    "DOT-USD":   ["dot", "CryptoCurrency"],
    "AAPL":      ["apple", "wallstreetbets", "stocks"],
    "TSLA":      ["teslainvestorsclub", "wallstreetbets"],
    "NVDA":      ["nvidia", "wallstreetbets"],
    "MSFT":      ["microsoft", "stocks"],
    "AMZN":      ["amazon", "wallstreetbets"],
    "GOOGL":     ["google", "stocks"],
    "META":      ["facebook", "stocks"],
    "SPY":       ["wallstreetbets", "investing"],
    "AMD":       ["AMD_Stock", "hardware"],
    "NFLX":      ["netflix", "wallstreetbets"],
}

TIMEFRAME_PERIODS: Dict[str, str] = {
    "1 Week":   "5d",
    "1 Month":  "1mo",
    "3 Months": "3mo",
    "6 Months": "6mo",
    "1 Year":   "1y",
    "2 Years":  "2y",
}

TRAINING_PERIOD = "2y"   # Always train on 2 years of daily data


# ──────────────────────────────────────────────────────────────────
# Price Data
# ──────────────────────────────────────────────────────────────────

@st.cache_data(ttl=1800, show_spinner=False)
def fetch_price_data(ticker: str, period: str = "2y") -> pd.DataFrame:
    """
    Download OHLCV daily data for *ticker* using yfinance.

    Uses Ticker.history() — returns clean flat columns in all yfinance versions.
    Falls back to yf.download() if history() fails.

    Returns a clean DataFrame with columns [Open, High, Low, Close, Volume]
    indexed by date, or an empty DataFrame on failure.
    """
    def _clean(df: pd.DataFrame) -> pd.DataFrame:
        """Normalise columns and dtypes."""
        # Flatten MultiIndex if present (yf.download in 1.4+)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        required = {"Open", "High", "Low", "Close", "Volume"}
        if not required.issubset(set(df.columns)):
            return pd.DataFrame()

        df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
        # Strip timezone so index is plain dates
        df.index = pd.to_datetime(df.index).tz_localize(None)
        df = df.sort_index()
        df = df.apply(pd.to_numeric, errors="coerce")
        df = df.dropna()
        return df

    # ── Primary: Ticker.history() — clean flat columns every time ──
    try:
        raw = yf.Ticker(ticker).history(period=period, interval="1d", auto_adjust=True)
        if not raw.empty:
            result = _clean(raw)
            if not result.empty:
                return result
    except Exception:
        pass

    # ── Fallback: yf.download() ────────────────────────────────────
    try:
        raw = yf.download(ticker, period=period, interval="1d",
                          progress=False, auto_adjust=True)
        if not raw.empty:
            result = _clean(raw)
            if not result.empty:
                return result
    except Exception:
        pass

    return pd.DataFrame()


# ──────────────────────────────────────────────────────────────────
# Ticker Metadata
# ──────────────────────────────────────────────────────────────────

@st.cache_data(ttl=7200, show_spinner=False)
def fetch_ticker_info(ticker: str) -> dict:
    """Return a dict of metadata for *ticker*."""
    fallback = {
        "name":       ALL_ASSETS.get(ticker, ticker),
        "sector":     "Cryptocurrency" if "-USD" in ticker else "Equity",
        "market_cap": 0,
        "52w_high":   None,
        "52w_low":    None,
        "currency":   "USD",
    }
    try:
        info = yf.Ticker(ticker).fast_info
        return {
            "name":       ALL_ASSETS.get(ticker, ticker),
            "sector":     "Cryptocurrency" if "-USD" in ticker else "Equity",
            "market_cap": getattr(info, "market_cap", 0) or 0,
            "52w_high":   getattr(info, "year_high", None),
            "52w_low":    getattr(info, "year_low",  None),
            "currency":   getattr(info, "currency", "USD"),
        }
    except Exception:
        return fallback


# ──────────────────────────────────────────────────────────────────
# Fear & Greed Index  (Alternative.me — free, no key needed)
# ──────────────────────────────────────────────────────────────────

@st.cache_data(ttl=7200, show_spinner=False)
def fetch_fear_greed_index() -> dict:
    """
    Fetch the Crypto Fear & Greed Index (30-day history).

    Returns:
        {value: int, classification: str, history: pd.DataFrame}
    """
    empty = {
        "value": 50,
        "classification": "Neutral",
        "history": pd.DataFrame(columns=["date", "value", "classification"]),
    }
    try:
        resp = requests.get(
            "https://api.alternative.me/fng/?limit=30&format=json",
            timeout=8,
            headers={"User-Agent": "StockIQ/1.0"},
        )
        resp.raise_for_status()
        data = resp.json().get("data", [])
        if not data:
            return empty

        current = data[0]
        hist = pd.DataFrame(data)
        hist["value"] = pd.to_numeric(hist["value"])
        hist["date"] = pd.to_datetime(hist["timestamp"].astype(int), unit="s")
        hist = (
            hist[["date", "value", "value_classification"]]
            .rename(columns={"value_classification": "classification"})
            .sort_values("date")
            .reset_index(drop=True)
        )
        return {
            "value":          int(current["value"]),
            "classification": current["value_classification"],
            "history":        hist,
        }
    except Exception:
        return empty
