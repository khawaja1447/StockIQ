"""
Sentiment analysis module for StockIQ.

Three free, no-auth data sources:
  1. StockTwits public API  → trader mood with official Bullish/Bearish labels
  2. Google News RSS        → mainstream financial news headlines
  3. Yahoo Finance news     → company-specific news via yfinance

All scored with VADER NLP. Composite score ∈ [-1, +1].
"""

from __future__ import annotations

import re
import requests
import xml.etree.ElementTree as ET
import streamlit as st
import yfinance as yf
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from typing import List, Dict, Tuple

from src.data.fetcher import ALL_ASSETS

# ──────────────────────────────────────────────────────────────────
_analyzer = SentimentIntensityAnalyzer()
_HEADERS   = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    )
}
_TIMEOUT = 8

# Google News search queries per asset
_GN_QUERIES: Dict[str, str] = {
    "BTC-USD":   "Bitcoin BTC cryptocurrency price",
    "ETH-USD":   "Ethereum ETH cryptocurrency price",
    "BNB-USD":   "BNB Binance coin price",
    "SOL-USD":   "Solana SOL crypto price",
    "ADA-USD":   "Cardano ADA crypto price",
    "XRP-USD":   "XRP Ripple crypto price",
    "DOGE-USD":  "Dogecoin DOGE price",
    "AVAX-USD":  "Avalanche AVAX crypto",
    "MATIC-USD": "Polygon MATIC crypto",
    "DOT-USD":   "Polkadot DOT crypto",
    "AAPL":      "Apple AAPL stock earnings",
    "TSLA":      "Tesla TSLA stock",
    "NVDA":      "NVIDIA NVDA stock AI chips",
    "MSFT":      "Microsoft MSFT stock",
    "AMZN":      "Amazon AMZN stock",
    "GOOGL":     "Alphabet Google GOOGL stock",
    "META":      "Meta Platforms META stock",
    "SPY":       "S&P 500 stock market",
    "AMD":       "AMD semiconductor stock",
    "NFLX":      "Netflix NFLX stock",
}


# ──────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────

def _score_text(text: str) -> float:
    """Return VADER compound score in [-1, +1]."""
    if not text or not isinstance(text, str):
        return 0.0
    return _analyzer.polarity_scores(text)["compound"]


def _label(score: float) -> str:
    if score >= 0.05:
        return "Positive"
    if score <= -0.05:
        return "Negative"
    return "Neutral"


def _st_symbol(ticker: str) -> str:
    """Convert yfinance ticker to StockTwits symbol (BTC-USD → BTC.X)."""
    return ticker.replace("-USD", ".X") if ticker.endswith("-USD") else ticker


# ──────────────────────────────────────────────────────────────────
# Source 1 — StockTwits
# ──────────────────────────────────────────────────────────────────

@st.cache_data(ttl=900, show_spinner=False)
def fetch_stocktwits_sentiment(ticker: str) -> Tuple[float, List[Dict]]:
    """
    Fetch the latest 30 messages from StockTwits for *ticker*.

    StockTwits provides official Bullish / Bearish labels set by the poster.
    When the label is absent, VADER scores the message body instead.

    Returns:
        (weighted_score, list_of_message_dicts)
    """
    symbol = _st_symbol(ticker)
    url    = f"https://api.stocktwits.com/api/2/streams/symbol/{symbol}.json"

    try:
        resp = requests.get(url, headers=_HEADERS, timeout=_TIMEOUT)
        if resp.status_code != 200:
            return 0.0, []

        messages = resp.json().get("messages", [])
        posts: List[Dict] = []
        scores: List[float] = []

        for msg in messages:
            body = (msg.get("body") or "").strip()
            if not body:
                continue

            # Use official StockTwits label when available — it's a stronger signal
            st_sent = (msg.get("entities") or {}).get("sentiment") or {}
            st_basic = st_sent.get("basic", "")

            if st_basic == "Bullish":
                score = 0.65
            elif st_basic == "Bearish":
                score = -0.65
            else:
                score = _score_text(body)

            label  = st_basic if st_basic else _label(score)
            likes  = (msg.get("likes") or {}).get("total", 0)
            user   = (msg.get("user") or {}).get("username", "trader")

            scores.append(score)
            posts.append({
                "source": f"@{user}",
                "title":  body[:120],
                "score":  score,
                "label":  label,
                "likes":  likes,
                "url":    f"https://stocktwits.com/symbol/{symbol}",
            })

        if not scores:
            return 0.0, []

        # Weighted by likes — popular posts carry more signal
        weights = [max(p["likes"], 1) for p in posts]
        total_w = sum(weights)
        weighted = sum(s * w for s, w in zip(scores, weights)) / total_w

        posts.sort(key=lambda p: abs(p["score"]), reverse=True)
        return float(weighted), posts[:20]

    except Exception:
        return 0.0, []


# ──────────────────────────────────────────────────────────────────
# Source 2 — Google News RSS
# ──────────────────────────────────────────────────────────────────

@st.cache_data(ttl=900, show_spinner=False)
def fetch_google_news_sentiment(ticker: str) -> Tuple[float, List[Dict]]:
    """
    Fetch financial headlines from Google News RSS for *ticker*.
    No API key, no rate limits.

    Returns:
        (aggregate_score, list_of_article_dicts)
    """
    query   = _GN_QUERIES.get(
        ticker,
        f"{ALL_ASSETS.get(ticker, ticker.replace('-USD',''))} {ticker.replace('-USD','')} price",
    )
    encoded = requests.utils.quote(query)
    url     = (
        f"https://news.google.com/rss/search"
        f"?q={encoded}&hl=en-US&gl=US&ceid=US:en"
    )

    try:
        resp = requests.get(url, headers=_HEADERS, timeout=_TIMEOUT)
        if resp.status_code != 200:
            return 0.0, []

        items    = ET.fromstring(resp.content).findall(".//item")
        articles: List[Dict] = []
        scores:   List[float] = []

        for item in items[:25]:
            raw_title = item.findtext("title", "").strip()
            # Strip "- Publisher" suffix Google appends
            title = re.sub(r"\s+[-–]\s+[^-–]+$", "", raw_title).strip()
            title = re.sub(r"<[^>]+>", "", title).strip()   # remove any HTML

            if not title or len(title) < 10:
                continue

            src_el = item.find("source")
            source = src_el.text.strip() if src_el is not None else "Google News"
            link   = item.findtext("link", "")

            score = _score_text(title)
            scores.append(score)
            articles.append({
                "source": source,
                "title":  title[:120],
                "score":  score,
                "label":  _label(score),
                "url":    link,
            })

        if not scores:
            return 0.0, []

        aggregate = float(sum(scores) / len(scores))
        articles.sort(key=lambda a: abs(a["score"]), reverse=True)
        return aggregate, articles[:20]

    except Exception:
        return 0.0, []


# ──────────────────────────────────────────────────────────────────
# Source 3 — Yahoo Finance News (via yfinance)
# ──────────────────────────────────────────────────────────────────

@st.cache_data(ttl=900, show_spinner=False)
def fetch_yahoo_news_sentiment(ticker: str) -> Tuple[float, List[Dict]]:
    """
    Fetch recent headlines for *ticker* from Yahoo Finance via yfinance.
    Handles both old (flat) and new (nested content) API shapes.

    Returns:
        (aggregate_score, list_of_article_dicts)
    """
    try:
        news_items = yf.Ticker(ticker).news or []
    except Exception:
        return 0.0, []

    articles: List[Dict] = []
    scores:   List[float] = []

    for item in news_items[:20]:
        content = item.get("content") or item

        title   = (
            content.get("title") or content.get("headline")
            or item.get("title") or ""
        )
        summary = (
            content.get("summary") or content.get("description")
            or item.get("summary") or ""
        )

        title   = re.sub(r"<[^>]+>", "", title).strip()
        summary = re.sub(r"<[^>]+>", "", summary).strip()

        if not title:
            continue

        score = _score_text(f"{title}. {summary}"[:512])
        scores.append(score)

        provider = content.get("provider") or {}
        source   = (
            provider.get("displayName") or item.get("publisher") or "Yahoo Finance"
        )
        url = (
            (content.get("canonicalUrl") or {}).get("url")
            or (content.get("clickThroughUrl") or {}).get("url")
            or item.get("link") or ""
        )

        articles.append({
            "source": source,
            "title":  title[:120],
            "score":  score,
            "label":  _label(score),
            "url":    url,
        })

    if not scores:
        return 0.0, []

    aggregate = float(sum(scores) / len(scores))
    articles.sort(key=lambda a: abs(a["score"]), reverse=True)
    return aggregate, articles[:20]


# ──────────────────────────────────────────────────────────────────
# Combined Sentiment
# ──────────────────────────────────────────────────────────────────

@st.cache_data(ttl=900, show_spinner=False)
def get_combined_sentiment(ticker: str) -> Dict:
    """
    Merge StockTwits + Google News + Yahoo Finance into one composite score.

    Weights
    -------
    Crypto  : StockTwits 45% · Google News 35% · Yahoo Finance 20%
    Stocks  : StockTwits 30% · Google News 40% · Yahoo Finance 30%

    Returns
    -------
    dict with keys:
        score          composite score [-1, +1]
        label          "Positive" | "Negative" | "Neutral"
        social_score   StockTwits sub-score
        news_score     Google News sub-score
        yahoo_score    Yahoo Finance sub-score
        social_posts   StockTwits messages
        news_articles  Google News articles
        yahoo_articles Yahoo Finance articles
        available      bool
    """
    st_score,  st_posts    = fetch_stocktwits_sentiment(ticker)
    gn_score,  gn_articles = fetch_google_news_sentiment(ticker)
    yf_score,  yf_articles = fetch_yahoo_news_sentiment(ticker)

    st_ok = len(st_posts)    > 0
    gn_ok = len(gn_articles) > 0
    yf_ok = len(yf_articles) > 0

    is_crypto = "-USD" in ticker

    if is_crypto:
        W_ST, W_GN, W_YF = 0.45, 0.35, 0.20
    else:
        W_ST, W_GN, W_YF = 0.30, 0.40, 0.30

    # Normalise weights for available sources only
    avail_w = (W_ST if st_ok else 0) + (W_GN if gn_ok else 0) + (W_YF if yf_ok else 0)
    if avail_w == 0:
        return {
            "score": 0.0, "label": "Neutral",
            "social_score": 0.0, "news_score": 0.0, "yahoo_score": 0.0,
            "social_posts": [], "news_articles": [], "yahoo_articles": [],
            "available": False,
        }

    composite = (
        (st_score * W_ST if st_ok else 0)
        + (gn_score * W_GN if gn_ok else 0)
        + (yf_score * W_YF if yf_ok else 0)
    ) / avail_w

    return {
        "score":          float(composite),
        "label":          _label(composite),
        "social_score":   float(st_score),
        "news_score":     float(gn_score),
        "yahoo_score":    float(yf_score),
        "social_posts":   st_posts,
        "news_articles":  gn_articles,
        "yahoo_articles": yf_articles,
        "available":      True,
    }
