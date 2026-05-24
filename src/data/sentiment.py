"""
Sentiment analysis module for StockIQ.

Fetches text from two free, no-auth sources:
  1. Reddit public JSON feed  → hot posts from relevant subreddits
  2. Yahoo Finance news       → recent headlines via yfinance

Scores each piece of text with VADER (optimised for short social-media text)
and returns an aggregate compound score in [-1, +1].
"""

from __future__ import annotations

import time
import requests
import streamlit as st
import yfinance as yf
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from typing import List, Dict, Tuple

from src.data.fetcher import REDDIT_SUBS

# ──────────────────────────────────────────────────────────────────
_analyzer = SentimentIntensityAnalyzer()
_REDDIT_HEADERS = {"User-Agent": "StockIQ/1.0 (research project)"}
_REDDIT_TIMEOUT = 6   # seconds per subreddit request


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


# ──────────────────────────────────────────────────────────────────
# Reddit (public, no auth)
# ──────────────────────────────────────────────────────────────────

@st.cache_data(ttl=1800, show_spinner=False)
def fetch_reddit_sentiment(ticker: str) -> Tuple[float, List[Dict]]:
    """
    Scrape hot posts from relevant subreddits for *ticker*.

    Returns:
        (aggregate_score, list_of_post_dicts)
        aggregate_score ∈ [-1, +1]
    """
    subreddits = REDDIT_SUBS.get(ticker, ["investing"])
    posts: List[Dict] = []
    scores: List[float] = []

    for sub in subreddits[:2]:          # max 2 subreddits to avoid rate limits
        url = f"https://www.reddit.com/r/{sub}/hot.json?limit=30"
        try:
            resp = requests.get(url, headers=_REDDIT_HEADERS, timeout=_REDDIT_TIMEOUT)
            if resp.status_code != 200:
                continue
            children = resp.json().get("data", {}).get("children", [])
            for child in children:
                post = child.get("data", {})
                title = post.get("title", "")
                selftext = post.get("selftext", "")
                combined = f"{title}. {selftext}"[:512]
                score = _score_text(combined)
                scores.append(score)
                posts.append(
                    {
                        "source": f"r/{sub}",
                        "title":  title[:120],
                        "score":  score,
                        "label":  _label(score),
                        "upvotes": post.get("score", 0),
                        "url":    f"https://reddit.com{post.get('permalink', '')}",
                    }
                )
        except Exception:
            continue
        time.sleep(0.3)   # polite crawl delay

    if not scores:
        return 0.0, []

    # Weighted average — upvoted posts carry more weight
    upvotes = [max(p["upvotes"], 1) for p in posts]
    total_up = sum(upvotes)
    weighted_score = sum(s * u for s, u in zip(scores, upvotes)) / total_up

    # Sort by absolute sentiment (most opinionated first)
    posts.sort(key=lambda p: abs(p["score"]), reverse=True)
    return float(weighted_score), posts[:20]


# ──────────────────────────────────────────────────────────────────
# Yahoo Finance News (via yfinance — no key needed)
# ──────────────────────────────────────────────────────────────────

@st.cache_data(ttl=1800, show_spinner=False)
def fetch_news_sentiment(ticker: str) -> Tuple[float, List[Dict]]:
    """
    Fetch recent news headlines for *ticker* from Yahoo Finance.

    Returns:
        (aggregate_score, list_of_headline_dicts)
    """
    try:
        news_items = yf.Ticker(ticker).news or []
    except Exception:
        return 0.0, []

    articles: List[Dict] = []
    scores: List[float] = []

    for item in news_items[:25]:
        # yfinance ≥ 1.4 wraps everything inside item["content"]
        # Older versions put keys directly on item — support both.
        content = item.get("content") or item          # dict

        title   = (
            content.get("title")
            or content.get("headline")
            or item.get("title")
            or ""
        )
        summary = (
            content.get("summary")
            or content.get("description")
            or item.get("summary")
            or ""
        )

        # Strip accidental HTML tags (safety net)
        import re as _re
        title   = _re.sub(r"<[^>]+>", "", title).strip()
        summary = _re.sub(r"<[^>]+>", "", summary).strip()

        if not title:
            continue

        combined = f"{title}. {summary}"[:512]
        score = _score_text(combined)
        scores.append(score)

        # Provider/source name
        provider = content.get("provider") or {}
        source   = (
            provider.get("displayName")
            or item.get("publisher")
            or "Yahoo Finance"
        )

        # URL
        url = (
            (content.get("canonicalUrl") or {}).get("url")
            or (content.get("clickThroughUrl") or {}).get("url")
            or item.get("link")
            or ""
        )

        articles.append(
            {
                "source": source,
                "title":  title[:120],
                "score":  score,
                "label":  _label(score),
                "url":    url,
            }
        )

    if not scores:
        return 0.0, []

    aggregate = float(sum(scores) / len(scores))
    articles.sort(key=lambda a: abs(a["score"]), reverse=True)
    return aggregate, articles


# ──────────────────────────────────────────────────────────────────
# Combined Sentiment
# ──────────────────────────────────────────────────────────────────

@st.cache_data(ttl=1800, show_spinner=False)
def get_combined_sentiment(ticker: str) -> Dict:
    """
    Merge Reddit + News sentiment for *ticker*.

    Returns a dict with:
        score         → weighted composite [-1, +1]
        label         → "Positive" | "Negative" | "Neutral"
        reddit_score  → Reddit sub-score
        news_score    → News sub-score
        reddit_posts  → list of post dicts
        news_articles → list of article dicts
        available     → bool (False if both sources failed)
    """
    reddit_score, reddit_posts     = fetch_reddit_sentiment(ticker)
    news_score,   news_articles    = fetch_news_sentiment(ticker)

    reddit_ok = len(reddit_posts) > 0
    news_ok   = len(news_articles) > 0

    if reddit_ok and news_ok:
        # News is more signal-rich for stocks; Reddit for crypto
        is_crypto = "-USD" in ticker
        w_reddit  = 0.55 if is_crypto else 0.35
        w_news    = 1 - w_reddit
        composite = reddit_score * w_reddit + news_score * w_news
    elif reddit_ok:
        composite = reddit_score
    elif news_ok:
        composite = news_score
    else:
        return {
            "score": 0.0, "label": "Neutral",
            "reddit_score": 0.0, "news_score": 0.0,
            "reddit_posts": [], "news_articles": [],
            "available": False,
        }

    return {
        "score":         float(composite),
        "label":         _label(composite),
        "reddit_score":  float(reddit_score),
        "news_score":    float(news_score),
        "reddit_posts":  reddit_posts,
        "news_articles": news_articles,
        "available":     True,
    }
