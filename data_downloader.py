"""
data_downloader.py — Market Data & News Downloader
====================================================
Standalone module for:
  1. Downloading OHLCV historical price data via yfinance
  2. Caching downloaded data to disk (CSV) so re-runs are fast
  3. Scraping recent news articles for any ticker via yfinance
  4. Sentiment scoring of news headlines (rule-based, no extra deps)

Usage (standalone):
    python data_downloader.py NVDA TSLA AMD --period 2y

Usage (as module):
    from data_downloader import MarketDataDownloader
    dl = MarketDataDownloader(cache_dir="data_cache/")
    ohlcv = dl.get_ohlcv(["NVDA", "TSLA"], period="2y")
    news  = dl.get_news(["NVDA", "TSLA"], max_per_ticker=5)
"""

import os
import sys
import json
import time
import hashlib
import warnings
from datetime import datetime, timedelta
from typing import Optional

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

try:
    import yfinance as yf
except ImportError:
    print("[ERROR] yfinance not installed. Run: pip install yfinance")
    sys.exit(1)


# ── Sentiment word lists (no extra deps needed) ───────────────────────────────

_BULLISH_WORDS = {
    "surge", "surges", "surged", "soar", "soars", "soared", "rally", "rallies",
    "rallied", "gain", "gains", "gained", "jump", "jumps", "jumped", "rise",
    "rises", "rose", "beat", "beats", "exceeded", "record", "high", "growth",
    "profit", "profits", "upgrade", "upgraded", "buy", "bullish", "strong",
    "strength", "positive", "outperform", "outperforms", "boost", "boosted",
    "revenue", "earnings", "acquisition", "deal", "partnership", "breakthrough",
    "innovation", "expansion", "dividend", "buyback", "guidance", "raised",
}

_BEARISH_WORDS = {
    "plunge", "plunges", "plunged", "fall", "falls", "fell", "drop", "drops",
    "dropped", "decline", "declines", "declined", "loss", "losses", "miss",
    "missed", "downgrade", "downgraded", "sell", "bearish", "weak", "weakness",
    "negative", "underperform", "underperforms", "concern", "concerns", "risk",
    "risks", "lawsuit", "investigation", "recall", "layoff", "layoffs", "cut",
    "cuts", "warning", "warn", "warns", "disappointing", "disappointed",
    "volatile", "volatility", "uncertainty", "debt", "default", "fraud",
}


def score_headline(text: str) -> float:
    """
    Simple rule-based sentiment scorer for a news headline.
    Returns a float in [-1.0, +1.0].
    Positive = bullish, Negative = bearish, 0 = neutral.
    """
    if not text:
        return 0.0
    words = text.lower().split()
    bull = sum(1 for w in words if w.strip(".,!?;:\"'") in _BULLISH_WORDS)
    bear = sum(1 for w in words if w.strip(".,!?;:\"'") in _BEARISH_WORDS)
    total = bull + bear
    if total == 0:
        return 0.0
    return round((bull - bear) / total, 3)


def sentiment_label(score: float) -> str:
    if score >= 0.3:
        return "BULLISH"
    if score <= -0.3:
        return "BEARISH"
    return "NEUTRAL"


# ── Cache helpers ─────────────────────────────────────────────────────────────

def _cache_path(cache_dir: str, ticker: str, period: str) -> str:
    key = f"{ticker}_{period}"
    return os.path.join(cache_dir, f"{key}.csv")


def _news_cache_path(cache_dir: str, ticker: str) -> str:
    return os.path.join(cache_dir, f"{ticker}_news.json")


def _is_cache_fresh(path: str, max_age_hours: int = 12) -> bool:
    """True if the cache file exists and was written within max_age_hours."""
    if not os.path.exists(path):
        return False
    age = time.time() - os.path.getmtime(path)
    return age < max_age_hours * 3600


# ── Main downloader class ─────────────────────────────────────────────────────

class MarketDataDownloader:
    """
    Downloads and caches market data (OHLCV) and news articles from Yahoo Finance.

    Parameters
    ----------
    cache_dir : str
        Directory to store CSV/JSON cache files.
    verbose : bool
        Print progress messages.
    """

    def __init__(self, cache_dir: str = "data_cache/", verbose: bool = True):
        self.cache_dir = cache_dir
        self.verbose = verbose
        os.makedirs(cache_dir, exist_ok=True)

    # ── OHLCV ─────────────────────────────────────────────────────────────────

    def get_ohlcv(
        self,
        tickers: list,
        period: str = "2y",
        min_rows: int = 260,
        force_refresh: bool = False,
        cache_max_age_hours: int = 12,
    ) -> dict:
        """
        Download OHLCV data for a list of tickers.
        Results are cached as CSVs; subsequent calls within cache_max_age_hours
        load from disk instead of hitting the network.

        Returns
        -------
        dict[str, pd.DataFrame]
            Keys are ticker symbols, values are clean OHLCV DataFrames with
            columns: open, high, low, close, volume, date (str YYYY-MM-DD).
        """
        result = {}
        to_download = []

        # Check cache first
        for ticker in tickers:
            path = _cache_path(self.cache_dir, ticker, period)
            if not force_refresh and _is_cache_fresh(path, cache_max_age_hours):
                df = pd.read_csv(path)
                if len(df) >= min_rows:
                    result[ticker] = df
                    if self.verbose:
                        print(f"  [CACHE] {ticker}: {len(df)} days (loaded from disk)")
                    continue
            to_download.append(ticker)

        # Download missing tickers
        if to_download:
            if self.verbose:
                print(f"\n  Downloading {len(to_download)} tickers from Yahoo Finance...")

            for ticker in to_download:
                df = self._download_single(ticker, period)
                if df is not None and len(df) >= min_rows:
                    result[ticker] = df
                    path = _cache_path(self.cache_dir, ticker, period)
                    df.to_csv(path, index=False)
                    if self.verbose:
                        print(f"  [OK]    {ticker}: {len(df)} trading days")
                else:
                    rows = len(df) if df is not None else 0
                    if self.verbose:
                        print(f"  [SKIP]  {ticker}: only {rows} rows (need {min_rows})")

        return result

    def _download_single(self, ticker: str, period: str) -> Optional[pd.DataFrame]:
        """Download one ticker from yfinance and clean the DataFrame."""
        try:
            raw = yf.download(ticker, period=period, progress=False, auto_adjust=True)
            if raw is None or len(raw) == 0:
                return None

            # Flatten multi-level columns if present
            if isinstance(raw.columns, pd.MultiIndex):
                raw.columns = raw.columns.get_level_values(0)
            raw.columns = [str(c).lower().strip() for c in raw.columns]

            col_map = {"adj close": "close"}
            raw = raw.rename(columns=col_map)

            required = ["open", "high", "low", "close", "volume"]
            if not all(c in raw.columns for c in required):
                return None

            df = raw[required].copy()
            df["date"] = pd.to_datetime(df.index).strftime("%Y-%m-%d")
            df = df.reset_index(drop=True).dropna()
            return df

        except Exception as e:
            if self.verbose:
                print(f"  [FAIL]  {ticker}: {e}")
            return None

    # ── News ──────────────────────────────────────────────────────────────────

    def get_news(
        self,
        tickers: list,
        max_per_ticker: int = 5,
        force_refresh: bool = False,
        cache_max_age_hours: int = 2,
    ) -> dict:
        """
        Fetch recent news articles for each ticker via yfinance.
        Returns sentiment-scored articles sorted by recency.

        Returns
        -------
        dict[str, list[dict]]
            Keys are ticker symbols. Each article dict has:
              title, publisher, url, published_at (ISO str), age_hours,
              sentiment_score (float -1..+1), sentiment_label (str)
        """
        result = {}

        for ticker in tickers:
            path = _news_cache_path(self.cache_dir, ticker)

            if not force_refresh and _is_cache_fresh(path, cache_max_age_hours):
                with open(path) as f:
                    result[ticker] = json.load(f)
                continue

            articles = self._fetch_news_single(ticker, max_per_ticker)
            result[ticker] = articles

            # Cache even empty results so we don't hammer Yahoo
            with open(path, "w") as f:
                json.dump(articles, f, indent=2)

        return result

    def _fetch_news_single(self, ticker: str, max_items: int) -> list:
        """
        Fetch and enrich news for one ticker.
        Handles both legacy yfinance news format (flat keys) and the new
        nested-content format introduced in yfinance ~0.2.50+.
        """
        articles = []
        try:
            t = yf.Ticker(ticker)
            raw_news = t.news or []

            now_ts = time.time()
            for item in raw_news[:max_items]:
                # ── New nested format: item = {id, content: {title, pubDate, ...}} ──
                content = item.get("content") or {}
                if content:
                    title     = content.get("title") or ""
                    publisher = (
                        (content.get("provider") or {}).get("displayName") or "Unknown"
                    )
                    url = (
                        (content.get("canonicalUrl") or {}).get("url")
                        or (content.get("clickThroughUrl") or {}).get("url")
                        or ""
                    )
                    pub_str = content.get("pubDate") or content.get("displayTime") or ""
                    try:
                        pub_dt    = datetime.strptime(pub_str, "%Y-%m-%dT%H:%M:%SZ")
                        pub_iso   = pub_dt.strftime("%Y-%m-%d %H:%M UTC")
                        pub_epoch = pub_dt.timestamp()
                        age_hours = round((now_ts - pub_epoch) / 3600, 1)
                    except Exception:
                        pub_iso   = "Unknown"
                        age_hours = 999.0

                # ── Legacy flat format: item = {title, publisher, link, providerPublishTime} ──
                else:
                    title     = item.get("title") or item.get("headline") or ""
                    publisher = item.get("publisher") or item.get("source") or "Unknown"
                    url       = item.get("link") or item.get("url") or ""
                    pub_ts_val = item.get("providerPublishTime") or item.get("published") or now_ts
                    try:
                        pub_dt    = datetime.utcfromtimestamp(float(pub_ts_val))
                        pub_iso   = pub_dt.strftime("%Y-%m-%d %H:%M UTC")
                        age_hours = round((now_ts - float(pub_ts_val)) / 3600, 1)
                    except Exception:
                        pub_iso   = "Unknown"
                        age_hours = 999.0

                if not title:
                    continue

                score = score_headline(title)
                articles.append({
                    "title":           title,
                    "publisher":       publisher,
                    "url":             url,
                    "published_at":    pub_iso,
                    "age_hours":       age_hours,
                    "sentiment_score": score,
                    "sentiment_label": sentiment_label(score),
                })

        except Exception as e:
            if self.verbose:
                print(f"  [NEWS FAIL] {ticker}: {e}")

        # Sort newest first
        articles.sort(key=lambda x: x["age_hours"])
        return articles

    # ── Summary helpers ───────────────────────────────────────────────────────

    def aggregate_news_sentiment(self, news_for_ticker: list) -> dict:
        """
        Aggregate sentiment across all articles for a ticker.

        Returns
        -------
        dict with keys: article_count, avg_sentiment, overall_label,
                        bullish_count, bearish_count, neutral_count
        """
        if not news_for_ticker:
            return {
                "article_count": 0,
                "avg_sentiment": 0.0,
                "overall_label": "NO DATA",
                "bullish_count": 0,
                "bearish_count": 0,
                "neutral_count": 0,
            }

        scores = [a["sentiment_score"] for a in news_for_ticker]
        labels = [a["sentiment_label"] for a in news_for_ticker]
        avg = round(float(np.mean(scores)), 3)

        return {
            "article_count": len(news_for_ticker),
            "avg_sentiment":  avg,
            "overall_label":  sentiment_label(avg),
            "bullish_count":  labels.count("BULLISH"),
            "bearish_count":  labels.count("BEARISH"),
            "neutral_count":  labels.count("NEUTRAL"),
        }

    def print_news_summary(self, ticker: str, articles: list, max_show: int = 5):
        """Pretty-print news + sentiment for a single ticker."""
        agg = self.aggregate_news_sentiment(articles)
        label_sym = {"BULLISH": "[+]", "BEARISH": "[-]", "NEUTRAL": "[~]",
                     "NO DATA": "[?]"}
        sym = label_sym.get(agg["overall_label"], "[?]")

        print(f"\n  {ticker} News Sentiment: {sym} {agg['overall_label']}  "
              f"(avg score: {agg['avg_sentiment']:+.2f}  |  "
              f"{agg['bullish_count']} bullish, {agg['bearish_count']} bearish, "
              f"{agg['neutral_count']} neutral  across {agg['article_count']} articles)")

        for i, art in enumerate(articles[:max_show], 1):
            sym2 = label_sym.get(art["sentiment_label"], "[?]")
            age = f"{art['age_hours']:.0f}h ago" if art["age_hours"] < 999 else "unknown age"
            title_short = art["title"][:80] + ("..." if len(art["title"]) > 80 else "")
            print(f"    {i}. {sym2} [{art['publisher']:<18}] {age:<10} {title_short}")


# ── CLI entry-point ───────────────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Download market data and news via yfinance")
    parser.add_argument("tickers", nargs="+", help="Ticker symbols (e.g. NVDA TSLA AMD)")
    parser.add_argument("--period",    default="2y", help="yfinance period string (default: 2y)")
    parser.add_argument("--cache-dir", default="data_cache/", help="Cache directory")
    parser.add_argument("--news",      action="store_true",  help="Also fetch news articles")
    parser.add_argument("--refresh",   action="store_true",  help="Force refresh (ignore cache)")
    parser.add_argument("--max-news",  type=int, default=5,  help="Max articles per ticker")
    args = parser.parse_args()

    dl = MarketDataDownloader(cache_dir=args.cache_dir, verbose=True)

    print(f"\nDownloading OHLCV data for: {', '.join(args.tickers)}")
    print(f"Period: {args.period}  |  Cache: {args.cache_dir}")
    print("-" * 60)
    ohlcv = dl.get_ohlcv(args.tickers, period=args.period, force_refresh=args.refresh)
    print(f"\n  {len(ohlcv)}/{len(args.tickers)} tickers ready.")

    if args.news:
        print(f"\nFetching news for: {', '.join(args.tickers)}")
        print("-" * 60)
        news = dl.get_news(args.tickers, max_per_ticker=args.max_news,
                           force_refresh=args.refresh)
        for ticker in args.tickers:
            if ticker in news:
                dl.print_news_summary(ticker, news[ticker], max_show=args.max_news)
            else:
                print(f"\n  {ticker}: no news available")


if __name__ == "__main__":
    main()
