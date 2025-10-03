"""
Simplified data sources for market data, news, and social media sentiment.
"""
import asyncio
import logging
import requests
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
import json
import os

from config.settings import DataSourceConfig, SystemConfig
from src.hf_client import HFClassifier

class MarketDataProvider:
    """Simplified market data provider."""
    
    def __init__(self, config: DataSourceConfig):
        self.config = config
        self.logger = logging.getLogger("MarketDataProvider")
        # Optional external provider API keys
        self.alpha_vantage_key = getattr(config, "alpha_vantage_key", None)
        self.binance_api_key = getattr(config, "binance_api_key", None)
        self.oanda_api_key = getattr(config, "oanda_api_key", None)
        # Alpaca Data API (read from env)
        self.alpaca_key = os.getenv("ALPACA_API_KEY", "").strip()
        self.alpaca_secret = os.getenv("ALPACA_SECRET_KEY", "").strip()
        self.alpaca_data_url = os.getenv("ALPACA_DATA_URL", "https://data.alpaca.markets").rstrip("/")
        self.use_alpaca_data = bool(self.alpaca_key and self.alpaca_secret)

    async def get_price_data(self, symbols: List[str], period: str = "1d", interval: str = "15m") -> Dict[str, List[Dict[str, Any]]]:
        """Get price data for symbols.

        Attempts to use real providers when API keys are available (Alpha Vantage for FOREX, Binance for crypto).
        Falls back to mock data generation for testing or when external providers are not configured.
        """
        price_data: Dict[str, List[Dict[str, Any]]] = {}

        for symbol in symbols:
            try:
                # Determine provider by symbol heuristics
                # Treat common crypto pairs ending with USD as crypto first
                if self._is_crypto_symbol(symbol) and getattr(self.config, "crypto_enabled", False):
                    if self.use_alpaca_data:
                        data = self._fetch_crypto_alpaca(symbol, interval)
                    else:
                        # Fallback to Binance if possible
                        data = self._fetch_crypto_binance(symbol, period, interval)
                elif self._is_forex_symbol(symbol) and getattr(self.config, "forex_enabled", False) and self.alpha_vantage_key:
                    data = self._fetch_forex_alpha_vantage(symbol, period, interval)
                elif self.use_alpaca_data:
                    data = self._fetch_stock_alpaca(symbol, interval)
                else:
                    data = self._generate_mock_price_data(symbol)
            except Exception as e:
                self.logger.warning(f"Failed to fetch real data for {symbol}: {e}. Using mock data.")
                data = self._generate_mock_price_data(symbol)

            price_data[symbol] = data

        return price_data

    def _is_forex_symbol(self, symbol: str) -> bool:
        return isinstance(symbol, str) and len(symbol) == 6 and symbol.isalpha()

    def _is_crypto_symbol(self, symbol: str) -> bool:
        return isinstance(symbol, str) and symbol.upper().endswith("USD") and len(symbol) >= 6

    def _alpaca_headers(self) -> Dict[str, str]:
        return {
            "APCA-API-KEY-ID": self.alpaca_key,
            "APCA-API-SECRET-KEY": self.alpaca_secret,
            "Content-Type": "application/json",
        }

    def _normalize_timeframe(self, interval: str) -> str:
        if not isinstance(interval, str):
            return "1Min"
        key = interval.strip().lower()
        mapping = {
            "1m": "1Min",
            "1min": "1Min",
            "5m": "5Min",
            "5min": "5Min",
            "15m": "15Min",
            "15min": "15Min",
            "1h": "1Hour",
            "1hour": "1Hour",
            "1d": "1Day",
            "1day": "1Day",
        }
        return mapping.get(key, "1Min")

    def _crypto_symbol_for_alpaca(self, symbol: str) -> str:
        s = (symbol or "").upper()
        if len(s) >= 6 and s.endswith("USD"):
            return f"{s[:-3]}/USD"
        return s

    def _fetch_stock_alpaca(self, symbol: str, interval: str = "1Min", limit: int = 50) -> List[Dict[str, Any]]:
        """Fetch recent stock bars from Alpaca Data API v2."""
        try:
            url = f"{self.alpaca_data_url}/v2/stocks/{symbol}/bars"
            params = {"timeframe": self._normalize_timeframe(interval), "limit": limit}
            r = requests.get(url, headers=self._alpaca_headers(), params=params, timeout=15)
            r.raise_for_status()
            payload = r.json() or {}
            bars = payload.get("bars") or []
            if not bars:
                # Try 1Hour timeframe if 1Min fails
                if interval == "1Min":
                    return self._fetch_stock_alpaca(symbol, "1Hour", limit)
                return self._generate_mock_price_data(symbol)
            
            data: List[Dict[str, Any]] = []
            for b in bars:
                data.append({
                    "timestamp": b.get("t"),
                    "open": float(b.get("o", 0.0) or 0.0),
                    "high": float(b.get("h", 0.0) or 0.0),
                    "low": float(b.get("l", 0.0) or 0.0),
                    "close": float(b.get("c", 0.0) or 0.0),
                    "volume": int(b.get("v", 0) or 0),
                })
            return data if data else self._generate_mock_price_data(symbol)
        except Exception as e:
            self.logger.warning(f"Alpaca stock bars failed for {symbol}: {e}")
            return self._generate_mock_price_data(symbol)

    def _fetch_crypto_alpaca(self, symbol: str, interval: str = "1Min", limit: int = 50) -> List[Dict[str, Any]]:
        """Fetch recent crypto bars via Alpaca Data API (try v2 then v1beta3)."""
        # Try v2 endpoint
        try:
            api_symbol = self._crypto_symbol_for_alpaca(symbol)
            url = f"{self.alpaca_data_url}/v2/crypto/{api_symbol}/bars"
            params = {"timeframe": self._normalize_timeframe(interval), "limit": limit}
            r = requests.get(url, headers=self._alpaca_headers(), params=params, timeout=15)
            if r.status_code == 200:
                payload = r.json() or {}
                bars = payload.get("bars") or []
                if not bars:
                    # Try 1Hour timeframe if 1Min fails
                    if interval == "1Min":
                        return self._fetch_crypto_alpaca(symbol, "1Hour", limit)
                    return self._generate_mock_price_data(symbol)
                
                data: List[Dict[str, Any]] = []
                for b in bars:
                    data.append({
                        "timestamp": b.get("t"),
                        "open": float(b.get("o", 0.0) or 0.0),
                        "high": float(b.get("h", 0.0) or 0.0),
                        "low": float(b.get("l", 0.0) or 0.0),
                        "close": float(b.get("c", 0.0) or 0.0),
                        "volume": int(b.get("v", 0) or 0),
                    })
                if data:
                    return data
        except Exception as e:
            self.logger.debug(f"Alpaca v2 crypto bars failed for {symbol}: {e}")
        # Fallback v1beta3 multi-symbol
        try:
            api_symbol = self._crypto_symbol_for_alpaca(symbol)
            url = f"{self.alpaca_data_url}/v1beta3/crypto/us/bars"
            params = {"symbols": api_symbol, "timeframe": self._normalize_timeframe(interval), "limit": limit}
            r = requests.get(url, headers=self._alpaca_headers(), params=params, timeout=15)
            r.raise_for_status()
            payload = r.json() or {}
            bars = payload.get("bars") or {}
            series = bars.get(api_symbol) if isinstance(bars, dict) else None
            data: List[Dict[str, Any]] = []
            if isinstance(series, list):
                for b in series:
                    data.append({
                        "timestamp": b.get("t"),
                        "open": float(b.get("o", 0.0) or 0.0),
                        "high": float(b.get("h", 0.0) or 0.0),
                        "low": float(b.get("l", 0.0) or 0.0),
                        "close": float(b.get("c", 0.0) or 0.0),
                        "volume": int(b.get("v", 0) or 0),
                    })
            return data if data else self._generate_mock_price_data(symbol)
        except Exception as e:
            self.logger.warning(f"Alpaca crypto bars failed for {symbol}: {e}")
            return self._generate_mock_price_data(symbol)

    def _generate_mock_price_data(self, symbol: str, points: int = 100) -> List[Dict[str, Any]]:
        base_price = 100.0 + abs(hash(symbol)) % 100
        data: List[Dict[str, Any]] = []
        for i in range(points):
            price_change = np.random.normal(0, 0.02)
            base_price *= (1 + price_change)
            # Generate more realistic timestamps
            timestamp = datetime.now() - timedelta(minutes=points-i)
            data.append({
                "timestamp": timestamp.isoformat(),
                "open": base_price * 0.999,
                "high": base_price * 1.001,
                "low": base_price * 0.998,
                "close": base_price,
                "volume": int(np.random.uniform(1000, 1000000))
            })
        return data

    def _fetch_forex_alpha_vantage(self, symbol: str, period: str, interval: str) -> List[Dict[str, Any]]:
        """Fetch FOREX data from Alpha Vantage using FX_INTRADAY.

        This is a synchronous HTTP call; production code should handle rate limits and retries.
        """
        base, quote = symbol[:3].upper(), symbol[3:].upper()
        url = "https://www.alphavantage.co/query"
        params = {
            "function": "FX_INTRADAY",
            "from_symbol": base,
            "to_symbol": quote,
            "interval": "60min",
            "apikey": self.alpha_vantage_key,
            "outputsize": "compact"
        }
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        payload = resp.json()
        ts_key = next((k for k in payload.keys() if k.startswith("Time Series")), None)
        if not ts_key:
            raise ValueError("Alpha Vantage returned no time series for forex")

        series = payload[ts_key]
        items = list(series.items())
        data: List[Dict[str, Any]] = []
        # oldest first
        for t, vals in items[::-1][:50]:
            data.append({
                "timestamp": t,
                "open": float(vals.get("1. open", 0.0)),
                "high": float(vals.get("2. high", 0.0)),
                "low": float(vals.get("3. low", 0.0)),
                "close": float(vals.get("4. close", 0.0)),
                "volume": int(float(vals.get("5. volume", 0)))
            })
        return data

    def _fetch_crypto_binance(self, symbol: str, period: str, interval: str) -> List[Dict[str, Any]]:
        """Fetch recent candlestick data from Binance public API.

        Maps XXXUSD -> XXXUSDT for Binance market data.
        """
        api_symbol = symbol.upper()
        if api_symbol.endswith("USD"):
            api_symbol = api_symbol[:-3] + "USDT"

        url = "https://api.binance.com/api/v3/klines"
        params = {"symbol": api_symbol, "interval": "1m", "limit": 50}
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        klines = resp.json()
        data: List[Dict[str, Any]] = []
        for k in klines:
            data.append({
                "timestamp": datetime.fromtimestamp(k[0] / 1000).isoformat(),
                "open": float(k[1]),
                "high": float(k[2]),
                "low": float(k[3]),
                "close": float(k[4]),
                "volume": int(float(k[5]))
            })
        return data
    
    async def get_technical_indicators(self, price_data: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Dict[str, Any]]:
        """Calculate technical indicators for price data."""
        indicators = {}
        
        for symbol, data in price_data.items():
            if len(data) < 20:
                continue
            
            df = pd.DataFrame(data)
            df['close'] = pd.to_numeric(df['close'])
            
            symbol_indicators = {}
            
            # Moving averages
            symbol_indicators['sma_20'] = df['close'].rolling(window=20).mean().iloc[-1]
            symbol_indicators['sma_50'] = df['close'].rolling(window=50).mean().iloc[-1] if len(df) >= 50 else None
            
            # RSI
            symbol_indicators['rsi'] = self._calculate_rsi(df['close'])
            
            # MACD
            macd_line, signal_line, histogram = self._calculate_macd(df['close'])
            symbol_indicators['macd'] = macd_line
            symbol_indicators['macd_signal'] = signal_line
            symbol_indicators['macd_histogram'] = histogram
            
            # Volume indicators
            symbol_indicators['volume_sma'] = df['volume'].rolling(window=20).mean().iloc[-1]
            symbol_indicators['volume_ratio'] = df['volume'].iloc[-1] / symbol_indicators['volume_sma']
            
            indicators[symbol] = symbol_indicators
        
        return indicators
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> float:
        """Calculate RSI indicator."""
        if len(prices) < period + 1:
            return 50.0
        
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi.iloc[-1] if not pd.isna(rsi.iloc[-1]) else 50.0
    
    def _calculate_macd(self, prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[float, float, float]:
        """Calculate MACD indicator."""
        if len(prices) < slow + signal:
            return 0.0, 0.0, 0.0
         
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal).mean()
        histogram = macd_line - signal_line
         
        return macd_line.iloc[-1], signal_line.iloc[-1], histogram.iloc[-1]


class NewsProvider:
    """Simplified news provider."""
    
    def __init__(self, config: DataSourceConfig):
        self.config = config
        self.logger = logging.getLogger("NewsProvider")
        # Optional Hugging Face classifier
        self.hf = HFClassifier()
        
    async def get_financial_news(self, symbols: List[str], hours_back: int = 24) -> Dict[str, List[Dict[str, Any]]]:
        """Get mock financial news for testing."""
        news_data = {}
        
        for symbol in symbols:
            # Generate mock news
            mock_news = [
                {
                    "title": f"{symbol} shows strong quarterly performance",
                    "content": f"Company {symbol} reported better than expected earnings... ",
                    "url": f"https://example.com/news/{symbol.lower()}-earnings",
                    "source": "Financial Times",
                    "timestamp": datetime.now().isoformat(),
                    "sentiment": {"vader": 0.2, "textblob": 0.3, "combined": 0.25}
                },
                {
                    "title": f"Market analysts bullish on {symbol}",
                    "content": f"Several analysts have upgraded their ratings for {symbol}... ",
                    "url": f"https://example.com/news/{symbol.lower()}-upgrade",
                    "source": "Reuters",
                    "timestamp": (datetime.now() - timedelta(hours=2)).isoformat(),
                    "sentiment": {"vader": 0.4, "textblob": 0.5, "combined": 0.45}
                }
            ]

            # Enrich with HF if available
            if getattr(self.hf, "enabled", False):
                texts = [f"{a.get('title','')} {a.get('content','')}".strip() for a in mock_news]
                try:
                    fin_preds = self.hf.classify_financial(texts)
                except Exception:
                    fin_preds = [{} for _ in texts]
                try:
                    summaries = self.hf.summarize(texts)
                except Exception:
                    summaries = [""] * len(texts)
                for i, article in enumerate(mock_news):
                    if isinstance(fin_preds[i], dict) and fin_preds[i]:
                        article["hf_sentiment"] = fin_preds[i]
                    if summaries[i]:
                        article["summary"] = summaries[i]
            
            news_data[symbol] = mock_news
        
        return news_data
    
    async def get_market_sentiment(self, news_data: Dict[str, List[Dict[str, Any]]]) -> Dict[str, float]:
        """Calculate overall market sentiment for each symbol."""
        sentiment_scores = {}
        
        for symbol, articles in news_data.items():
            if not articles:
                sentiment_scores[symbol] = 0.0
                continue
            
            # Calculate weighted sentiment
            total_sentiment = 0
            total_weight = 0
            
            for article in articles:
                sentiment = article.get('sentiment', {})
                combined_sentiment = sentiment.get('combined', 0.0)
                # If HF enriched, blend signed score
                hf = article.get('hf_sentiment')
                if isinstance(hf, dict):
                    combined_sentiment = (combined_sentiment + float(hf.get('score_signed', 0.0))) / 2.0
                
                # Weight by recency (more recent news has higher weight)
                weight = 1.0  # Could implement time-based weighting here
                
                total_sentiment += combined_sentiment * weight
                total_weight += weight
            
            sentiment_scores[symbol] = total_sentiment / total_weight if total_weight > 0 else 0.0
        
        return sentiment_scores


class SocialMediaProvider:
    """Simplified social media provider."""
    
    def __init__(self, config: DataSourceConfig):
        self.config = config
        self.logger = logging.getLogger("SocialMediaProvider")
        self.hf = HFClassifier()
        
    async def get_social_sentiment(self, symbols: List[str], hours_back: int = 24) -> Dict[str, Dict[str, float]]:
        """Get mock social media sentiment for testing."""
        social_sentiment = {}
        
        for symbol in symbols:
            # Generate mock social sentiment
            symbol_sentiment = {
                "twitter": np.random.normal(0, 0.3),  # Random sentiment between -1 and 1
                "reddit": np.random.normal(0, 0.2),
                "combined": 0.0
            }

            # Optional HF-based social signal using a simple prompt
            if getattr(self.hf, "enabled", False):
                try:
                    preds = self.hf.classify_social([f"{symbol} community sentiment today"]) or [{}]
                    hf_signed = float(preds[0].get("score_signed", 0.0)) if isinstance(preds[0], dict) else 0.0
                    symbol_sentiment["hf_social"] = hf_signed
                except Exception:
                    pass
            
            # Calculate combined sentiment
            sentiments = [v for k, v in symbol_sentiment.items() if isinstance(v, (int, float))]
            if sentiments:
                symbol_sentiment["combined"] = sum(sentiments) / len(sentiments)
            
            social_sentiment[symbol] = symbol_sentiment
        
        return social_sentiment


class YouTubeTranscriptProvider:
    """Fetch YouTube transcripts via an MCP server and enrich via HF."""

    def __init__(self, config: DataSourceConfig):
        self.config = config
        self.logger = logging.getLogger("YouTubeTranscriptProvider")
        self.base_url = os.getenv("YOUTUBE_MCP_URL", "").rstrip("/")
        ids_raw = os.getenv("YOUTUBE_VIDEO_IDS", "").strip()
        self.video_ids: List[str] = [v.strip() for v in ids_raw.split(",") if v.strip()]
        self.hf = HFClassifier()

    def _fetch_transcript(self, video_id: str) -> str:
        if not self.base_url:
            return ""
        # Try common endpoint styles on MCP server
        endpoints = [f"{self.base_url}/transcript", f"{self.base_url}/transcripts"]
        for ep in endpoints:
            try:
                resp = requests.get(ep, params={"video_id": video_id}, timeout=15)
                if resp.status_code == 200:
                    try:
                        payload = resp.json()
                        txt = payload.get("transcript") or payload.get("text") or ""
                        if txt:
                            return txt
                    except Exception:
                        if resp.text:
                            return resp.text
                # POST fallback
                resp2 = requests.post(ep, json={"video_id": video_id}, timeout=15)
                if resp2.status_code == 200:
                    payload = resp2.json()
                    txt = payload.get("transcript") or payload.get("text") or ""
                    if txt:
                        return txt
            except Exception:
                continue
        return ""

    async def get_transcript_insights(self) -> Dict[str, Dict[str, Any]]:
        results: Dict[str, Dict[str, Any]] = {}
        if not self.base_url or not self.video_ids:
            return results
        texts: List[str] = []
        ids: List[str] = []
        for vid in self.video_ids:
            txt = self._fetch_transcript(vid)
            if txt:
                texts.append(txt)
                ids.append(vid)
        if not texts:
            return results
        summaries = self.hf.summarize(texts) if getattr(self.hf, "enabled", False) else [""] * len(texts)
        fin_sent = self.hf.classify_financial(texts) if getattr(self.hf, "enabled", False) else [{} for _ in texts]
        for i, vid in enumerate(ids):
            results[vid] = {
                "transcript_excerpt": " ".join(texts[i].split()[:120]),
                "summary": summaries[i] if i < len(summaries) else "",
                "hf_sentiment": fin_sent[i] if i < len(fin_sent) else {},
            }
        return results


class DataAggregator:
    """Simplified data aggregator."""
    
    def __init__(self, market_provider: MarketDataProvider, news_provider: NewsProvider, social_provider: SocialMediaProvider):
        self.market_provider = market_provider
        self.news_provider = news_provider
        self.social_provider = social_provider
        self.logger = logging.getLogger("DataAggregator")
        # Optional YouTube provider
        try:
            self.youtube_provider = YouTubeTranscriptProvider(market_provider.config)
        except Exception:
            self.youtube_provider = None
        
    async def get_comprehensive_data(self, symbols: List[str]) -> Dict[str, Any]:
        """Get comprehensive data from all sources."""
        try:
            # Get data from all sources in parallel
            market_data_task = self.market_provider.get_price_data(symbols)
            news_data_task = self.news_provider.get_financial_news(symbols)
            social_sentiment_task = self.social_provider.get_social_sentiment(symbols)
            
            market_data, news_data, social_sentiment = await asyncio.gather(
                market_data_task,
                news_data_task,
                social_sentiment_task
            )
            
            # Calculate technical indicators
            technical_indicators = await self.market_provider.get_technical_indicators(market_data)
            
            # Calculate news sentiment
            news_sentiment = await self.news_provider.get_market_sentiment(news_data)

            # Optional YouTube insights
            youtube_insights: Dict[str, Any] = {}
            if self.youtube_provider and self.youtube_provider.base_url and self.youtube_provider.video_ids:
                try:
                    youtube_insights = await self.youtube_provider.get_transcript_insights()
                except Exception:
                    youtube_insights = {}
            
            # Derive closes array for convenience
            prices_series: Dict[str, List[float]] = {
                sym: [float(row.get("close", 0.0)) for row in rows] for sym, rows in market_data.items() if rows
            }
            
            # Aggregate all data
            comprehensive_data = {
                "timestamp": datetime.now().isoformat(),
                "symbols": symbols,
                "price_data": market_data,
                "prices": prices_series,
                "technical_indicators": technical_indicators,
                "news_data": news_data,
                "news_sentiment": news_sentiment,
                "social_sentiment": social_sentiment,
                "youtube_insights": youtube_insights,
                "market_overview": self._generate_market_overview(market_data, technical_indicators, news_sentiment, social_sentiment)
            }
            
            return comprehensive_data
            
        except Exception as e:
            self.logger.error(f"Error aggregating data: {e}")
            return {}
    
    def _generate_market_overview(self, market_data: Dict[str, List[Dict[str, Any]]], 
                                technical_indicators: Dict[str, Dict[str, Any]], 
                                news_sentiment: Dict[str, float], 
                                social_sentiment: Dict[str, Dict[str, float]]) -> Dict[str, Any]:
        """Generate overall market overview."""
        overview = {
            "market_trend": "neutral",
            "volatility_level": "medium",
            "sentiment_score": 0.0,
            "key_events": [],
            "risk_level": "medium"
        }
        
        if not market_data:
            return overview
        
        # Calculate overall market trend
        trend_scores = []
        for symbol, data in market_data.items():
            if len(data) >= 2:
                current_price = data[-1]["close"]
                previous_price = data[-2]["close"]
                trend = (current_price - previous_price) / previous_price
                trend_scores.append(trend)
        
        if trend_scores:
            avg_trend = np.mean(trend_scores)
            if avg_trend > 0.01:
                overview["market_trend"] = "bullish"
            elif avg_trend < -0.01:
                overview["market_trend"] = "bearish"
        
        # Calculate overall sentiment
        sentiment_scores = []
        sentiment_scores.extend(news_sentiment.values())
        for symbol_sentiment in social_sentiment.values():
            sentiment_scores.append(symbol_sentiment.get("combined", 0.0))
        
        if sentiment_scores:
            overview["sentiment_score"] = np.mean(sentiment_scores)
        
        # Calculate volatility
        volatility_scores = []
        for symbol, indicators in technical_indicators.items():
            if "rsi" in indicators:
                # Use RSI as a proxy for volatility
                rsi = indicators["rsi"]
                volatility = abs(rsi - 50) / 50  # Normalize RSI to 0-1 volatility
                volatility_scores.append(volatility)
        
        if volatility_scores:
            avg_volatility = np.mean(volatility_scores)
            if avg_volatility < 0.2:
                overview["volatility_level"] = "low"
            elif avg_volatility > 0.4:
                overview["volatility_level"] = "high"
        
        # Determine risk level
        if overview["volatility_level"] == "high" or abs(overview["sentiment_score"]) > 0.5:
            overview["risk_level"] = "high"
        elif overview["volatility_level"] == "low" and abs(overview["sentiment_score"]) < 0.2:
            overview["risk_level"] = "low"
        
        return overview
