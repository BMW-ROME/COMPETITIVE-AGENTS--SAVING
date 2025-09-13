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

from config.settings import DataSourceConfig, SystemConfig

class MarketDataProvider:
    """Simplified market data provider."""
    
    def __init__(self, config: DataSourceConfig):
        self.config = config
        self.logger = logging.getLogger("MarketDataProvider")
        
    async def get_price_data(self, symbols: List[str], period: str = "1d", interval: str = "1m") -> Dict[str, List[Dict[str, Any]]]:
        """Get mock price data for testing."""
        price_data = {}
        
        for symbol in symbols:
            # Generate mock price data
            base_price = 100.0 + hash(symbol) % 100  # Different base price per symbol
            data = []
            
            for i in range(50):  # 50 data points
                # Simulate price movement
                price_change = np.random.normal(0, 0.02)  # 2% volatility
                base_price *= (1 + price_change)
                
                data.append({
                    "timestamp": (datetime.now() - timedelta(minutes=50-i)).isoformat(),
                    "open": base_price * 0.999,
                    "high": base_price * 1.001,
                    "low": base_price * 0.998,
                    "close": base_price,
                    "volume": int(np.random.uniform(100000, 1000000))
                })
            
            price_data[symbol] = data
        
        return price_data
    
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
        
    async def get_financial_news(self, symbols: List[str], hours_back: int = 24) -> Dict[str, List[Dict[str, Any]]]:
        """Get mock financial news for testing."""
        news_data = {}
        
        for symbol in symbols:
            # Generate mock news
            mock_news = [
                {
                    "title": f"{symbol} shows strong quarterly performance",
                    "content": f"Company {symbol} reported better than expected earnings...",
                    "url": f"https://example.com/news/{symbol.lower()}-earnings",
                    "source": "Financial Times",
                    "timestamp": datetime.now().isoformat(),
                    "sentiment": {"vader": 0.2, "textblob": 0.3, "combined": 0.25}
                },
                {
                    "title": f"Market analysts bullish on {symbol}",
                    "content": f"Several analysts have upgraded their ratings for {symbol}...",
                    "url": f"https://example.com/news/{symbol.lower()}-upgrade",
                    "source": "Reuters",
                    "timestamp": (datetime.now() - timedelta(hours=2)).isoformat(),
                    "sentiment": {"vader": 0.4, "textblob": 0.5, "combined": 0.45}
                }
            ]
            
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
            
            # Calculate combined sentiment
            sentiments = [v for v in symbol_sentiment.values() if v != 0.0]
            if sentiments:
                symbol_sentiment["combined"] = sum(sentiments) / len(sentiments)
            
            social_sentiment[symbol] = symbol_sentiment
        
        return social_sentiment


class DataAggregator:
    """Simplified data aggregator."""
    
    def __init__(self, market_provider: MarketDataProvider, news_provider: NewsProvider, social_provider: SocialMediaProvider):
        self.market_provider = market_provider
        self.news_provider = news_provider
        self.social_provider = social_provider
        self.logger = logging.getLogger("DataAggregator")
        
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
            
            # Aggregate all data
            comprehensive_data = {
                "timestamp": datetime.now().isoformat(),
                "symbols": symbols,
                "price_data": market_data,
                "technical_indicators": technical_indicators,
                "news_data": news_data,
                "news_sentiment": news_sentiment,
                "social_sentiment": social_sentiment,
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
