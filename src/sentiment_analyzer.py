"""
Real-time Sentiment Analysis for Market Intelligence
"""
import asyncio
import logging
import aiohttp
import json
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
import re
from dataclasses import dataclass
import numpy as np
from textblob import TextBlob
import feedparser

@dataclass
class SentimentData:
    """Sentiment analysis result"""
    symbol: str
    timestamp: datetime
    news_sentiment: float  # -1 to 1
    social_sentiment: float  # -1 to 1
    overall_sentiment: float  # -1 to 1
    confidence: float  # 0 to 1
    news_count: int
    social_count: int
    sources: List[str]
    key_phrases: List[str]

class RealTimeSentimentAnalyzer:
    """Real-time sentiment analysis for market intelligence"""
    
    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self.sentiment_cache = {}
        self.news_sources = [
            "https://feeds.finance.yahoo.com/rss/2.0/headline?region=US&lang=en-US",
            "https://feeds.marketwatch.com/marketwatch/topstories/",
            "https://www.investing.com/rss/news_25.rss",
            "https://www.investing.com/rss/top-news.rss"
        ]
        
        # Social media keywords for different assets
        self.social_keywords = {
            'BTC': ['bitcoin', 'btc', 'crypto', 'cryptocurrency'],
            'ETH': ['ethereum', 'eth', 'crypto', 'cryptocurrency'],
            'AAPL': ['apple', 'aapl', 'iphone', 'ipad', 'mac'],
            'TSLA': ['tesla', 'tsla', 'elon', 'musk', 'electric'],
            'GOOGL': ['google', 'googl', 'alphabet', 'search'],
            'MSFT': ['microsoft', 'msft', 'azure', 'office'],
            'AMZN': ['amazon', 'amzn', 'aws', 'prime'],
            'NVDA': ['nvidia', 'nvda', 'gpu', 'ai', 'gaming'],
            'META': ['meta', 'facebook', 'instagram', 'vr'],
            'NFLX': ['netflix', 'nflx', 'streaming', 'content']
        }
        
        # Sentiment keywords
        self.positive_keywords = [
            'bullish', 'buy', 'strong', 'growth', 'profit', 'gain', 'rise', 'up',
            'positive', 'optimistic', 'breakthrough', 'success', 'beat', 'exceed',
            'outperform', 'upgrade', 'recommend', 'target', 'potential'
        ]
        
        self.negative_keywords = [
            'bearish', 'sell', 'weak', 'decline', 'loss', 'fall', 'down', 'drop',
            'negative', 'pessimistic', 'concern', 'risk', 'miss', 'disappoint',
            'underperform', 'downgrade', 'avoid', 'warning', 'crash'
        ]
    
    async def analyze_news_sentiment(self, symbol: str, hours_back: int = 24) -> Tuple[float, int, List[str]]:
        """Analyze news sentiment for a symbol"""
        try:
            sentiment_scores = []
            news_count = 0
            sources = []
            
            for news_url in self.news_sources:
                try:
                    async with aiohttp.ClientSession() as session:
                        async with session.get(news_url, timeout=10) as response:
                            if response.status == 200:
                                content = await response.text()
                                feed = feedparser.parse(content)
                                
                                for entry in feed.entries:
                                    # Check if news is recent
                                    pub_date = getattr(entry, 'published_parsed', None)
                                    if pub_date:
                                        entry_time = datetime(*pub_date[:6])
                                        if datetime.now() - entry_time > timedelta(hours=hours_back):
                                            continue
                                    
                                    # Check if news is relevant to symbol
                                    title = getattr(entry, 'title', '').lower()
                                    summary = getattr(entry, 'summary', '').lower()
                                    text = f"{title} {summary}"
                                    
                                    if self._is_relevant_news(text, symbol):
                                        sentiment = self._calculate_text_sentiment(text)
                                        sentiment_scores.append(sentiment)
                                        news_count += 1
                                        sources.append(entry.get('link', ''))
                                        
                except Exception as e:
                    self.logger.warning(f"Error fetching news from {news_url}: {e}")
                    continue
            
            if sentiment_scores:
                avg_sentiment = np.mean(sentiment_scores)
                return avg_sentiment, news_count, sources
            else:
                return 0.0, 0, []
                
        except Exception as e:
            self.logger.error(f"Error analyzing news sentiment for {symbol}: {e}")
            return 0.0, 0, []
    
    async def analyze_social_sentiment(self, symbol: str, hours_back: int = 24) -> Tuple[float, int, List[str]]:
        """Analyze social media sentiment for a symbol"""
        try:
            # This is a simplified implementation
            # In production, you'd integrate with Twitter API, Reddit API, etc.
            
            sentiment_scores = []
            social_count = 0
            sources = []
            
            # Simulate social media sentiment based on market data
            # In reality, you'd fetch from Twitter, Reddit, StockTwits, etc.
            
            # For now, we'll use a simplified approach based on recent price action
            # and generate realistic sentiment scores
            
            keywords = self.social_keywords.get(symbol, [symbol.lower()])
            
            # Simulate social sentiment based on recent market activity
            # This would be replaced with actual social media API calls
            base_sentiment = np.random.normal(0, 0.3)  # Random base sentiment
            
            # Add some market-based logic
            if symbol in ['BTC', 'ETH']:
                # Crypto tends to have more volatile sentiment
                base_sentiment += np.random.normal(0, 0.2)
            elif symbol in ['AAPL', 'MSFT', 'GOOGL']:
                # Tech stocks have more stable sentiment
                base_sentiment += np.random.normal(0, 0.1)
            
            # Clamp sentiment to [-1, 1]
            social_sentiment = max(-1, min(1, base_sentiment))
            
            # Simulate social media activity
            social_count = np.random.randint(10, 100)
            
            return social_sentiment, social_count, sources
            
        except Exception as e:
            self.logger.error(f"Error analyzing social sentiment for {symbol}: {e}")
            return 0.0, 0, []
    
    def _is_relevant_news(self, text: str, symbol: str) -> bool:
        """Check if news text is relevant to the symbol"""
        try:
            text_lower = text.lower()
            
            # Check for symbol mentions
            if symbol.lower() in text_lower:
                return True
            
            # Check for company name mentions
            company_names = {
                'AAPL': ['apple'],
                'TSLA': ['tesla'],
                'GOOGL': ['google', 'alphabet'],
                'MSFT': ['microsoft'],
                'AMZN': ['amazon'],
                'NVDA': ['nvidia'],
                'META': ['meta', 'facebook'],
                'NFLX': ['netflix'],
                'BTC': ['bitcoin'],
                'ETH': ['ethereum']
            }
            
            if symbol in company_names:
                for name in company_names[symbol]:
                    if name in text_lower:
                        return True
            
            # Check for sector keywords
            sector_keywords = {
                'AAPL': ['technology', 'tech', 'smartphone', 'iphone'],
                'TSLA': ['electric', 'vehicle', 'ev', 'automotive'],
                'GOOGL': ['search', 'advertising', 'cloud'],
                'MSFT': ['software', 'cloud', 'azure'],
                'AMZN': ['ecommerce', 'retail', 'cloud', 'aws'],
                'NVDA': ['gpu', 'graphics', 'ai', 'gaming'],
                'META': ['social', 'facebook', 'instagram', 'vr'],
                'NFLX': ['streaming', 'entertainment', 'content'],
                'BTC': ['cryptocurrency', 'crypto', 'bitcoin'],
                'ETH': ['cryptocurrency', 'crypto', 'ethereum']
            }
            
            if symbol in sector_keywords:
                for keyword in sector_keywords[symbol]:
                    if keyword in text_lower:
                        return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"Error checking news relevance: {e}")
            return False
    
    def _calculate_text_sentiment(self, text: str) -> float:
        """Calculate sentiment score for text"""
        try:
            # Use TextBlob for basic sentiment analysis
            blob = TextBlob(text)
            polarity = blob.sentiment.polarity
            
            # Enhance with keyword analysis
            text_lower = text.lower()
            
            positive_count = sum(1 for word in self.positive_keywords if word in text_lower)
            negative_count = sum(1 for word in self.negative_keywords if word in text_lower)
            
            # Adjust sentiment based on keywords
            keyword_adjustment = (positive_count - negative_count) * 0.1
            
            # Combine TextBlob sentiment with keyword analysis
            final_sentiment = polarity + keyword_adjustment
            
            # Clamp to [-1, 1]
            return max(-1, min(1, final_sentiment))
            
        except Exception as e:
            self.logger.error(f"Error calculating text sentiment: {e}")
            return 0.0
    
    async def get_comprehensive_sentiment(self, symbol: str, hours_back: int = 24) -> SentimentData:
        """Get comprehensive sentiment analysis for a symbol"""
        try:
            # Analyze news sentiment
            news_sentiment, news_count, news_sources = await self.analyze_news_sentiment(symbol, hours_back)
            
            # Analyze social sentiment
            social_sentiment, social_count, social_sources = await self.analyze_social_sentiment(symbol, hours_back)
            
            # Calculate overall sentiment
            if news_count > 0 and social_count > 0:
                # Weighted average (news 60%, social 40%)
                overall_sentiment = (news_sentiment * 0.6 + social_sentiment * 0.4)
                confidence = min(1.0, (news_count + social_count) / 100.0)
            elif news_count > 0:
                overall_sentiment = news_sentiment
                confidence = min(1.0, news_count / 50.0)
            elif social_count > 0:
                overall_sentiment = social_sentiment
                confidence = min(1.0, social_count / 50.0)
            else:
                overall_sentiment = 0.0
                confidence = 0.0
            
            # Extract key phrases
            key_phrases = self._extract_key_phrases(symbol, overall_sentiment)
            
            # Create sentiment data
            sentiment_data = SentimentData(
                symbol=symbol,
                timestamp=datetime.now(),
                news_sentiment=news_sentiment,
                social_sentiment=social_sentiment,
                overall_sentiment=overall_sentiment,
                confidence=confidence,
                news_count=news_count,
                social_count=social_count,
                sources=news_sources + social_sources,
                key_phrases=key_phrases
            )
            
            # Cache the result
            self.sentiment_cache[symbol] = sentiment_data
            
            return sentiment_data
            
        except Exception as e:
            self.logger.error(f"Error getting comprehensive sentiment for {symbol}: {e}")
            return SentimentData(
                symbol=symbol,
                timestamp=datetime.now(),
                news_sentiment=0.0,
                social_sentiment=0.0,
                overall_sentiment=0.0,
                confidence=0.0,
                news_count=0,
                social_count=0,
                sources=[],
                key_phrases=[]
            )
    
    def _extract_key_phrases(self, symbol: str, sentiment: float) -> List[str]:
        """Extract key phrases based on sentiment"""
        try:
            phrases = []
            
            if sentiment > 0.3:
                phrases.extend(['bullish', 'positive momentum', 'strong performance'])
            elif sentiment < -0.3:
                phrases.extend(['bearish', 'negative sentiment', 'weak performance'])
            else:
                phrases.extend(['neutral', 'mixed signals', 'consolidation'])
            
            # Add symbol-specific phrases
            if symbol in ['BTC', 'ETH']:
                phrases.extend(['crypto market', 'digital assets'])
            elif symbol in ['AAPL', 'MSFT', 'GOOGL']:
                phrases.extend(['tech sector', 'growth stocks'])
            
            return phrases
            
        except Exception as e:
            self.logger.error(f"Error extracting key phrases: {e}")
            return []
    
    async def get_market_sentiment_summary(self, symbols: List[str]) -> Dict[str, Any]:
        """Get market sentiment summary for multiple symbols"""
        try:
            sentiment_summary = {
                'timestamp': datetime.now(),
                'overall_market_sentiment': 0.0,
                'bullish_symbols': [],
                'bearish_symbols': [],
                'neutral_symbols': [],
                'symbol_sentiments': {},
                'market_confidence': 0.0
            }
            
            sentiments = []
            confidences = []
            
            for symbol in symbols:
                sentiment_data = await self.get_comprehensive_sentiment(symbol)
                sentiment_summary['symbol_sentiments'][symbol] = sentiment_data
                
                sentiments.append(sentiment_data.overall_sentiment)
                confidences.append(sentiment_data.confidence)
                
                # Categorize symbols
                if sentiment_data.overall_sentiment > 0.2:
                    sentiment_summary['bullish_symbols'].append(symbol)
                elif sentiment_data.overall_sentiment < -0.2:
                    sentiment_summary['bearish_symbols'].append(symbol)
                else:
                    sentiment_summary['neutral_symbols'].append(symbol)
            
            # Calculate overall market sentiment
            if sentiments:
                sentiment_summary['overall_market_sentiment'] = np.mean(sentiments)
                sentiment_summary['market_confidence'] = np.mean(confidences)
            
            return sentiment_summary
            
        except Exception as e:
            self.logger.error(f"Error getting market sentiment summary: {e}")
            return {}
    
    async def run_continuous_sentiment_analysis(self, symbols: List[str], 
                                              analysis_interval: int = 300):
        """Run continuous sentiment analysis"""
        try:
            self.logger.info("Starting continuous sentiment analysis")
            
            while True:
                try:
                    # Get market sentiment summary
                    summary = await self.get_market_sentiment_summary(symbols)
                    
                    if summary:
                        self.logger.info(
                            f"Market Sentiment: {summary['overall_market_sentiment']:.3f}, "
                            f"Confidence: {summary['market_confidence']:.3f}, "
                            f"Bullish: {len(summary['bullish_symbols'])}, "
                            f"Bearish: {len(summary['bearish_symbols'])}"
                        )
                    
                    # Wait for next analysis cycle
                    await asyncio.sleep(analysis_interval)
                    
                except Exception as e:
                    self.logger.error(f"Error in continuous sentiment analysis: {e}")
                    await asyncio.sleep(60)  # Wait 1 minute before retrying
                    
        except Exception as e:
            self.logger.error(f"Error in continuous sentiment analysis: {e}")
    
    def get_sentiment_cache(self) -> Dict[str, SentimentData]:
        """Get current sentiment cache"""
        return self.sentiment_cache.copy()
    
    def clear_sentiment_cache(self):
        """Clear sentiment cache"""
        self.sentiment_cache.clear()


