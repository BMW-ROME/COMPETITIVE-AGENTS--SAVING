#!/usr/bin/env python3
"""
Simple Data Provider - No pandas required
"""
import asyncio
import random
import json
from datetime import datetime, timedelta
from typing import Dict, List, Any

class SimpleDataProvider:
    def __init__(self):
        self.symbols = ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'AMZN', 'NVDA', 'META', 'NFLX', 'BTCUSD', 'ETHUSD', 'EURUSD', 'GBPUSD']
        self.base_prices = {
            'AAPL': 150.0, 'GOOGL': 2800.0, 'MSFT': 300.0, 'TSLA': 200.0,
            'AMZN': 3200.0, 'NVDA': 400.0, 'META': 250.0, 'NFLX': 400.0,
            'BTCUSD': 45000.0, 'ETHUSD': 3000.0, 'EURUSD': 1.05, 'GBPUSD': 1.25
        }
    
    async def get_price_data(self, symbols: List[str], period: str = "1d", interval: str = "15m") -> Dict[str, List[Dict[str, Any]]]:
        """Generate realistic mock price data"""
        price_data = {}
        
        for symbol in symbols:
            if symbol not in self.symbols:
                symbol = 'AAPL'  # Default fallback
            
            base_price = self.base_prices.get(symbol, 100.0)
            data_points = []
            
            # Generate 24 hours of 15-minute data
            current_time = datetime.now() - timedelta(hours=24)
            
            for i in range(96):  # 24 hours * 4 (15-min intervals)
                # Add some realistic price movement
                price_change = random.uniform(-0.02, 0.02)  # ±2% change
                base_price *= (1 + price_change)
                
                # Generate OHLC data
                high = base_price * random.uniform(1.0, 1.01)
                low = base_price * random.uniform(0.99, 1.0)
                open_price = base_price * random.uniform(0.995, 1.005)
                close_price = base_price
                volume = random.randint(1000, 10000)
                
                data_points.append({
                    "timestamp": current_time.isoformat(),
                    "open": round(open_price, 2),
                    "high": round(high, 2),
                    "low": round(low, 2),
                    "close": round(close_price, 2),
                    "volume": volume
                })
                
                current_time += timedelta(minutes=15)
            
            price_data[symbol] = data_points
        
        return price_data
    
    async def get_technical_indicators(self, price_data: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
        """Generate mock technical indicators"""
        indicators = {}
        
        for symbol, data in price_data.items():
            if len(data) >= 20:
                closes = [float(d['close']) for d in data[-20:]]
                sma_20 = sum(closes) / len(closes)
                rsi = random.uniform(30, 70)  # Realistic RSI range
                
                indicators[symbol] = {
                    "sma_20": round(sma_20, 2),
                    "rsi": round(rsi, 2),
                    "macd": random.uniform(-1, 1),
                    "bollinger_upper": round(sma_20 * 1.02, 2),
                    "bollinger_lower": round(sma_20 * 0.98, 2)
                }
        
        return indicators
    
    async def get_news_sentiment(self, symbols: List[str]) -> Dict[str, Any]:
        """Generate mock news sentiment"""
        sentiment = {}
        
        for symbol in symbols:
            sentiment[symbol] = {
                "sentiment_score": random.uniform(-1, 1),
                "confidence": random.uniform(0.6, 0.9),
                "news_count": random.randint(5, 20),
                "last_updated": datetime.now().isoformat()
            }
        
        return sentiment

# Export for use in other modules
simple_provider = SimpleDataProvider()
