#!/usr/bin/env python3
"""
Forex and Crypto Trading Agents
==============================

Specialized agents for forex and cryptocurrency trading.
"""

import asyncio
import logging
import random
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from dataclasses import dataclass

from src.base_agent import BaseTradingAgent, TradeDecision
from config.settings import AgentConfig, AgentType

@dataclass
class ForexMarketData:
    """Forex market data structure."""
    symbol: str
    bid: float
    ask: float
    spread: float
    volume: float
    timestamp: datetime

@dataclass
class CryptoMarketData:
    """Cryptocurrency market data structure."""
    symbol: str
    price: float
    volume_24h: float
    market_cap: float
    volatility: float
    timestamp: datetime

class ForexSpecialistAgent(BaseTradingAgent):
    """
    Specialist agent for forex trading.
    Focuses on major currency pairs and economic indicators.
    """
    
    def __init__(self, config: AgentConfig):
        super().__init__(config)
        self.specialization = "forex"
        self.major_pairs = ["EURUSD", "GBPUSD", "USDJPY", "USDCHF", "AUDUSD", "USDCAD"]
        self.minor_pairs = ["EURGBP", "EURJPY", "GBPJPY", "CHFJPY", "AUDJPY", "NZDUSD"]
        self.exotic_pairs = ["USDTRY", "USDZAR", "USDMXN", "USDSEK", "USDNOK"]
        
        # Forex-specific parameters
        self.max_spread = 0.0005  # 5 pips max spread
        self.leverage = 50  # 50:1 leverage for forex
        self.position_size_multiplier = 0.02  # 2% risk per trade
        
    async def analyze_market_data(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze forex market data."""
        try:
            analysis = {
                "market_trend": "neutral",
                "volatility": "low",
                "opportunities": [],
                "warnings": [],
                "forex_analysis": {}
            }
            
            # Analyze major pairs
            for pair in self.major_pairs:
                if pair in market_data:
                    pair_data = market_data[pair]
                    if isinstance(pair_data, dict) and 'price_data' in pair_data and isinstance(pair_data['price_data'], list):
                        price_data = pair_data['price_data']
                        if price_data and len(price_data) > 0:
                            # Calculate spread
                            first_bar = price_data[0] if isinstance(price_data[0], dict) else {}
                            if 'bid' in first_bar and 'ask' in first_bar:
                                spread = first_bar.get('ask', 0) - first_bar.get('bid', 0)
                                if spread <= self.max_spread:
                                    analysis["opportunities"].append(f"low_spread_{pair}")
                            
                            # Check for trend
                            if len(price_data) >= 10:
                                recent_prices = [ (p.get('close', 0) if isinstance(p, dict) else 0) for p in price_data[-10:]]
                                if recent_prices and all(p > 0 for p in recent_prices):
                                    trend = (recent_prices[-1] - recent_prices[0]) / recent_prices[0]
                                    if abs(trend) > 0.001:  # 0.1% movement
                                        analysis["opportunities"].append(f"trend_{pair}")
                                        analysis["market_trend"] = "bullish" if trend > 0 else "bearish"
            
            # Check for economic events (simplified)
            if isinstance(market_data.get("news_data"), list):
                news_data = market_data["news_data"]
                if news_data:
                    # Look for high-impact news
                    for news_item in news_data:
                        title = news_item.get('title', '') if isinstance(news_item, dict) else ''
                        if any(keyword in title.lower() 
                               for keyword in ['fed', 'ecb', 'boe', 'rate', 'inflation', 'gdp']):
                            analysis["opportunities"].append("high_impact_news")
                            break
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"Error analyzing forex market data: {e}")
            return {"market_trend": "neutral", "volatility": "low", "opportunities": [], "warnings": []}
    
    async def make_trading_decision(self, market_data: Dict[str, Any]) -> Optional[TradeDecision]:
        """Make forex trading decision."""
        try:
            analysis = await self.analyze_market_data(market_data)
            
            if not analysis["opportunities"]:
                return None
            
            # Find best opportunity
            best_pair = None
            best_opportunity = None
            
            for opportunity in analysis["opportunities"]:
                if opportunity.startswith("trend_"):
                    pair = opportunity.split("_")[1]
                    if not best_pair or pair in self.major_pairs:
                        best_pair = pair
                        best_opportunity = opportunity
                        break
            
            if not best_pair:
                return None
            
            # Get current price
            pair_data = market_data.get(best_pair, {})
            if isinstance(pair_data, dict) and 'price_data' in pair_data and isinstance(pair_data['price_data'], list):
                price_data = pair_data['price_data']
                if price_data and len(price_data) > 0:
                    last_bar = price_data[-1] if isinstance(price_data[-1], dict) else {}
                    current_price = last_bar.get('close', 0)
                    if current_price <= 0:
                        return None
                    
                    # Determine action based on trend
                    action = "BUY" if "bullish" in analysis["market_trend"] else "SELL"
                    
                    # Calculate position size
                    account_value = self.current_capital
                    risk_amount = account_value * self.position_size_multiplier
                    position_size = risk_amount / current_price
                    
                    # Apply leverage
                    position_size *= self.leverage
                    
                    # Ensure minimum position size
                    if position_size < 1000:  # Minimum 1000 units for forex
                        return None
                    
                    return TradeDecision(
                        symbol=best_pair,
                        action=action,
                        quantity=position_size,
                        price=current_price,
                        confidence=0.7,
                        reasoning=f"Forex specialist: {best_opportunity} on {best_pair}",
                        timestamp=datetime.now(),
                        agent_id=self.agent_id
                    )
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error making forex trading decision: {e}")
            return None

    async def update_strategy(self, performance_feedback: Dict[str, Any]) -> None:
        """Update forex trading strategy based on performance feedback."""
        self.logger.info(f"ForexSpecialistAgent {self.agent_id} updating strategy.")
        
        # Adjust confidence threshold based on performance
        win_rate = performance_feedback.get('win_rate', 0.5)
        if win_rate < 0.4:
            self.max_spread *= 0.9  # Reduce spread tolerance
            self.position_size_multiplier *= 0.8  # Reduce position size
        elif win_rate > 0.7:
            self.max_spread *= 1.1  # Increase spread tolerance
            self.position_size_multiplier = min(0.05, self.position_size_multiplier * 1.1)  # Increase position size
        
        # Adjust leverage based on volatility
        volatility = performance_feedback.get('volatility', 0.02)
        if volatility > 0.05:
            self.leverage = max(10, self.leverage * 0.8)  # Reduce leverage in high volatility
        elif volatility < 0.01:
            self.leverage = min(100, self.leverage * 1.1)  # Increase leverage in low volatility

class CryptoSpecialistAgent(BaseTradingAgent):
    """
    Specialist agent for cryptocurrency trading.
    Focuses on major cryptocurrencies and market sentiment.
    """
    
    def __init__(self, config: AgentConfig):
        super().__init__(config)
        self.specialization = "crypto"
        self.major_cryptos = ["BTCUSD", "ETHUSD", "ADAUSD", "DOTUSD", "LINKUSD"]
        self.defi_tokens = ["UNIUSD", "AAVEUSD", "SUSHIUSD", "COMPUSD", "YFIUSD"]
        self.meme_coins = ["DOGEUSD", "SHIBUSD", "PEPEUSD"]
        
        # Crypto-specific parameters
        self.high_volatility_threshold = 0.05  # 5% daily volatility
        self.momentum_threshold = 0.02  # 2% momentum
        self.max_position_size = 0.1  # 10% max position size
        
    async def analyze_market_data(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze crypto market data."""
        try:
            analysis = {
                "market_trend": "neutral",
                "volatility": "low",
                "opportunities": [],
                "warnings": [],
                "crypto_analysis": {}
            }
            
            # Analyze major cryptocurrencies
            for crypto in self.major_cryptos:
                if crypto in market_data:
                    crypto_data = market_data[crypto]
                    if isinstance(crypto_data, dict) and 'price_data' in crypto_data and isinstance(crypto_data['price_data'], list):
                        price_data = crypto_data['price_data']
                        if price_data and len(price_data) >= 24:  # 24 hours of data
                            # Calculate volatility
                            prices = [(p.get('close', 0) if isinstance(p, dict) else 0) for p in price_data[-24:]]
                            if prices and all(p > 0 for p in prices):
                                volatility = self._calculate_volatility(prices)
                                if volatility > self.high_volatility_threshold:
                                    analysis["opportunities"].append(f"high_volatility_{crypto}")
                                    analysis["volatility"] = "high"
                                
                                # Calculate momentum
                                momentum = (prices[-1] - prices[0]) / prices[0]
                                if abs(momentum) > self.momentum_threshold:
                                    analysis["opportunities"].append(f"momentum_{crypto}")
                                    analysis["market_trend"] = "bullish" if momentum > 0 else "bearish"
            
            # Check for DeFi opportunities
            for token in self.defi_tokens:
                if token in market_data:
                    token_data = market_data[token]
                    if isinstance(token_data, dict) and 'price_data' in token_data and isinstance(token_data['price_data'], list):
                        price_data = token_data['price_data']
                        if price_data and len(price_data) > 0:
                            # Check for unusual volume
                            last_bar = price_data[-1] if isinstance(price_data[-1], dict) else {}
                            if 'volume' in last_bar:
                                volume = last_bar.get('volume', 0)
                                if volume > 1000000:  # High volume threshold
                                    analysis["opportunities"].append(f"high_volume_{token}")
            
            # Check social sentiment
            if isinstance(market_data.get("social_sentiment"), dict):
                sentiment_data = market_data["social_sentiment"]
                if sentiment_data:
                    for crypto in self.major_cryptos:
                        if crypto in sentiment_data and isinstance(sentiment_data[crypto], dict):
                            sentiment = sentiment_data[crypto]
                            if sentiment.get('sentiment_score', 0) > 0.7:
                                analysis["opportunities"].append(f"positive_sentiment_{crypto}")
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"Error analyzing crypto market data: {e}")
            return {"market_trend": "neutral", "volatility": "low", "opportunities": [], "warnings": []}
    
    def _calculate_volatility(self, prices: List[float]) -> float:
        """Calculate price volatility."""
        if len(prices) < 2:
            return 0.0
        
        returns = []
        for i in range(1, len(prices)):
            if prices[i-1] > 0:
                returns.append((prices[i] - prices[i-1]) / prices[i-1])
        
        if not returns:
            return 0.0
        
        mean_return = sum(returns) / len(returns)
        variance = sum((r - mean_return) ** 2 for r in returns) / len(returns)
        return variance ** 0.5
    
    async def make_trading_decision(self, market_data: Dict[str, Any]) -> Optional[TradeDecision]:
        """Make crypto trading decision."""
        try:
            analysis = await self.analyze_market_data(market_data)
            
            if not analysis["opportunities"]:
                return None
            
            # Find best opportunity
            best_crypto = None
            best_opportunity = None
            
            for opportunity in analysis["opportunities"]:
                if opportunity.startswith("momentum_") or opportunity.startswith("high_volatility_"):
                    crypto = opportunity.split("_", 1)[1]
                    if not best_crypto or crypto in self.major_cryptos:
                        best_crypto = crypto
                        best_opportunity = opportunity
                        break
            
            if not best_crypto:
                return None
            
            # Get current price
            crypto_data = market_data.get(best_crypto, {})
            if isinstance(crypto_data, dict) and 'price_data' in crypto_data and isinstance(crypto_data['price_data'], list):
                price_data = crypto_data['price_data']
                if price_data and len(price_data) > 0:
                    last_bar = price_data[-1] if isinstance(price_data[-1], dict) else {}
                    current_price = last_bar.get('close', 0)
                    if current_price <= 0:
                        return None
                    
                    # Determine action based on trend
                    action = "BUY" if "bullish" in analysis["market_trend"] else "SELL"
                    
                    # Calculate position size
                    account_value = self.current_capital
                    max_position_value = account_value * self.max_position_size
                    position_size = max_position_value / current_price
                    
                    # Ensure minimum position size
                    if position_size < 0.001:  # Minimum 0.001 units for crypto
                        return None
                    
                    return TradeDecision(
                        symbol=best_crypto,
                        action=action,
                        quantity=position_size,
                        price=current_price,
                        confidence=0.8,
                        reasoning=f"Crypto specialist: {best_opportunity} on {best_crypto}",
                        timestamp=datetime.now(),
                        agent_id=self.agent_id
                    )
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error making crypto trading decision: {e}")
            return None

    async def update_strategy(self, performance_feedback: Dict[str, Any]) -> None:
        """Update crypto trading strategy based on performance feedback."""
        self.logger.info(f"CryptoSpecialistAgent {self.agent_id} updating strategy.")
        
        # Adjust volatility threshold based on performance
        win_rate = performance_feedback.get('win_rate', 0.5)
        if win_rate < 0.4:
            self.high_volatility_threshold *= 1.2  # Increase volatility threshold
            self.momentum_threshold *= 1.1  # Increase momentum threshold
        elif win_rate > 0.7:
            self.high_volatility_threshold *= 0.9  # Decrease volatility threshold
            self.momentum_threshold *= 0.95  # Decrease momentum threshold
        
        # Adjust position size based on market conditions
        volatility = performance_feedback.get('volatility', 0.02)
        if volatility > 0.1:  # Very high volatility
            self.max_position_size = max(0.05, self.max_position_size * 0.8)  # Reduce position size
        elif volatility < 0.02:  # Low volatility
            self.max_position_size = min(0.15, self.max_position_size * 1.1)  # Increase position size

class MultiAssetArbitrageAgent(BaseTradingAgent):
    """
    Multi-asset arbitrage agent.
    Looks for arbitrage opportunities across different asset classes.
    """
    
    def __init__(self, config: AgentConfig):
        super().__init__(config)
        self.specialization = "arbitrage"
        self.arbitrage_threshold = 0.01  # 1% minimum arbitrage opportunity
        self.max_position_size = 0.05  # 5% max position size per arbitrage
        
    async def analyze_market_data(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze arbitrage opportunities."""
        try:
            analysis = {
                "market_trend": "neutral",
                "volatility": "low",
                "opportunities": [],
                "warnings": [],
                "arbitrage_analysis": {}
            }
            
            # Look for arbitrage opportunities
            symbols = list(market_data.keys())
            
            for i, symbol1 in enumerate(symbols):
                for j, symbol2 in enumerate(symbols[i+1:], i+1):
                    if self._are_related_assets(symbol1, symbol2):
                        arbitrage_opportunity = self._find_arbitrage_opportunity(
                            symbol1, symbol2, market_data
                        )
                        if arbitrage_opportunity:
                            analysis["opportunities"].append(arbitrage_opportunity)
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"Error analyzing arbitrage opportunities: {e}")
            return {"market_trend": "neutral", "volatility": "low", "opportunities": [], "warnings": []}
    
    def _are_related_assets(self, symbol1: str, symbol2: str) -> bool:
        """Check if two assets are related for arbitrage."""
        # Forex pairs
        if symbol1.endswith("USD") and symbol2.endswith("USD"):
            return True
        
        # Crypto pairs
        if symbol1.endswith("USD") and symbol2.endswith("USD") and "BTC" in symbol1 and "ETH" in symbol2:
            return True
        
        # ETF and underlying
        if (symbol1 == "SPY" and symbol2 in ["AAPL", "MSFT", "GOOGL"]) or \
           (symbol2 == "SPY" and symbol1 in ["AAPL", "MSFT", "GOOGL"]):
            return True
        
        return False
    
    def _find_arbitrage_opportunity(self, symbol1: str, symbol2: str, market_data: Dict[str, Any]) -> Optional[str]:
        """Find arbitrage opportunity between two assets."""
        try:
            data1 = market_data.get(symbol1, {})
            data2 = market_data.get(symbol2, {})
            
            if not (isinstance(data1, dict) and isinstance(data2, dict)):
                return None
            
            price_data1 = data1.get('price_data', [])
            price_data2 = data2.get('price_data', [])
            
            if not (price_data1 and price_data2):
                return None
            
            price1 = price_data1[-1].get('close', 0)
            price2 = price_data2[-1].get('close', 0)
            
            if price1 <= 0 or price2 <= 0:
                return None
            
            # Calculate price ratio
            ratio = price1 / price2
            
            # Check if ratio is significantly different from expected
            # This is a simplified arbitrage detection
            if abs(ratio - 1.0) > self.arbitrage_threshold:
                return f"arbitrage_{symbol1}_{symbol2}"
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error finding arbitrage opportunity: {e}")
            return None
    
    async def make_trading_decision(self, market_data: Dict[str, Any]) -> Optional[TradeDecision]:
        """Make arbitrage trading decision."""
        try:
            analysis = await self.analyze_market_data(market_data)
            
            if not analysis["opportunities"]:
                return None
            
            # Find best arbitrage opportunity
            best_opportunity = None
            for opportunity in analysis["opportunities"]:
                if opportunity.startswith("arbitrage_"):
                    best_opportunity = opportunity
                    break
            
            if not best_opportunity:
                return None
            
            # Parse opportunity
            parts = best_opportunity.split("_")
            if len(parts) >= 3:
                symbol1 = parts[1]
                symbol2 = parts[2]
                
                # Get current prices
                data1 = market_data.get(symbol1, {})
                data2 = market_data.get(symbol2, {})
                
                if isinstance(data1, dict) and isinstance(data2, dict):
                    price_data1 = data1.get('price_data', [])
                    price_data2 = data2.get('price_data', [])
                    
                    if price_data1 and price_data2:
                        price1 = price_data1[-1].get('close', 0)
                        price2 = price_data2[-1].get('close', 0)
                        
                        if price1 > 0 and price2 > 0:
                            # Determine which asset to buy/sell
                            if price1 / price2 > 1.0:
                                # Buy symbol2, sell symbol1
                                action = "BUY"
                                symbol = symbol2
                                price = price2
                            else:
                                # Buy symbol1, sell symbol2
                                action = "BUY"
                                symbol = symbol1
                                price = price1
                            
                            # Calculate position size
                            account_value = self.current_capital
                            max_position_value = account_value * self.max_position_size
                            position_size = max_position_value / price
                            
                            if position_size > 0:
                                return TradeDecision(
                                    symbol=symbol,
                                    action=action,
                                    quantity=position_size,
                                    price=price,
                                    confidence=0.9,
                                    reasoning=f"Arbitrage opportunity: {best_opportunity}",
                                    timestamp=datetime.now(),
                                    agent_id=self.agent_id
                                )
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error making arbitrage trading decision: {e}")
            return None

    async def update_strategy(self, performance_feedback: Dict[str, Any]) -> None:
        """Update arbitrage trading strategy based on performance feedback."""
        self.logger.info(f"MultiAssetArbitrageAgent {self.agent_id} updating strategy.")
        
        # Adjust arbitrage threshold based on performance
        win_rate = performance_feedback.get('win_rate', 0.5)
        if win_rate < 0.3:
            self.arbitrage_threshold *= 1.5  # Increase threshold for more selective opportunities
            self.max_position_size *= 0.7  # Reduce position size
        elif win_rate > 0.8:
            self.arbitrage_threshold *= 0.8  # Decrease threshold for more opportunities
            self.max_position_size = min(0.1, self.max_position_size * 1.2)  # Increase position size
        
        # Adjust based on market efficiency
        market_efficiency = performance_feedback.get('market_efficiency', 0.5)
        if market_efficiency > 0.8:  # High efficiency markets
            self.arbitrage_threshold *= 1.2  # Be more selective
        elif market_efficiency < 0.3:  # Low efficiency markets
            self.arbitrage_threshold *= 0.9  # Be less selective
