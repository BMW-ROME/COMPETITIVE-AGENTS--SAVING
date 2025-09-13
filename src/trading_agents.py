"""
Concrete implementations of competitive trading agents.
"""
import asyncio
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
import json

from src.base_agent import BaseTradingAgent, TradeDecision, PerformanceMetrics
from config.settings import AgentConfig, AgentType

class ConservativeTradingAgent(BaseTradingAgent):
    """
    Conservative trading agent focused on risk management and steady returns.
    """
    
    def __init__(self, config: AgentConfig):
        super().__init__(config)
        self.risk_threshold = 0.02  # 2% max risk per trade
        self.confidence_threshold = 0.7  # High confidence required
        self.position_limit = 0.05  # Max 5% of portfolio per position
        self.stop_loss_pct = 0.03  # 3% stop loss
        self.take_profit_pct = 0.06  # 6% take profit
        
    async def analyze_market_data(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze market data with conservative approach."""
        analysis = {
            "market_trend": "neutral",
            "volatility": "low",
            "risk_level": "low",
            "opportunities": [],
            "warnings": []
        }
        
        # Analyze price trends
        for symbol, data in market_data.get("price_data", {}).items():
            if len(data) < 20:
                continue
                
            prices = [float(d["close"]) for d in data[-20:]]
            sma_20 = np.mean(prices)
            current_price = prices[-1]
            
            # Calculate trend strength
            trend_strength = (current_price - sma_20) / sma_20
            
            if abs(trend_strength) < 0.02:  # Less than 2% deviation
                analysis["market_trend"] = "neutral"
            elif trend_strength > 0.02:
                analysis["market_trend"] = "bullish"
            else:
                analysis["market_trend"] = "bearish"
            
            # Calculate volatility
            returns = np.diff(prices) / prices[:-1]
            volatility = np.std(returns)
            
            if volatility < 0.02:
                analysis["volatility"] = "low"
            elif volatility < 0.05:
                analysis["volatility"] = "medium"
            else:
                analysis["volatility"] = "high"
                analysis["warnings"].append(f"High volatility in {symbol}")
        
        # Conservative opportunity identification
        if analysis["market_trend"] == "bullish" and analysis["volatility"] == "low":
            analysis["opportunities"].append("stable_uptrend")
        elif analysis["market_trend"] == "bearish" and analysis["volatility"] == "low":
            analysis["opportunities"].append("stable_downtrend")
        
        return analysis
    
    async def make_trading_decision(self, market_data: Dict[str, Any]) -> Optional[TradeDecision]:
        """Make conservative trading decisions."""
        analysis = await self.analyze_market_data(market_data)
        
        # Only trade in low-risk conditions
        if analysis["volatility"] == "high" or len(analysis["warnings"]) > 0:
            return None
        
        # Look for stable trends
        if "stable_uptrend" in analysis["opportunities"]:
            return await self._create_buy_decision(market_data, analysis)
        elif "stable_downtrend" in analysis["opportunities"]:
            return await self._create_sell_decision(market_data, analysis)
        
        return None
    
    async def _create_buy_decision(self, market_data: Dict[str, Any], analysis: Dict[str, Any]) -> TradeDecision:
        """Create a conservative buy decision."""
        # Find the most stable symbol
        best_symbol = None
        best_confidence = 0
        
        for symbol, data in market_data.get("price_data", {}).items():
            if len(data) < 20:
                continue
                
            prices = [float(d["close"]) for d in data[-20:]]
            sma_20 = np.mean(prices)
            current_price = prices[-1]
            
            # Calculate trend consistency
            trend_consistency = self._calculate_trend_consistency(prices)
            volatility = np.std(np.diff(prices) / prices[:-1])
            
            # Conservative scoring
            confidence = trend_consistency * (1 - volatility) * 0.8  # Conservative multiplier
            
            if confidence > best_confidence and confidence > self.confidence_threshold:
                best_confidence = confidence
                best_symbol = symbol
        
        if best_symbol:
            # Conservative position sizing
            position_value = self.current_capital * self.position_limit
            quantity = position_value / market_data["price_data"][best_symbol][-1]["close"]
            
            return TradeDecision(
                symbol=best_symbol,
                action="BUY",
                quantity=quantity,
                price=market_data["price_data"][best_symbol][-1]["close"],
                confidence=best_confidence,
                reasoning=f"Conservative buy: stable uptrend, low volatility, confidence: {best_confidence:.2f}",
                timestamp=datetime.now(),
                agent_id=self.agent_id
            )
        
        return None
    
    async def _create_sell_decision(self, market_data: Dict[str, Any], analysis: Dict[str, Any]) -> TradeDecision:
        """Create a conservative sell decision."""
        # Only sell if we have positions
        if not self.positions:
            return None
        
        # Find positions to sell
        for symbol, quantity in self.positions.items():
            if quantity > 0 and symbol in market_data.get("price_data", {}):
                data = market_data["price_data"][symbol]
                if len(data) < 20:
                    continue
                
                prices = [float(d["close"]) for d in data[-20:]]
                sma_20 = np.mean(prices)
                current_price = prices[-1]
                
                # Conservative sell conditions
                if current_price < sma_20 * 0.97:  # 3% below SMA
                    return TradeDecision(
                        symbol=symbol,
                        action="SELL",
                        quantity=quantity,
                        price=current_price,
                        confidence=0.8,
                        reasoning=f"Conservative sell: price below SMA, risk management",
                        timestamp=datetime.now(),
                        agent_id=self.agent_id
                    )
        
        return None
    
    def _calculate_trend_consistency(self, prices: List[float]) -> float:
        """Calculate how consistent the trend is."""
        if len(prices) < 5:
            return 0
        
        # Calculate moving averages
        sma_5 = np.mean(prices[-5:])
        sma_10 = np.mean(prices[-10:]) if len(prices) >= 10 else sma_5
        
        # Trend consistency score
        consistency = 1 - abs(sma_5 - sma_10) / sma_10
        return max(0, min(1, consistency))
    
    async def update_strategy(self, performance_feedback: Dict[str, Any]) -> None:
        """Update strategy based on performance feedback."""
        # Conservative agents adapt slowly
        if performance_feedback.get("win_rate", 0) < 0.6:
            self.confidence_threshold += 0.05  # Increase confidence requirement
        elif performance_feedback.get("win_rate", 0) > 0.8:
            self.confidence_threshold = max(0.6, self.confidence_threshold - 0.02)
        
        # Adjust position sizing based on performance
        if performance_feedback.get("max_drawdown", 0) > 0.05:
            self.position_limit *= 0.9  # Reduce position size
        elif performance_feedback.get("sharpe_ratio", 0) > 1.5:
            self.position_limit = min(0.1, self.position_limit * 1.05)  # Slightly increase


class AggressiveTradingAgent(BaseTradingAgent):
    """
    Aggressive trading agent focused on maximizing returns and capitalizing on opportunities.
    """
    
    def __init__(self, config: AgentConfig):
        super().__init__(config)
        self.risk_threshold = 0.05  # 5% max risk per trade
        self.confidence_threshold = 0.5  # Lower confidence threshold
        self.position_limit = 0.15  # Max 15% of portfolio per position
        self.stop_loss_pct = 0.05  # 5% stop loss
        self.take_profit_pct = 0.10  # 10% take profit
        self.momentum_threshold = 0.03  # 3% momentum threshold
        
    async def analyze_market_data(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze market data with aggressive approach."""
        analysis = {
            "market_trend": "neutral",
            "volatility": "medium",
            "risk_level": "medium",
            "opportunities": [],
            "momentum_signals": []
        }
        
        # Analyze price trends and momentum
        for symbol, data in market_data.get("price_data", {}).items():
            if len(data) < 10:
                continue
                
            prices = [float(d["close"]) for d in data[-10:]]
            current_price = prices[-1]
            
            # Calculate momentum
            momentum = (current_price - prices[0]) / prices[0]
            
            if abs(momentum) > self.momentum_threshold:
                if momentum > 0:
                    analysis["momentum_signals"].append(f"{symbol}_bullish")
                else:
                    analysis["momentum_signals"].append(f"{symbol}_bearish")
            
            # Calculate volatility
            returns = np.diff(prices) / prices[:-1]
            volatility = np.std(returns)
            
            if volatility > 0.03:  # Higher volatility threshold for aggressive trading
                analysis["opportunities"].append(f"{symbol}_high_volatility")
        
        # Aggressive opportunity identification
        if len(analysis["momentum_signals"]) > 0:
            analysis["opportunities"].extend(["momentum_trading", "volatility_trading"])
        
        return analysis
    
    async def make_trading_decision(self, market_data: Dict[str, Any]) -> Optional[TradeDecision]:
        """Make aggressive trading decisions."""
        analysis = await self.analyze_market_data(market_data)
        
        # Look for momentum opportunities
        if "momentum_trading" in analysis["opportunities"]:
            return await self._create_momentum_decision(market_data, analysis)
        
        # Look for volatility opportunities
        if "volatility_trading" in analysis["opportunities"]:
            return await self._create_volatility_decision(market_data, analysis)
        
        return None
    
    async def _create_momentum_decision(self, market_data: Dict[str, Any], analysis: Dict[str, Any]) -> TradeDecision:
        """Create a momentum-based trading decision."""
        best_symbol = None
        best_momentum = 0
        best_confidence = 0
        
        for signal in analysis["momentum_signals"]:
            symbol = signal.split("_")[0]
            direction = signal.split("_")[1]
            
            if symbol in market_data.get("price_data", {}):
                data = market_data["price_data"][symbol]
                if len(data) < 10:
                    continue
                
                prices = [float(d["close"]) for d in data[-10:]]
                momentum = (prices[-1] - prices[0]) / prices[0]
                
                # Aggressive confidence calculation
                confidence = min(0.9, abs(momentum) * 10)  # Scale momentum to confidence
                
                if abs(momentum) > abs(best_momentum) and confidence > self.confidence_threshold:
                    best_momentum = momentum
                    best_symbol = symbol
                    best_confidence = confidence
        
        if best_symbol:
            # Aggressive position sizing
            position_value = self.current_capital * self.position_limit
            quantity = position_value / market_data["price_data"][best_symbol][-1]["close"]
            
            action = "BUY" if best_momentum > 0 else "SELL"
            
            return TradeDecision(
                symbol=best_symbol,
                action=action,
                quantity=quantity,
                price=market_data["price_data"][best_symbol][-1]["close"],
                confidence=best_confidence,
                reasoning=f"Aggressive {action.lower()}: momentum {best_momentum:.3f}, confidence: {best_confidence:.2f}",
                timestamp=datetime.now(),
                agent_id=self.agent_id
            )
        
        return None
    
    async def _create_volatility_decision(self, market_data: Dict[str, Any], analysis: Dict[str, Any]) -> TradeDecision:
        """Create a volatility-based trading decision."""
        # Find highest volatility symbol
        best_symbol = None
        best_volatility = 0
        
        for symbol, data in market_data.get("price_data", {}).items():
            if len(data) < 10:
                continue
            
            prices = [float(d["close"]) for d in data[-10:]]
            returns = np.diff(prices) / prices[:-1]
            volatility = np.std(returns)
            
            if volatility > best_volatility:
                best_volatility = volatility
                best_symbol = symbol
        
        if best_symbol and best_volatility > 0.03:
            # Volatility trading strategy
            data = market_data["price_data"][best_symbol]
            current_price = float(data[-1]["close"])
            
            # Simple mean reversion strategy
            sma_5 = np.mean([float(d["close"]) for d in data[-5:]])
            
            if current_price > sma_5 * 1.02:  # Price above SMA
                action = "SELL"
                confidence = min(0.8, best_volatility * 10)
            else:  # Price below SMA
                action = "BUY"
                confidence = min(0.8, best_volatility * 10)
            
            position_value = self.current_capital * self.position_limit
            quantity = position_value / current_price
            
            return TradeDecision(
                symbol=best_symbol,
                action=action,
                quantity=quantity,
                price=current_price,
                confidence=confidence,
                reasoning=f"Volatility trading: {action.lower()} on high volatility {best_volatility:.3f}",
                timestamp=datetime.now(),
                agent_id=self.agent_id
            )
        
        return None
    
    async def update_strategy(self, performance_feedback: Dict[str, Any]) -> None:
        """Update strategy based on performance feedback."""
        # Aggressive agents adapt quickly
        if performance_feedback.get("win_rate", 0) < 0.5:
            self.confidence_threshold += 0.1  # Increase confidence requirement
            self.position_limit *= 0.8  # Reduce position size
        elif performance_feedback.get("win_rate", 0) > 0.7:
            self.confidence_threshold = max(0.3, self.confidence_threshold - 0.05)
            self.position_limit = min(0.2, self.position_limit * 1.1)  # Increase position size
        
        # Adjust momentum threshold based on market conditions
        if performance_feedback.get("volatility", 0) > 0.05:
            self.momentum_threshold *= 1.2  # Increase momentum threshold
        else:
            self.momentum_threshold = max(0.02, self.momentum_threshold * 0.9)


class BalancedTradingAgent(BaseTradingAgent):
    """
    Balanced trading agent that combines conservative and aggressive strategies.
    """
    
    def __init__(self, config: AgentConfig):
        super().__init__(config)
        self.risk_threshold = 0.035  # 3.5% max risk per trade
        self.confidence_threshold = 0.6  # Medium confidence threshold
        self.position_limit = 0.1  # Max 10% of portfolio per position
        self.stop_loss_pct = 0.04  # 4% stop loss
        self.take_profit_pct = 0.08  # 8% take profit
        
        # Strategy weights
        self.conservative_weight = 0.6
        self.aggressive_weight = 0.4
        
    async def analyze_market_data(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze market data with balanced approach."""
        # Combine conservative and aggressive analysis
        conservative_analysis = await self._conservative_analysis(market_data)
        aggressive_analysis = await self._aggressive_analysis(market_data)
        
        # Weighted combination
        analysis = {
            "market_trend": self._weighted_decision(
                conservative_analysis["market_trend"],
                aggressive_analysis["market_trend"]
            ),
            "volatility": self._weighted_decision(
                conservative_analysis["volatility"],
                aggressive_analysis["volatility"]
            ),
            "opportunities": list(set(
                conservative_analysis["opportunities"] + 
                aggressive_analysis["opportunities"]
            )),
            "confidence": (self.conservative_weight + self.aggressive_weight) / 2
        }
        
        return analysis
    
    async def _conservative_analysis(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Perform conservative analysis."""
        # Simplified conservative analysis
        return {
            "market_trend": "neutral",
            "volatility": "low",
            "opportunities": ["stable_trends"]
        }
    
    async def _aggressive_analysis(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Perform aggressive analysis."""
        # Simplified aggressive analysis
        return {
            "market_trend": "bullish",
            "volatility": "medium",
            "opportunities": ["momentum_trading"]
        }
    
    def _weighted_decision(self, conservative: str, aggressive: str) -> str:
        """Make weighted decision between conservative and aggressive inputs."""
        # Simple weighted decision logic
        if conservative == aggressive:
            return conservative
        else:
            # Return the more conservative option
            return conservative
    
    async def make_trading_decision(self, market_data: Dict[str, Any]) -> Optional[TradeDecision]:
        """Make balanced trading decisions."""
        analysis = await self.analyze_market_data(market_data)
        
        # Use both strategies with weights
        if "stable_trends" in analysis["opportunities"]:
            return await self._create_balanced_decision(market_data, analysis, "conservative")
        elif "momentum_trading" in analysis["opportunities"]:
            return await self._create_balanced_decision(market_data, analysis, "aggressive")
        
        return None
    
    async def _create_balanced_decision(self, market_data: Dict[str, Any], analysis: Dict[str, Any], strategy: str) -> TradeDecision:
        """Create a balanced trading decision."""
        # Find best symbol based on strategy
        best_symbol = None
        best_confidence = 0
        
        for symbol, data in market_data.get("price_data", {}).items():
            if len(data) < 10:
                continue
            
            prices = [float(d["close"]) for d in data[-10:]]
            current_price = prices[-1]
            
            if strategy == "conservative":
                # Conservative scoring
                sma_10 = np.mean(prices)
                trend_consistency = 1 - abs(current_price - sma_10) / sma_10
                confidence = trend_consistency * self.conservative_weight
            else:
                # Aggressive scoring
                momentum = (current_price - prices[0]) / prices[0]
                confidence = abs(momentum) * 5 * self.aggressive_weight
            
            if confidence > best_confidence and confidence > self.confidence_threshold:
                best_confidence = confidence
                best_symbol = symbol
        
        if best_symbol:
            position_value = self.current_capital * self.position_limit
            quantity = position_value / market_data["price_data"][best_symbol][-1]["close"]
            
            # Determine action based on trend
            data = market_data["price_data"][best_symbol]
            prices = [float(d["close"]) for d in data[-10:]]
            sma_5 = np.mean(prices[-5:])
            current_price = prices[-1]
            
            action = "BUY" if current_price > sma_5 else "SELL"
            
            return TradeDecision(
                symbol=best_symbol,
                action=action,
                quantity=quantity,
                price=current_price,
                confidence=best_confidence,
                reasoning=f"Balanced {action.lower()}: {strategy} strategy, confidence: {best_confidence:.2f}",
                timestamp=datetime.now(),
                agent_id=self.agent_id
            )
        
        return None
    
    async def update_strategy(self, performance_feedback: Dict[str, Any]) -> None:
        """Update strategy based on performance feedback."""
        # Adjust strategy weights based on performance
        if performance_feedback.get("win_rate", 0) < 0.6:
            # Increase conservative weight
            self.conservative_weight = min(0.8, self.conservative_weight + 0.1)
            self.aggressive_weight = max(0.2, self.aggressive_weight - 0.1)
        elif performance_feedback.get("total_return", 0) > 0.1:
            # Increase aggressive weight
            self.aggressive_weight = min(0.6, self.aggressive_weight + 0.1)
            self.conservative_weight = max(0.4, self.conservative_weight - 0.1)
        
        # Adjust confidence threshold
        if performance_feedback.get("sharpe_ratio", 0) < 1.0:
            self.confidence_threshold += 0.05
        elif performance_feedback.get("sharpe_ratio", 0) > 1.5:
            self.confidence_threshold = max(0.5, self.confidence_threshold - 0.02)
