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
# import talib  # Commented out - requires special installation
from scipy import stats
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

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


class FractalAnalysisAgent(BaseTradingAgent):
    """
    Advanced trading agent using fractal analysis on candlestick charts.
    Identifies fractal patterns and uses them for trading decisions.
    """
    
    def __init__(self, config: AgentConfig):
        super().__init__(config)
        self.risk_threshold = 0.04  # 4% max risk per trade
        self.confidence_threshold = 0.65  # Medium-high confidence
        self.position_limit = 0.12  # Max 12% of portfolio per position
        self.stop_loss_pct = 0.04  # 4% stop loss
        self.take_profit_pct = 0.08  # 8% take profit
        
        # Fractal analysis parameters
        self.fractal_period = 5  # Look for fractals over 5 periods
        self.min_fractal_strength = 0.02  # Minimum 2% price movement
        self.confirmation_periods = 3  # Confirm fractals over 3 periods
        
        # Multi-timeframe analysis
        self.timeframes = ['1min', '5min', '15min', '1hour']
        self.timeframe_weights = {'1min': 0.1, '5min': 0.2, '15min': 0.3, '1hour': 0.4}
        
    async def analyze_market_data(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze market data using fractal patterns."""
        analysis = {
            "fractal_signals": [],
            "timeframe_consensus": {},
            "pattern_strength": 0.0,
            "opportunities": [],
            "risk_level": "medium"
        }
        
        for symbol, data in market_data.get("price_data", {}).items():
            if len(data) < 20:
                continue
                
            # Extract OHLC data
            ohlc_data = self._extract_ohlc(data)
            
            # Analyze fractals across timeframes
            fractal_analysis = await self._analyze_fractals(ohlc_data, symbol)
            analysis["fractal_signals"].extend(fractal_analysis)
            
            # Multi-timeframe consensus
            consensus = self._calculate_timeframe_consensus(fractal_analysis)
            analysis["timeframe_consensus"][symbol] = consensus
            
            # Pattern strength calculation
            pattern_strength = self._calculate_pattern_strength(fractal_analysis)
            analysis["pattern_strength"] = max(analysis["pattern_strength"], pattern_strength)
            
            # Identify opportunities
            if pattern_strength > 0.7:
                analysis["opportunities"].append(f"{symbol}_strong_fractal")
            elif pattern_strength > 0.5:
                analysis["opportunities"].append(f"{symbol}_moderate_fractal")
        
        return analysis
    
    def _extract_ohlc(self, data: List[Dict]) -> Dict[str, List[float]]:
        """Extract OHLC data from market data."""
        return {
            'open': [float(d.get('open', d.get('close', 0))) for d in data],
            'high': [float(d.get('high', d.get('close', 0))) for d in data],
            'low': [float(d.get('low', d.get('close', 0))) for d in data],
            'close': [float(d.get('close', 0)) for d in data],
            'volume': [float(d.get('volume', 1000)) for d in data]
        }
    
    async def _analyze_fractals(self, ohlc_data: Dict[str, List[float]], symbol: str) -> List[Dict]:
        """Analyze fractal patterns in price data."""
        fractals = []
        
        highs = np.array(ohlc_data['high'])
        lows = np.array(ohlc_data['low'])
        closes = np.array(ohlc_data['close'])
        
        # Identify fractal highs and lows
        fractal_highs = self._find_fractal_highs(highs)
        fractal_lows = self._find_fractal_lows(lows)
        
        # Analyze fractal patterns
        for i, is_high in enumerate(fractal_highs):
            if is_high:
                fractal_data = self._analyze_fractal_pattern(highs, lows, closes, i, 'high')
                if fractal_data:
                    fractals.append({
                        'symbol': symbol,
                        'type': 'fractal_high',
                        'index': i,
                        'price': highs[i],
                        'strength': fractal_data['strength'],
                        'confidence': fractal_data['confidence'],
                        'pattern': fractal_data['pattern']
                    })
        
        for i, is_low in enumerate(fractal_lows):
            if is_low:
                fractal_data = self._analyze_fractal_pattern(highs, lows, closes, i, 'low')
                if fractal_data:
                    fractals.append({
                        'symbol': symbol,
                        'type': 'fractal_low',
                        'index': i,
                        'price': lows[i],
                        'strength': fractal_data['strength'],
                        'confidence': fractal_data['confidence'],
                        'pattern': fractal_data['pattern']
                    })
        
        return fractals
    
    def _find_fractal_highs(self, highs: np.ndarray) -> np.ndarray:
        """Find fractal high points."""
        fractal_highs = np.zeros(len(highs), dtype=bool)
        
        for i in range(self.fractal_period, len(highs) - self.fractal_period):
            # Check if current high is higher than surrounding highs
            left_max = np.max(highs[i-self.fractal_period:i])
            right_max = np.max(highs[i+1:i+self.fractal_period+1])
            
            if highs[i] > left_max and highs[i] > right_max:
                # Check minimum strength requirement
                strength = (highs[i] - max(left_max, right_max)) / max(left_max, right_max)
                if strength >= self.min_fractal_strength:
                    fractal_highs[i] = True
        
        return fractal_highs
    
    def _find_fractal_lows(self, lows: np.ndarray) -> np.ndarray:
        """Find fractal low points."""
        fractal_lows = np.zeros(len(lows), dtype=bool)
        
        for i in range(self.fractal_period, len(lows) - self.fractal_period):
            # Check if current low is lower than surrounding lows
            left_min = np.min(lows[i-self.fractal_period:i])
            right_min = np.min(lows[i+1:i+self.fractal_period+1])
            
            if lows[i] < left_min and lows[i] < right_min:
                # Check minimum strength requirement
                strength = (min(left_min, right_min) - lows[i]) / min(left_min, right_min)
                if strength >= self.min_fractal_strength:
                    fractal_lows[i] = True
        
        return fractal_lows
    
    def _analyze_fractal_pattern(self, highs: np.ndarray, lows: np.ndarray, closes: np.ndarray, 
                                index: int, fractal_type: str) -> Optional[Dict]:
        """Analyze a specific fractal pattern."""
        if index < self.confirmation_periods or index >= len(closes) - self.confirmation_periods:
            return None
        
        # Get surrounding data
        start_idx = max(0, index - self.confirmation_periods)
        end_idx = min(len(closes), index + self.confirmation_periods + 1)
        
        pattern_data = {
            'highs': highs[start_idx:end_idx],
            'lows': lows[start_idx:end_idx],
            'closes': closes[start_idx:end_idx]
        }
        
        # Calculate pattern strength
        strength = self._calculate_fractal_strength(pattern_data, fractal_type)
        
        # Calculate confidence based on volume and price action
        confidence = self._calculate_fractal_confidence(pattern_data, fractal_type)
        
        # Identify pattern type
        pattern = self._identify_fractal_pattern(pattern_data, fractal_type)
        
        if strength > 0.3 and confidence > 0.5:
            return {
                'strength': strength,
                'confidence': confidence,
                'pattern': pattern
            }
        
        return None
    
    def _calculate_fractal_strength(self, pattern_data: Dict, fractal_type: str) -> float:
        """Calculate the strength of a fractal pattern."""
        if fractal_type == 'high':
            current_price = pattern_data['highs'][len(pattern_data['highs'])//2]
            surrounding_max = max(np.max(pattern_data['highs'][:len(pattern_data['highs'])//2]),
                                np.max(pattern_data['highs'][len(pattern_data['highs'])//2+1:]))
            strength = (current_price - surrounding_max) / surrounding_max
        else:  # low
            current_price = pattern_data['lows'][len(pattern_data['lows'])//2]
            surrounding_min = min(np.min(pattern_data['lows'][:len(pattern_data['lows'])//2]),
                                np.min(pattern_data['lows'][len(pattern_data['lows'])//2+1:]))
            strength = (surrounding_min - current_price) / surrounding_min
        
        return max(0, min(1, strength * 10))  # Normalize to 0-1
    
    def _calculate_fractal_confidence(self, pattern_data: Dict, fractal_type: str) -> float:
        """Calculate confidence in fractal pattern."""
        # Base confidence on price consistency
        closes = pattern_data['closes']
        price_consistency = 1 - (np.std(closes) / np.mean(closes))
        
        # Adjust for fractal type
        if fractal_type == 'high':
            # Higher confidence if price is consistently high
            confidence = price_consistency * 0.8
        else:
            # Higher confidence if price is consistently low
            confidence = price_consistency * 0.8
        
        return max(0, min(1, confidence))
    
    def _identify_fractal_pattern(self, pattern_data: Dict, fractal_type: str) -> str:
        """Identify the type of fractal pattern."""
        closes = pattern_data['closes']
        
        # Simple pattern identification
        if len(closes) >= 3:
            if closes[0] < closes[1] < closes[2]:
                return "ascending_triangle" if fractal_type == 'high' else "descending_triangle"
            elif closes[0] > closes[1] > closes[2]:
                return "descending_triangle" if fractal_type == 'high' else "ascending_triangle"
            else:
                return "symmetrical_triangle"
        
        return "simple_fractal"
    
    def _calculate_timeframe_consensus(self, fractals: List[Dict]) -> Dict:
        """Calculate consensus across timeframes."""
        if not fractals:
            return {"consensus": 0, "direction": "neutral"}
        
        # Count bullish vs bearish signals
        bullish_signals = sum(1 for f in fractals if f['type'] == 'fractal_low')
        bearish_signals = sum(1 for f in fractals if f['type'] == 'fractal_high')
        
        total_signals = len(fractals)
        if total_signals == 0:
            return {"consensus": 0, "direction": "neutral"}
        
        if bullish_signals > bearish_signals:
            consensus = bullish_signals / total_signals
            direction = "bullish"
        elif bearish_signals > bullish_signals:
            consensus = bearish_signals / total_signals
            direction = "bearish"
        else:
            consensus = 0.5
            direction = "neutral"
        
        return {"consensus": consensus, "direction": direction}
    
    def _calculate_pattern_strength(self, fractals: List[Dict]) -> float:
        """Calculate overall pattern strength."""
        if not fractals:
            return 0.0
        
        # Weight by confidence and strength
        weighted_strength = sum(f['strength'] * f['confidence'] for f in fractals)
        total_weight = sum(f['confidence'] for f in fractals)
        
        return weighted_strength / total_weight if total_weight > 0 else 0.0
    
    async def make_trading_decision(self, market_data: Dict[str, Any]) -> Optional[TradeDecision]:
        """Make trading decisions based on fractal analysis."""
        analysis = await self.analyze_market_data(market_data)
        
        # Look for strong fractal opportunities
        strong_fractals = [f for f in analysis["fractal_signals"] if f['strength'] > 0.7]
        
        if not strong_fractals:
            return None
        
        # Find the best fractal signal
        best_fractal = max(strong_fractals, key=lambda f: f['strength'] * f['confidence'])
        
        # Check timeframe consensus
        symbol = best_fractal['symbol']
        consensus = analysis["timeframe_consensus"].get(symbol, {})
        
        if consensus.get("consensus", 0) < 0.6:
            return None
        
        # Create trading decision
        return await self._create_fractal_decision(market_data, best_fractal, consensus)
    
    async def _create_fractal_decision(self, market_data: Dict[str, Any], fractal: Dict, consensus: Dict) -> TradeDecision:
        """Create a trading decision based on fractal analysis."""
        symbol = fractal['symbol']
        data = market_data["price_data"][symbol]
        current_price = float(data[-1]["close"])
        
        # Determine action based on fractal type and consensus
        if fractal['type'] == 'fractal_low' and consensus['direction'] == 'bullish':
            action = "BUY"
            confidence = fractal['confidence'] * consensus['consensus']
        elif fractal['type'] == 'fractal_high' and consensus['direction'] == 'bearish':
            action = "SELL"
            confidence = fractal['confidence'] * consensus['consensus']
        else:
            return None
        
        # Position sizing based on fractal strength
        strength_multiplier = fractal['strength']
        position_value = self.current_capital * self.position_limit * strength_multiplier
        quantity = position_value / current_price
        
        return TradeDecision(
            symbol=symbol,
            action=action,
            quantity=quantity,
            price=current_price,
            confidence=confidence,
            reasoning=f"Fractal {action.lower()}: {fractal['pattern']} pattern, strength: {fractal['strength']:.2f}, consensus: {consensus['consensus']:.2f}",
            timestamp=datetime.now(),
            agent_id=self.agent_id
        )
    
    async def update_strategy(self, performance_feedback: Dict[str, Any]) -> None:
        """Update fractal analysis strategy."""
        # Adjust fractal parameters based on performance
        if performance_feedback.get("win_rate", 0) < 0.6:
            self.min_fractal_strength += 0.005  # Require stronger fractals
            self.confidence_threshold += 0.05
        elif performance_feedback.get("win_rate", 0) > 0.75:
            self.min_fractal_strength = max(0.01, self.min_fractal_strength - 0.002)
            self.confidence_threshold = max(0.5, self.confidence_threshold - 0.02)
        
        # Adjust position sizing
        if performance_feedback.get("max_drawdown", 0) > 0.06:
            self.position_limit *= 0.9
        elif performance_feedback.get("sharpe_ratio", 0) > 1.8:
            self.position_limit = min(0.15, self.position_limit * 1.05)


class CandleRangeTheoryAgent(BaseTradingAgent):
    """
    Advanced trading agent using Candle Range Theory with multi-timeframe confirmation.
    Analyzes candle ranges, body-to-wick ratios, and multi-timeframe patterns.
    """
    
    def __init__(self, config: AgentConfig):
        super().__init__(config)
        self.risk_threshold = 0.035  # 3.5% max risk per trade
        self.confidence_threshold = 0.7  # High confidence required
        self.position_limit = 0.1  # Max 10% of portfolio per position
        self.stop_loss_pct = 0.035  # 3.5% stop loss
        self.take_profit_pct = 0.07  # 7% take profit
        
        # Candle Range Theory parameters
        self.min_body_ratio = 0.3  # Minimum body-to-range ratio
        self.max_wick_ratio = 0.4  # Maximum wick-to-range ratio
        self.confirmation_timeframes = ['5min', '15min', '1hour', '4hour']
        self.timeframe_weights = {'5min': 0.2, '15min': 0.3, '1hour': 0.3, '4hour': 0.2}
        
        # Pattern recognition
        self.pattern_confidence_threshold = 0.75
        self.volume_confirmation_required = True
        
    async def analyze_market_data(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze market data using Candle Range Theory."""
        analysis = {
            "candle_patterns": [],
            "range_analysis": {},
            "multi_timeframe_signals": {},
            "volume_confirmation": {},
            "opportunities": [],
            "risk_level": "medium"
        }
        
        for symbol, data in market_data.get("price_data", {}).items():
            if len(data) < 20:
                continue
                
            # Extract OHLCV data
            ohlcv_data = self._extract_ohlcv(data)
            
            # Analyze candle patterns
            candle_patterns = await self._analyze_candle_patterns(ohlcv_data, symbol)
            analysis["candle_patterns"].extend(candle_patterns)
            
            # Range analysis
            range_analysis = self._analyze_candle_ranges(ohlcv_data)
            analysis["range_analysis"][symbol] = range_analysis
            
            # Multi-timeframe confirmation
            mtf_signals = self._analyze_multi_timeframe(ohlcv_data, symbol)
            analysis["multi_timeframe_signals"][symbol] = mtf_signals
            
            # Volume confirmation
            volume_confirmation = self._analyze_volume_patterns(ohlcv_data)
            analysis["volume_confirmation"][symbol] = volume_confirmation
            
            # Identify opportunities
            opportunities = self._identify_candle_opportunities(
                candle_patterns, range_analysis, mtf_signals, volume_confirmation
            )
            analysis["opportunities"].extend(opportunities)
        
        return analysis
    
    def _extract_ohlcv(self, data: List[Dict]) -> Dict[str, List[float]]:
        """Extract OHLCV data from market data."""
        return {
            'open': [float(d.get('open', d.get('close', 0))) for d in data],
            'high': [float(d.get('high', d.get('close', 0))) for d in data],
            'low': [float(d.get('low', d.get('close', 0))) for d in data],
            'close': [float(d.get('close', 0)) for d in data],
            'volume': [float(d.get('volume', 1000)) for d in data]
        }
    
    async def _analyze_candle_patterns(self, ohlcv_data: Dict[str, List[float]], symbol: str) -> List[Dict]:
        """Analyze candlestick patterns using Range Theory."""
        patterns = []
        
        opens = np.array(ohlcv_data['open'])
        highs = np.array(ohlcv_data['high'])
        lows = np.array(ohlcv_data['low'])
        closes = np.array(ohlcv_data['close'])
        volumes = np.array(ohlcv_data['volume'])
        
        for i in range(1, len(opens)):
            # Calculate candle metrics
            candle_range = highs[i] - lows[i]
            body_size = abs(closes[i] - opens[i])
            upper_wick = highs[i] - max(opens[i], closes[i])
            lower_wick = min(opens[i], closes[i]) - lows[i]
            
            if candle_range == 0:
                continue
            
            # Calculate ratios
            body_ratio = body_size / candle_range
            upper_wick_ratio = upper_wick / candle_range
            lower_wick_ratio = lower_wick / candle_range
            
            # Identify candle type
            candle_type = self._identify_candle_type(opens[i], closes[i], body_ratio)
            
            # Analyze pattern strength
            pattern_strength = self._calculate_pattern_strength(
                body_ratio, upper_wick_ratio, lower_wick_ratio, volumes[i]
            )
            
            # Multi-candle pattern analysis
            if i >= 2:
                multi_pattern = self._analyze_multi_candle_pattern(
                    opens[i-2:i+1], highs[i-2:i+1], lows[i-2:i+1], closes[i-2:i+1]
                )
                
                if multi_pattern and pattern_strength > 0.6:
                    patterns.append({
                        'symbol': symbol,
                        'index': i,
                        'candle_type': candle_type,
                        'pattern': multi_pattern,
                        'strength': pattern_strength,
                        'body_ratio': body_ratio,
                        'upper_wick_ratio': upper_wick_ratio,
                        'lower_wick_ratio': lower_wick_ratio,
                        'volume': volumes[i],
                        'confidence': self._calculate_pattern_confidence(
                            pattern_strength, volumes[i], candle_type
                        )
                    })
        
        return patterns
    
    def _identify_candle_type(self, open_price: float, close_price: float, body_ratio: float) -> str:
        """Identify the type of candlestick."""
        if body_ratio < 0.1:
            return "doji"
        elif body_ratio < 0.3:
            return "spinning_top"
        elif close_price > open_price:
            if body_ratio > 0.7:
                return "strong_bullish"
            else:
                return "bullish"
        else:
            if body_ratio > 0.7:
                return "strong_bearish"
            else:
                return "bearish"
    
    def _calculate_pattern_strength(self, body_ratio: float, upper_wick_ratio: float, 
                                  lower_wick_ratio: float, volume: float) -> float:
        """Calculate the strength of a candle pattern."""
        # Base strength from body ratio
        body_strength = body_ratio * 0.4
        
        # Wick analysis (lower wicks are generally more significant)
        wick_strength = (1 - upper_wick_ratio) * 0.3 + (1 - lower_wick_ratio) * 0.3
        
        # Volume confirmation
        volume_strength = min(1.0, volume / 1000000) * 0.3  # Normalize volume
        
        return body_strength + wick_strength + volume_strength
    
    def _analyze_multi_candle_pattern(self, opens: np.ndarray, highs: np.ndarray, 
                                    lows: np.ndarray, closes: np.ndarray) -> Optional[str]:
        """Analyze multi-candle patterns."""
        if len(opens) < 3:
            return None
        
        # Engulfing patterns
        if self._is_bullish_engulfing(opens, closes):
            return "bullish_engulfing"
        elif self._is_bearish_engulfing(opens, closes):
            return "bearish_engulfing"
        
        # Hammer and shooting star
        if self._is_hammer(opens, highs, lows, closes):
            return "hammer"
        elif self._is_shooting_star(opens, highs, lows, closes):
            return "shooting_star"
        
        # Three candle patterns
        if self._is_morning_star(opens, closes):
            return "morning_star"
        elif self._is_evening_star(opens, closes):
            return "evening_star"
        
        return None
    
    def _is_bullish_engulfing(self, opens: np.ndarray, closes: np.ndarray) -> bool:
        """Check for bullish engulfing pattern."""
        if len(opens) < 2:
            return False
        
        # First candle is bearish, second is bullish and engulfs first
        first_bearish = closes[0] < opens[0]
        second_bullish = closes[1] > opens[1]
        engulfing = opens[1] < closes[0] and closes[1] > opens[0]
        
        return first_bearish and second_bullish and engulfing
    
    def _is_bearish_engulfing(self, opens: np.ndarray, closes: np.ndarray) -> bool:
        """Check for bearish engulfing pattern."""
        if len(opens) < 2:
            return False
        
        # First candle is bullish, second is bearish and engulfs first
        first_bullish = closes[0] > opens[0]
        second_bearish = closes[1] < opens[1]
        engulfing = opens[1] > closes[0] and closes[1] < opens[0]
        
        return first_bullish and second_bearish and engulfing
    
    def _is_hammer(self, opens: np.ndarray, highs: np.ndarray, 
                   lows: np.ndarray, closes: np.ndarray) -> bool:
        """Check for hammer pattern."""
        if len(opens) < 1:
            return False
        
        candle_range = highs[0] - lows[0]
        body_size = abs(closes[0] - opens[0])
        lower_wick = min(opens[0], closes[0]) - lows[0]
        upper_wick = highs[0] - max(opens[0], closes[0])
        
        # Hammer: small body, long lower wick, small upper wick
        return (body_size < candle_range * 0.3 and 
                lower_wick > candle_range * 0.6 and 
                upper_wick < candle_range * 0.1)
    
    def _is_shooting_star(self, opens: np.ndarray, highs: np.ndarray, 
                         lows: np.ndarray, closes: np.ndarray) -> bool:
        """Check for shooting star pattern."""
        if len(opens) < 1:
            return False
        
        candle_range = highs[0] - lows[0]
        body_size = abs(closes[0] - opens[0])
        lower_wick = min(opens[0], closes[0]) - lows[0]
        upper_wick = highs[0] - max(opens[0], closes[0])
        
        # Shooting star: small body, long upper wick, small lower wick
        return (body_size < candle_range * 0.3 and 
                upper_wick > candle_range * 0.6 and 
                lower_wick < candle_range * 0.1)
    
    def _is_morning_star(self, opens: np.ndarray, closes: np.ndarray) -> bool:
        """Check for morning star pattern."""
        if len(opens) < 3:
            return False
        
        # First candle is bearish, second is small (doji-like), third is bullish
        first_bearish = closes[0] < opens[0]
        second_small = abs(closes[1] - opens[1]) < abs(closes[0] - opens[0]) * 0.3
        third_bullish = closes[2] > opens[2]
        gap_down = opens[1] < closes[0]
        gap_up = opens[2] > closes[1]
        
        return first_bearish and second_small and third_bullish and gap_down and gap_up
    
    def _is_evening_star(self, opens: np.ndarray, closes: np.ndarray) -> bool:
        """Check for evening star pattern."""
        if len(opens) < 3:
            return False
        
        # First candle is bullish, second is small (doji-like), third is bearish
        first_bullish = closes[0] > opens[0]
        second_small = abs(closes[1] - opens[1]) < abs(closes[0] - opens[0]) * 0.3
        third_bearish = closes[2] < opens[2]
        gap_up = opens[1] > closes[0]
        gap_down = opens[2] < closes[1]
        
        return first_bullish and second_small and third_bearish and gap_up and gap_down
    
    def _calculate_pattern_confidence(self, pattern_strength: float, volume: float, candle_type: str) -> float:
        """Calculate confidence in the pattern."""
        base_confidence = pattern_strength
        
        # Volume confirmation
        volume_factor = min(1.0, volume / 1000000) * 0.3
        
        # Candle type factor
        type_factors = {
            "strong_bullish": 0.9,
            "strong_bearish": 0.9,
            "bullish": 0.7,
            "bearish": 0.7,
            "spinning_top": 0.5,
            "doji": 0.3
        }
        type_factor = type_factors.get(candle_type, 0.6) * 0.4
        
        return min(1.0, base_confidence + volume_factor + type_factor)
    
    def _analyze_candle_ranges(self, ohlcv_data: Dict[str, List[float]]) -> Dict:
        """Analyze candle range characteristics."""
        highs = np.array(ohlcv_data['high'])
        lows = np.array(ohlcv_data['low'])
        opens = np.array(ohlcv_data['open'])
        closes = np.array(ohlcv_data['close'])
        
        # Calculate ranges
        ranges = highs - lows
        avg_range = np.mean(ranges[-10:])  # Average of last 10 candles
        current_range = ranges[-1]
        
        # Range expansion/contraction
        range_expansion = current_range / avg_range if avg_range > 0 else 1.0
        
        # Volatility analysis
        range_volatility = np.std(ranges[-10:]) / np.mean(ranges[-10:]) if np.mean(ranges[-10:]) > 0 else 0
        
        return {
            "current_range": current_range,
            "avg_range": avg_range,
            "range_expansion": range_expansion,
            "range_volatility": range_volatility,
            "range_trend": "expanding" if range_expansion > 1.2 else "contracting" if range_expansion < 0.8 else "stable"
        }
    
    def _analyze_multi_timeframe(self, ohlcv_data: Dict[str, List[float]], symbol: str) -> Dict:
        """Analyze multi-timeframe confirmation."""
        # Simulate multi-timeframe analysis with current data
        # In a real implementation, you would fetch data from different timeframes
        
        highs = np.array(ohlcv_data['high'])
        lows = np.array(ohlcv_data['low'])
        closes = np.array(ohlcv_data['close'])
        
        # Calculate trend across different "timeframes" (using different lookback periods)
        short_trend = self._calculate_trend(closes[-5:])  # 5-period trend
        medium_trend = self._calculate_trend(closes[-10:])  # 10-period trend
        long_trend = self._calculate_trend(closes[-20:])  # 20-period trend
        
        # Multi-timeframe consensus
        trends = [short_trend, medium_trend, long_trend]
        bullish_count = sum(1 for t in trends if t > 0)
        bearish_count = sum(1 for t in trends if t < 0)
        
        if bullish_count > bearish_count:
            consensus = "bullish"
            strength = bullish_count / len(trends)
        elif bearish_count > bullish_count:
            consensus = "bearish"
            strength = bearish_count / len(trends)
        else:
            consensus = "neutral"
            strength = 0.5
        
        return {
            "short_trend": short_trend,
            "medium_trend": medium_trend,
            "long_trend": long_trend,
            "consensus": consensus,
            "strength": strength
        }
    
    def _calculate_trend(self, prices: np.ndarray) -> float:
        """Calculate trend strength."""
        if len(prices) < 2:
            return 0.0
        
        # Simple linear regression slope
        x = np.arange(len(prices))
        slope, _, _, _, _ = stats.linregress(x, prices)
        
        return slope / prices[0] if prices[0] > 0 else 0.0  # Normalize by initial price
    
    def _analyze_volume_patterns(self, ohlcv_data: Dict[str, List[float]]) -> Dict:
        """Analyze volume patterns for confirmation."""
        volumes = np.array(ohlcv_data['volume'])
        
        if len(volumes) < 10:
            return {"volume_trend": "neutral", "volume_confirmation": False}
        
        # Volume trend
        avg_volume = np.mean(volumes[-10:])
        current_volume = volumes[-1]
        volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1.0
        
        # Volume confirmation (above average volume)
        volume_confirmation = volume_ratio > 1.2
        
        return {
            "volume_trend": "increasing" if volume_ratio > 1.1 else "decreasing" if volume_ratio < 0.9 else "stable",
            "volume_confirmation": volume_confirmation,
            "volume_ratio": volume_ratio
        }
    
    def _identify_candle_opportunities(self, patterns: List[Dict], range_analysis: Dict, 
                                     mtf_signals: Dict, volume_confirmation: Dict) -> List[str]:
        """Identify trading opportunities based on candle analysis."""
        opportunities = []
        
        if not patterns:
            return opportunities
        
        # Get the most recent pattern
        latest_pattern = max(patterns, key=lambda p: p['index'])
        
        # Check for high-confidence patterns
        if latest_pattern['confidence'] > self.pattern_confidence_threshold:
            pattern_type = latest_pattern['pattern']
            symbol = latest_pattern['symbol']
            
            # Check multi-timeframe confirmation
            mtf_consensus = mtf_signals.get('consensus', 'neutral')
            mtf_strength = mtf_signals.get('strength', 0)
            
            # Check volume confirmation
            volume_confirm = volume_confirmation.get('volume_confirmation', False)
            
            # Identify opportunities based on pattern and confirmations
            if pattern_type in ['bullish_engulfing', 'hammer', 'morning_star']:
                if mtf_consensus == 'bullish' and mtf_strength > 0.6:
                    if not self.volume_confirmation_required or volume_confirm:
                        opportunities.append(f"{symbol}_bullish_candle")
            
            elif pattern_type in ['bearish_engulfing', 'shooting_star', 'evening_star']:
                if mtf_consensus == 'bearish' and mtf_strength > 0.6:
                    if not self.volume_confirmation_required or volume_confirm:
                        opportunities.append(f"{symbol}_bearish_candle")
        
        return opportunities
    
    async def make_trading_decision(self, market_data: Dict[str, Any]) -> Optional[TradeDecision]:
        """Make trading decisions based on Candle Range Theory."""
        analysis = await self.analyze_market_data(market_data)
        
        # Look for high-confidence opportunities
        high_confidence_patterns = [p for p in analysis["candle_patterns"] 
                                  if p['confidence'] > self.pattern_confidence_threshold]
        
        if not high_confidence_patterns:
            return None
        
        # Find the best opportunity
        best_pattern = max(high_confidence_patterns, key=lambda p: p['confidence'])
        
        # Check multi-timeframe and volume confirmation
        symbol = best_pattern['symbol']
        mtf_signals = analysis["multi_timeframe_signals"].get(symbol, {})
        volume_confirmation = analysis["volume_confirmation"].get(symbol, {})
        
        # Create trading decision
        return await self._create_candle_decision(market_data, best_pattern, mtf_signals, volume_confirmation)
    
    async def _create_candle_decision(self, market_data: Dict[str, Any], pattern: Dict, 
                                    mtf_signals: Dict, volume_confirmation: Dict) -> TradeDecision:
        """Create a trading decision based on candle analysis."""
        symbol = pattern['symbol']
        data = market_data["price_data"][symbol]
        current_price = float(data[-1]["close"])
        
        # Determine action based on pattern
        pattern_type = pattern['pattern']
        if pattern_type in ['bullish_engulfing', 'hammer', 'morning_star']:
            action = "BUY"
        elif pattern_type in ['bearish_engulfing', 'shooting_star', 'evening_star']:
            action = "SELL"
        else:
            return None
        
        # Calculate confidence with confirmations
        base_confidence = pattern['confidence']
        mtf_factor = mtf_signals.get('strength', 0.5)
        volume_factor = 1.2 if volume_confirmation.get('volume_confirmation', False) else 0.8
        
        final_confidence = base_confidence * mtf_factor * volume_factor
        
        # Position sizing
        position_value = self.current_capital * self.position_limit
        quantity = position_value / current_price
        
        return TradeDecision(
            symbol=symbol,
            action=action,
            quantity=quantity,
            price=current_price,
            confidence=final_confidence,
            reasoning=f"Candle Range Theory {action.lower()}: {pattern_type}, confidence: {final_confidence:.2f}, MTF: {mtf_signals.get('consensus', 'neutral')}",
            timestamp=datetime.now(),
            agent_id=self.agent_id
        )
    
    async def update_strategy(self, performance_feedback: Dict[str, Any]) -> None:
        """Update Candle Range Theory strategy."""
        # Adjust pattern confidence threshold
        if performance_feedback.get("win_rate", 0) < 0.65:
            self.pattern_confidence_threshold += 0.05
        elif performance_feedback.get("win_rate", 0) > 0.8:
            self.pattern_confidence_threshold = max(0.6, self.pattern_confidence_threshold - 0.02)
        
        # Adjust volume confirmation requirement
        if performance_feedback.get("win_rate", 0) < 0.6:
            self.volume_confirmation_required = True
        elif performance_feedback.get("win_rate", 0) > 0.75:
            self.volume_confirmation_required = False
        
        # Adjust position sizing
        if performance_feedback.get("max_drawdown", 0) > 0.05:
            self.position_limit *= 0.9
        elif performance_feedback.get("sharpe_ratio", 0) > 1.6:
            self.position_limit = min(0.12, self.position_limit * 1.05)


class QuantitativePatternAgent(BaseTradingAgent):
    """
    Advanced trading agent using quantitative pattern recognition and machine learning.
    Uses statistical analysis, pattern matching, and predictive models for trading decisions.
    """
    
    def __init__(self, config: AgentConfig):
        super().__init__(config)
        self.risk_threshold = 0.03  # 3% max risk per trade
        self.confidence_threshold = 0.8  # High confidence required
        self.position_limit = 0.08  # Max 8% of portfolio per position
        self.stop_loss_pct = 0.03  # 3% stop loss
        self.take_profit_pct = 0.06  # 6% take profit
        
        # Quantitative analysis parameters
        self.lookback_periods = [5, 10, 20, 50]  # Multiple lookback periods
        self.pattern_window = 20  # Window for pattern analysis
        self.min_pattern_confidence = 0.75  # Minimum pattern confidence
        self.volatility_threshold = 0.02  # Volatility threshold
        
        # Machine learning components
        self.pattern_classifier = None
        self.price_predictor = None
        self.scaler = StandardScaler()
        self.is_model_trained = False
        
        # Pattern recognition
        self.known_patterns = self._initialize_patterns()
        self.pattern_weights = self._initialize_pattern_weights()
        
    def _initialize_patterns(self) -> Dict[str, Dict]:
        """Initialize known quantitative patterns."""
        return {
            "double_top": {
                "description": "Two peaks at similar price levels",
                "bullish": False,
                "confidence_threshold": 0.8
            },
            "double_bottom": {
                "description": "Two troughs at similar price levels", 
                "bullish": True,
                "confidence_threshold": 0.8
            },
            "head_and_shoulders": {
                "description": "Three peaks with middle peak highest",
                "bullish": False,
                "confidence_threshold": 0.85
            },
            "inverse_head_and_shoulders": {
                "description": "Three troughs with middle trough lowest",
                "bullish": True,
                "confidence_threshold": 0.85
            },
            "ascending_triangle": {
                "description": "Horizontal resistance with rising support",
                "bullish": True,
                "confidence_threshold": 0.75
            },
            "descending_triangle": {
                "description": "Horizontal support with falling resistance",
                "bullish": False,
                "confidence_threshold": 0.75
            },
            "flag_pattern": {
                "description": "Brief consolidation after strong move",
                "bullish": True,  # Depends on prior trend
                "confidence_threshold": 0.7
            },
            "pennant_pattern": {
                "description": "Symmetrical triangle consolidation",
                "bullish": True,  # Depends on prior trend
                "confidence_threshold": 0.7
            }
        }
    
    def _initialize_pattern_weights(self) -> Dict[str, float]:
        """Initialize pattern weights for scoring."""
        return {
            "double_top": 0.9,
            "double_bottom": 0.9,
            "head_and_shoulders": 0.95,
            "inverse_head_and_shoulders": 0.95,
            "ascending_triangle": 0.8,
            "descending_triangle": 0.8,
            "flag_pattern": 0.7,
            "pennant_pattern": 0.7
        }
    
    async def analyze_market_data(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze market data using quantitative pattern recognition."""
        analysis = {
            "quantitative_patterns": [],
            "statistical_signals": {},
            "ml_predictions": {},
            "volatility_analysis": {},
            "opportunities": [],
            "risk_level": "medium"
        }
        
        for symbol, data in market_data.get("price_data", {}).items():
            if len(data) < 50:  # Need sufficient data for quantitative analysis
                continue
                
            # Extract OHLCV data
            ohlcv_data = self._extract_ohlcv(data)
            
            # Pattern recognition
            patterns = await self._recognize_patterns(ohlcv_data, symbol)
            analysis["quantitative_patterns"].extend(patterns)
            
            # Statistical analysis
            stats_signals = self._perform_statistical_analysis(ohlcv_data)
            analysis["statistical_signals"][symbol] = stats_signals
            
            # Machine learning predictions
            ml_predictions = await self._generate_ml_predictions(ohlcv_data, symbol)
            analysis["ml_predictions"][symbol] = ml_predictions
            
            # Volatility analysis
            volatility_analysis = self._analyze_volatility(ohlcv_data)
            analysis["volatility_analysis"][symbol] = volatility_analysis
            
            # Identify opportunities
            opportunities = self._identify_quantitative_opportunities(
                patterns, stats_signals, ml_predictions, volatility_analysis
            )
            analysis["opportunities"].extend(opportunities)
        
        return analysis
    
    def _extract_ohlcv(self, data: List[Dict]) -> Dict[str, List[float]]:
        """Extract OHLCV data from market data."""
        return {
            'open': [float(d.get('open', d.get('close', 0))) for d in data],
            'high': [float(d.get('high', d.get('close', 0))) for d in data],
            'low': [float(d.get('low', d.get('close', 0))) for d in data],
            'close': [float(d.get('close', 0)) for d in data],
            'volume': [float(d.get('volume', 1000)) for d in data]
        }
    
    async def _recognize_patterns(self, ohlcv_data: Dict[str, List[float]], symbol: str) -> List[Dict]:
        """Recognize quantitative patterns in price data."""
        patterns = []
        
        highs = np.array(ohlcv_data['high'])
        lows = np.array(ohlcv_data['low'])
        closes = np.array(ohlcv_data['close'])
        volumes = np.array(ohlcv_data['volume'])
        
        # Analyze each known pattern
        for pattern_name, pattern_info in self.known_patterns.items():
            pattern_result = await self._detect_pattern(
                pattern_name, highs, lows, closes, volumes, symbol
            )
            
            if pattern_result and pattern_result['confidence'] >= pattern_info['confidence_threshold']:
                patterns.append(pattern_result)
        
        return patterns
    
    async def _detect_pattern(self, pattern_name: str, highs: np.ndarray, lows: np.ndarray, 
                            closes: np.ndarray, volumes: np.ndarray, symbol: str) -> Optional[Dict]:
        """Detect a specific pattern in the data."""
        if pattern_name == "double_top":
            return self._detect_double_top(highs, lows, closes, volumes, symbol)
        elif pattern_name == "double_bottom":
            return self._detect_double_bottom(highs, lows, closes, volumes, symbol)
        elif pattern_name == "head_and_shoulders":
            return self._detect_head_and_shoulders(highs, lows, closes, volumes, symbol)
        elif pattern_name == "inverse_head_and_shoulders":
            return self._detect_inverse_head_and_shoulders(highs, lows, closes, volumes, symbol)
        elif pattern_name == "ascending_triangle":
            return self._detect_ascending_triangle(highs, lows, closes, volumes, symbol)
        elif pattern_name == "descending_triangle":
            return self._detect_descending_triangle(highs, lows, closes, volumes, symbol)
        elif pattern_name == "flag_pattern":
            return self._detect_flag_pattern(highs, lows, closes, volumes, symbol)
        elif pattern_name == "pennant_pattern":
            return self._detect_pennant_pattern(highs, lows, closes, volumes, symbol)
        
        return None
    
    def _detect_double_top(self, highs: np.ndarray, lows: np.ndarray, closes: np.ndarray, 
                          volumes: np.ndarray, symbol: str) -> Optional[Dict]:
        """Detect double top pattern."""
        if len(highs) < 20:
            return None
        
        # Find local maxima
        peaks = self._find_peaks(highs, min_distance=5)
        
        if len(peaks) < 2:
            return None
        
        # Check for two similar peaks
        recent_peaks = peaks[-2:]
        peak1_price = highs[recent_peaks[0]]
        peak2_price = highs[recent_peaks[1]]
        
        # Price similarity (within 2%)
        price_similarity = 1 - abs(peak1_price - peak2_price) / max(peak1_price, peak2_price)
        
        if price_similarity > 0.98:  # Very similar prices
            # Check volume pattern (second peak should have lower volume)
            vol1 = volumes[recent_peaks[0]]
            vol2 = volumes[recent_peaks[1]]
            volume_decline = vol2 < vol1 * 0.8
            
            confidence = price_similarity * 0.7 + (0.3 if volume_decline else 0.1)
            
            return {
                'symbol': symbol,
                'pattern': 'double_top',
                'confidence': confidence,
                'entry_price': closes[-1],
                'target_price': lows[recent_peaks[0]:recent_peaks[1]].min(),
                'stop_loss': peak2_price * 1.02,
                'pattern_strength': confidence,
                'volume_confirmation': volume_decline
            }
        
        return None
    
    def _detect_double_bottom(self, highs: np.ndarray, lows: np.ndarray, closes: np.ndarray, 
                            volumes: np.ndarray, symbol: str) -> Optional[Dict]:
        """Detect double bottom pattern."""
        if len(lows) < 20:
            return None
        
        # Find local minima
        troughs = self._find_troughs(lows, min_distance=5)
        
        if len(troughs) < 2:
            return None
        
        # Check for two similar troughs
        recent_troughs = troughs[-2:]
        trough1_price = lows[recent_troughs[0]]
        trough2_price = lows[recent_troughs[1]]
        
        # Price similarity (within 2%)
        price_similarity = 1 - abs(trough1_price - trough2_price) / max(trough1_price, trough2_price)
        
        if price_similarity > 0.98:  # Very similar prices
            # Check volume pattern (second trough should have lower volume)
            vol1 = volumes[recent_troughs[0]]
            vol2 = volumes[recent_troughs[1]]
            volume_decline = vol2 < vol1 * 0.8
            
            confidence = price_similarity * 0.7 + (0.3 if volume_decline else 0.1)
            
            return {
                'symbol': symbol,
                'pattern': 'double_bottom',
                'confidence': confidence,
                'entry_price': closes[-1],
                'target_price': highs[recent_troughs[0]:recent_troughs[1]].max(),
                'stop_loss': trough2_price * 0.98,
                'pattern_strength': confidence,
                'volume_confirmation': volume_decline
            }
        
        return None
    
    def _detect_head_and_shoulders(self, highs: np.ndarray, lows: np.ndarray, closes: np.ndarray, 
                                 volumes: np.ndarray, symbol: str) -> Optional[Dict]:
        """Detect head and shoulders pattern."""
        if len(highs) < 30:
            return None
        
        # Find three peaks
        peaks = self._find_peaks(highs, min_distance=8)
        
        if len(peaks) < 3:
            return None
        
        recent_peaks = peaks[-3:]
        left_shoulder = highs[recent_peaks[0]]
        head = highs[recent_peaks[1]]
        right_shoulder = highs[recent_peaks[2]]
        
        # Head should be higher than shoulders
        head_higher = head > left_shoulder and head > right_shoulder
        
        # Shoulders should be similar height
        shoulder_similarity = 1 - abs(left_shoulder - right_shoulder) / max(left_shoulder, right_shoulder)
        
        if head_higher and shoulder_similarity > 0.95:
            # Volume should decline from left to right
            vol_pattern = volumes[recent_peaks[0]] > volumes[recent_peaks[1]] > volumes[recent_peaks[2]]
            
            confidence = shoulder_similarity * 0.6 + (0.4 if vol_pattern else 0.1)
            
            # Calculate neckline (lowest point between shoulders)
            neckline = lows[recent_peaks[0]:recent_peaks[2]].min()
            target_price = neckline - (head - neckline)
            
            return {
                'symbol': symbol,
                'pattern': 'head_and_shoulders',
                'confidence': confidence,
                'entry_price': closes[-1],
                'target_price': target_price,
                'stop_loss': head * 1.02,
                'pattern_strength': confidence,
                'volume_confirmation': vol_pattern
            }
        
        return None
    
    def _detect_inverse_head_and_shoulders(self, highs: np.ndarray, lows: np.ndarray, closes: np.ndarray, 
                                         volumes: np.ndarray, symbol: str) -> Optional[Dict]:
        """Detect inverse head and shoulders pattern."""
        if len(lows) < 30:
            return None
        
        # Find three troughs
        troughs = self._find_troughs(lows, min_distance=8)
        
        if len(troughs) < 3:
            return None
        
        recent_troughs = troughs[-3:]
        left_shoulder = lows[recent_troughs[0]]
        head = lows[recent_troughs[1]]
        right_shoulder = lows[recent_troughs[2]]
        
        # Head should be lower than shoulders
        head_lower = head < left_shoulder and head < right_shoulder
        
        # Shoulders should be similar height
        shoulder_similarity = 1 - abs(left_shoulder - right_shoulder) / max(left_shoulder, right_shoulder)
        
        if head_lower and shoulder_similarity > 0.95:
            # Volume should decline from left to right
            vol_pattern = volumes[recent_troughs[0]] > volumes[recent_troughs[1]] > volumes[recent_troughs[2]]
            
            confidence = shoulder_similarity * 0.6 + (0.4 if vol_pattern else 0.1)
            
            # Calculate neckline (highest point between shoulders)
            neckline = highs[recent_troughs[0]:recent_troughs[2]].max()
            target_price = neckline + (neckline - head)
            
            return {
                'symbol': symbol,
                'pattern': 'inverse_head_and_shoulders',
                'confidence': confidence,
                'entry_price': closes[-1],
                'target_price': target_price,
                'stop_loss': head * 0.98,
                'pattern_strength': confidence,
                'volume_confirmation': vol_pattern
            }
        
        return None
    
    def _detect_ascending_triangle(self, highs: np.ndarray, lows: np.ndarray, closes: np.ndarray, 
                                 volumes: np.ndarray, symbol: str) -> Optional[Dict]:
        """Detect ascending triangle pattern."""
        if len(highs) < 20:
            return None
        
        # Look for horizontal resistance and rising support
        recent_highs = highs[-20:]
        recent_lows = lows[-20:]
        
        # Check for horizontal resistance
        resistance_level = np.mean(recent_highs)
        resistance_consistency = 1 - (np.std(recent_highs) / resistance_level)
        
        # Check for rising support
        support_trend = self._calculate_trend(recent_lows)
        
        if resistance_consistency > 0.9 and support_trend > 0:
            confidence = resistance_consistency * 0.7 + min(1.0, support_trend * 100) * 0.3
            
            return {
                'symbol': symbol,
                'pattern': 'ascending_triangle',
                'confidence': confidence,
                'entry_price': closes[-1],
                'target_price': resistance_level * 1.05,
                'stop_loss': recent_lows.min() * 0.98,
                'pattern_strength': confidence,
                'volume_confirmation': True
            }
        
        return None
    
    def _detect_descending_triangle(self, highs: np.ndarray, lows: np.ndarray, closes: np.ndarray, 
                                  volumes: np.ndarray, symbol: str) -> Optional[Dict]:
        """Detect descending triangle pattern."""
        if len(highs) < 20:
            return None
        
        # Look for horizontal support and falling resistance
        recent_highs = highs[-20:]
        recent_lows = lows[-20:]
        
        # Check for horizontal support
        support_level = np.mean(recent_lows)
        support_consistency = 1 - (np.std(recent_lows) / support_level)
        
        # Check for falling resistance
        resistance_trend = self._calculate_trend(recent_highs)
        
        if support_consistency > 0.9 and resistance_trend < 0:
            confidence = support_consistency * 0.7 + min(1.0, abs(resistance_trend) * 100) * 0.3
            
            return {
                'symbol': symbol,
                'pattern': 'descending_triangle',
                'confidence': confidence,
                'entry_price': closes[-1],
                'target_price': support_level * 0.95,
                'stop_loss': recent_highs.max() * 1.02,
                'pattern_strength': confidence,
                'volume_confirmation': True
            }
        
        return None
    
    def _detect_flag_pattern(self, highs: np.ndarray, lows: np.ndarray, closes: np.ndarray, 
                           volumes: np.ndarray, symbol: str) -> Optional[Dict]:
        """Detect flag pattern."""
        if len(highs) < 15:
            return None
        
        # Look for strong move followed by consolidation
        recent_closes = closes[-15:]
        recent_volumes = volumes[-15:]
        
        # Check for strong initial move
        initial_move = (recent_closes[5] - recent_closes[0]) / recent_closes[0]
        
        if abs(initial_move) > 0.03:  # At least 3% move
            # Check for consolidation (low volatility)
            consolidation_volatility = np.std(recent_closes[5:]) / np.mean(recent_closes[5:])
            
            if consolidation_volatility < 0.01:  # Low volatility consolidation
                confidence = min(1.0, abs(initial_move) * 10) * 0.8
                
                # Determine direction based on initial move
                is_bullish = initial_move > 0
                target_price = recent_closes[-1] * (1.02 if is_bullish else 0.98)
                stop_loss = recent_closes[-1] * (0.98 if is_bullish else 1.02)
                
                return {
                    'symbol': symbol,
                    'pattern': 'flag_pattern',
                    'confidence': confidence,
                    'entry_price': recent_closes[-1],
                    'target_price': target_price,
                    'stop_loss': stop_loss,
                    'pattern_strength': confidence,
                    'volume_confirmation': True,
                    'bullish': is_bullish
                }
        
        return None
    
    def _detect_pennant_pattern(self, highs: np.ndarray, lows: np.ndarray, closes: np.ndarray, 
                              volumes: np.ndarray, symbol: str) -> Optional[Dict]:
        """Detect pennant pattern."""
        if len(highs) < 15:
            return None
        
        # Look for converging trend lines
        recent_highs = highs[-15:]
        recent_lows = lows[-15:]
        
        # Calculate trend lines
        high_trend = self._calculate_trend(recent_highs)
        low_trend = self._calculate_trend(recent_lows)
        
        # Pennant: converging trend lines (opposite slopes)
        if (high_trend < 0 and low_trend > 0) or (high_trend > 0 and low_trend < 0):
            convergence_strength = abs(high_trend) + abs(low_trend)
            confidence = min(1.0, convergence_strength * 50) * 0.7
            
            return {
                'symbol': symbol,
                'pattern': 'pennant_pattern',
                'confidence': confidence,
                'entry_price': closes[-1],
                'target_price': closes[-1] * 1.03,  # Conservative target
                'stop_loss': closes[-1] * 0.97,
                'pattern_strength': confidence,
                'volume_confirmation': True
            }
        
        return None
    
    def _find_peaks(self, data: np.ndarray, min_distance: int = 5) -> List[int]:
        """Find local peaks in data."""
        peaks = []
        
        for i in range(min_distance, len(data) - min_distance):
            is_peak = True
            for j in range(i - min_distance, i + min_distance + 1):
                if j != i and data[j] >= data[i]:
                    is_peak = False
                    break
            
            if is_peak:
                peaks.append(i)
        
        return peaks
    
    def _find_troughs(self, data: np.ndarray, min_distance: int = 5) -> List[int]:
        """Find local troughs in data."""
        troughs = []
        
        for i in range(min_distance, len(data) - min_distance):
            is_trough = True
            for j in range(i - min_distance, i + min_distance + 1):
                if j != i and data[j] <= data[i]:
                    is_trough = False
                    break
            
            if is_trough:
                troughs.append(i)
        
        return troughs
    
    def _calculate_trend(self, data: np.ndarray) -> float:
        """Calculate trend using linear regression."""
        if len(data) < 2:
            return 0.0
        
        x = np.arange(len(data))
        slope, _, _, _, _ = stats.linregress(x, data)
        
        return slope / data[0] if data[0] > 0 else 0.0
    
    def _perform_statistical_analysis(self, ohlcv_data: Dict[str, List[float]]) -> Dict:
        """Perform statistical analysis on price data."""
        closes = np.array(ohlcv_data['close'])
        volumes = np.array(ohlcv_data['volume'])
        
        # Calculate returns
        returns = np.diff(closes) / closes[:-1]
        
        # Statistical measures
        mean_return = np.mean(returns)
        std_return = np.std(returns)
        skewness = stats.skew(returns)
        kurtosis = stats.kurtosis(returns)
        
        # Volatility measures
        volatility = std_return * np.sqrt(252)  # Annualized
        
        # Volume analysis
        volume_trend = self._calculate_trend(volumes[-10:])
        
        # Price momentum
        momentum_5 = (closes[-1] - closes[-6]) / closes[-6] if len(closes) >= 6 else 0
        momentum_10 = (closes[-1] - closes[-11]) / closes[-11] if len(closes) >= 11 else 0
        
        return {
            "mean_return": mean_return,
            "volatility": volatility,
            "skewness": skewness,
            "kurtosis": kurtosis,
            "volume_trend": volume_trend,
            "momentum_5": momentum_5,
            "momentum_10": momentum_10,
            "sharpe_ratio": mean_return / std_return if std_return > 0 else 0
        }
    
    async def _generate_ml_predictions(self, ohlcv_data: Dict[str, List[float]], symbol: str) -> Dict:
        """Generate machine learning predictions."""
        # For now, use simple statistical predictions
        # In a full implementation, you would train ML models
        
        closes = np.array(ohlcv_data['close'])
        volumes = np.array(ohlcv_data['volume'])
        
        if len(closes) < 20:
            return {"prediction": 0, "confidence": 0}
        
        # Simple moving average crossover prediction
        sma_5 = np.mean(closes[-5:])
        sma_20 = np.mean(closes[-20:])
        
        # Trend prediction
        if sma_5 > sma_20:
            prediction = 1  # Bullish
            confidence = min(1.0, (sma_5 - sma_20) / sma_20 * 10)
        else:
            prediction = -1  # Bearish
            confidence = min(1.0, (sma_20 - sma_5) / sma_20 * 10)
        
        # Volume confirmation
        recent_volume = np.mean(volumes[-5:])
        avg_volume = np.mean(volumes[-20:])
        volume_factor = recent_volume / avg_volume if avg_volume > 0 else 1.0
        
        confidence *= min(1.2, volume_factor)
        
        return {
            "prediction": prediction,
            "confidence": confidence,
            "sma_5": sma_5,
            "sma_20": sma_20,
            "volume_factor": volume_factor
        }
    
    def _analyze_volatility(self, ohlcv_data: Dict[str, List[float]]) -> Dict:
        """Analyze volatility patterns."""
        closes = np.array(ohlcv_data['close'])
        highs = np.array(ohlcv_data['high'])
        lows = np.array(ohlcv_data['low'])
        
        # Calculate returns
        returns = np.diff(closes) / closes[:-1]
        
        # Volatility measures
        current_volatility = np.std(returns[-10:]) if len(returns) >= 10 else 0
        avg_volatility = np.std(returns) if len(returns) > 0 else 0
        
        # True Range
        tr = np.maximum(highs[1:] - lows[1:], 
                       np.maximum(abs(highs[1:] - closes[:-1]), 
                                 abs(lows[1:] - closes[:-1])))
        atr = np.mean(tr[-14:]) if len(tr) >= 14 else 0
        
        # Volatility regime
        if current_volatility > avg_volatility * 1.5:
            regime = "high"
        elif current_volatility < avg_volatility * 0.5:
            regime = "low"
        else:
            regime = "normal"
        
        return {
            "current_volatility": current_volatility,
            "avg_volatility": avg_volatility,
            "atr": atr,
            "regime": regime,
            "volatility_ratio": current_volatility / avg_volatility if avg_volatility > 0 else 1.0
        }
    
    def _identify_quantitative_opportunities(self, patterns: List[Dict], stats_signals: Dict, 
                                           ml_predictions: Dict, volatility_analysis: Dict) -> List[str]:
        """Identify trading opportunities based on quantitative analysis."""
        opportunities = []
        
        # Pattern-based opportunities
        for pattern in patterns:
            if pattern['confidence'] >= self.min_pattern_confidence:
                symbol = pattern['symbol']
                pattern_type = pattern['pattern']
                
                # Check volatility regime
                volatility_regime = volatility_analysis.get('regime', 'normal')
                
                # Only trade in normal or low volatility regimes
                if volatility_regime in ['normal', 'low']:
                    if pattern_type in ['double_bottom', 'inverse_head_and_shoulders', 'ascending_triangle']:
                        opportunities.append(f"{symbol}_quantitative_bullish")
                    elif pattern_type in ['double_top', 'head_and_shoulders', 'descending_triangle']:
                        opportunities.append(f"{symbol}_quantitative_bearish")
        
        # ML prediction opportunities
        ml_prediction = ml_predictions.get('prediction', 0)
        ml_confidence = ml_predictions.get('confidence', 0)
        
        if abs(ml_prediction) > 0 and ml_confidence > 0.7:
            # Find symbol with best ML prediction (simplified)
            for pattern in patterns:
                symbol = pattern['symbol']
                if ml_prediction > 0:
                    opportunities.append(f"{symbol}_ml_bullish")
                else:
                    opportunities.append(f"{symbol}_ml_bearish")
        
        return opportunities
    
    async def make_trading_decision(self, market_data: Dict[str, Any]) -> Optional[TradeDecision]:
        """Make trading decisions based on quantitative pattern recognition."""
        analysis = await self.analyze_market_data(market_data)
        
        # Look for high-confidence patterns
        high_confidence_patterns = [p for p in analysis["quantitative_patterns"] 
                                  if p['confidence'] >= self.min_pattern_confidence]
        
        if not high_confidence_patterns:
            return None
        
        # Find the best pattern
        best_pattern = max(high_confidence_patterns, key=lambda p: p['confidence'])
        
        # Check volatility regime
        symbol = best_pattern['symbol']
        volatility_analysis = analysis["volatility_analysis"].get(symbol, {})
        
        if volatility_analysis.get('regime') == 'high':
            return None  # Avoid trading in high volatility
        
        # Create trading decision
        return await self._create_quantitative_decision(market_data, best_pattern, analysis)
    
    async def _create_quantitative_decision(self, market_data: Dict[str, Any], pattern: Dict, 
                                          analysis: Dict) -> TradeDecision:
        """Create a trading decision based on quantitative analysis."""
        symbol = pattern['symbol']
        data = market_data["price_data"][symbol]
        current_price = float(data[-1]["close"])
        
        # Determine action based on pattern
        pattern_type = pattern['pattern']
        pattern_info = self.known_patterns.get(pattern_type, {})
        
        if pattern_info.get('bullish', False):
            action = "BUY"
        else:
            action = "SELL"
        
        # Calculate confidence with additional factors
        base_confidence = pattern['confidence']
        
        # ML prediction confirmation
        ml_predictions = analysis["ml_predictions"].get(symbol, {})
        ml_prediction = ml_predictions.get('prediction', 0)
        ml_confidence = ml_predictions.get('confidence', 0)
        
        # Statistical confirmation
        stats_signals = analysis["statistical_signals"].get(symbol, {})
        momentum_5 = stats_signals.get('momentum_5', 0)
        
        # Combine confirmations
        ml_factor = 1.2 if (ml_prediction > 0 and action == "BUY") or (ml_prediction < 0 and action == "SELL") else 0.8
        momentum_factor = 1.1 if (momentum_5 > 0 and action == "BUY") or (momentum_5 < 0 and action == "SELL") else 0.9
        
        final_confidence = base_confidence * ml_factor * momentum_factor
        
        # Position sizing based on pattern strength and volatility
        pattern_strength = pattern.get('pattern_strength', 0.5)
        volatility_ratio = analysis["volatility_analysis"].get(symbol, {}).get('volatility_ratio', 1.0)
        
        # Adjust position size based on volatility
        volatility_adjustment = 1.0 / volatility_ratio if volatility_ratio > 1.0 else 1.0
        position_value = self.current_capital * self.position_limit * pattern_strength * volatility_adjustment
        quantity = position_value / current_price
        
        return TradeDecision(
            symbol=symbol,
            action=action,
            quantity=quantity,
            price=current_price,
            confidence=final_confidence,
            reasoning=f"Quantitative {action.lower()}: {pattern_type} pattern, confidence: {final_confidence:.2f}, ML: {ml_confidence:.2f}",
            timestamp=datetime.now(),
            agent_id=self.agent_id
        )
    
    async def update_strategy(self, performance_feedback: Dict[str, Any]) -> None:
        """Update quantitative pattern recognition strategy."""
        # Adjust pattern confidence threshold
        if performance_feedback.get("win_rate", 0) < 0.7:
            self.min_pattern_confidence += 0.05
        elif performance_feedback.get("win_rate", 0) > 0.85:
            self.min_pattern_confidence = max(0.6, self.min_pattern_confidence - 0.02)
        
        # Adjust volatility threshold
        if performance_feedback.get("max_drawdown", 0) > 0.04:
            self.volatility_threshold *= 0.8  # Be more conservative
        elif performance_feedback.get("sharpe_ratio", 0) > 2.0:
            self.volatility_threshold *= 1.1  # Be slightly more aggressive
        
        # Adjust position sizing
        if performance_feedback.get("max_drawdown", 0) > 0.05:
            self.position_limit *= 0.9
        elif performance_feedback.get("sharpe_ratio", 0) > 2.5:
            self.position_limit = min(0.1, self.position_limit * 1.05)
