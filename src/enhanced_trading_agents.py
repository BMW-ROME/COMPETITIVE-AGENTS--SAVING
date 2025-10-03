#!/usr/bin/env python3
"""
Enhanced Trading Agents with Distinct Styles
============================================
Each agent has unique trading characteristics and risk profiles.
"""

import logging
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import random
import numpy as np

class EnhancedTradingAgent:
    """Base class for enhanced trading agents with distinct styles"""
    
    def __init__(self, agent_id: str, config: Dict[str, Any], logger: logging.Logger):
        self.agent_id = agent_id
        self.config = config
        self.logger = logger
        self.trading_style = self._get_trading_style()
        self.risk_profile = self._get_risk_profile()
        self.symbols = self._get_symbols()
        self.safe_zone_multiplier = 0.8  # Only use 80% of available buying power
        
    def _get_trading_style(self) -> str:
        """Get agent's trading style"""
        return "balanced"
    
    def _get_risk_profile(self) -> Dict[str, float]:
        """Get agent's risk profile"""
        return {
            'max_position_size': 0.05,
            'max_daily_loss': 0.02,
            'stop_loss_pct': 0.03,
            'take_profit_pct': 0.06,
            'confidence_threshold': 0.6
        }
    
    def _get_symbols(self) -> List[str]:
        """Get agent's trading symbols"""
        return ["AAPL", "MSFT", "GOOGL", "TSLA", "SPY", "QQQ"]
    
    def analyze_market(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze market and generate trading decisions"""
        decisions = []
        
        for symbol in self.symbols:
            if symbol in market_data:
                decision = self._analyze_symbol(symbol, market_data[symbol])
                if decision:
                    decisions.append(decision)
        
        return {
            'agent_id': self.agent_id,
            'trading_style': self.trading_style,
            'decisions': decisions,
            'confidence': self._calculate_confidence(decisions),
            'risk_level': self._calculate_risk_level(decisions)
        }
    
    def _analyze_symbol(self, symbol: str, data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Analyze individual symbol"""
        # Implement specific analysis logic
        return None
    
    def _calculate_confidence(self, decisions: List[Dict[str, Any]]) -> float:
        """Calculate overall confidence"""
        if not decisions:
            return 0.0
        return sum(d.get('confidence', 0.5) for d in decisions) / len(decisions)
    
    def _calculate_risk_level(self, decisions: List[Dict[str, Any]]) -> str:
        """Calculate risk level"""
        if not decisions:
            return 'LOW'
        
        total_position = sum(d.get('position_size', 0) for d in decisions)
        if total_position > 0.1:
            return 'HIGH'
        elif total_position > 0.05:
            return 'MEDIUM'
        else:
            return 'LOW'

class ConservativeAgent(EnhancedTradingAgent):
    """Conservative agent - Low risk, steady gains"""
    
    def _get_trading_style(self) -> str:
        return "conservative"
    
    def _get_risk_profile(self) -> Dict[str, float]:
        return {
            'max_position_size': 0.02,  # 2% max position
            'max_daily_loss': 0.01,      # 1% max daily loss
            'stop_loss_pct': 0.02,       # 2% stop loss
            'take_profit_pct': 0.04,     # 4% take profit
            'confidence_threshold': 0.8   # High confidence required
        }
    
    def _get_symbols(self) -> List[str]:
        return ["SPY", "QQQ", "AAPL", "MSFT"]  # Blue chip stocks only
    
    def _analyze_symbol(self, symbol: str, data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Conservative analysis - only high-confidence trades"""
        if not data or 'price' not in data:
            return None
        
        price = data['price']
        volume = data.get('volume', 0)
        
        # Only trade if volume is high (liquid)
        if volume < 1000000:  # 1M volume threshold
            return None
        
        # Simple momentum analysis
        if 'sma_20' in data and 'sma_50' in data:
            sma_20 = data['sma_20']
            sma_50 = data['sma_50']
            
            if sma_20 > sma_50 and price > sma_20:
                confidence = min(0.9, (sma_20 - sma_50) / sma_50 + 0.5)
                if confidence >= self.risk_profile['confidence_threshold']:
                    return {
                        'symbol': symbol,
                        'action': 'BUY',
                        'quantity': self._calculate_position_size(price, confidence),
                        'price': price,
                        'confidence': confidence,
                        'reason': 'Conservative momentum',
                        'stop_loss': price * (1 - self.risk_profile['stop_loss_pct']),
                        'take_profit': price * (1 + self.risk_profile['take_profit_pct'])
                    }
        
        return None
    
    def _calculate_position_size(self, price: float, confidence: float) -> float:
        """Calculate position size based on confidence and risk profile"""
        base_size = self.risk_profile['max_position_size'] * self.safe_zone_multiplier
        confidence_multiplier = min(1.0, confidence)
        return base_size * confidence_multiplier

class AggressiveAgent(EnhancedTradingAgent):
    """Aggressive agent - Higher risk, higher potential returns"""
    
    def _get_trading_style(self) -> str:
        return "aggressive"
    
    def _get_risk_profile(self) -> Dict[str, float]:
        return {
            'max_position_size': 0.08,  # 8% max position
            'max_daily_loss': 0.03,      # 3% max daily loss
            'stop_loss_pct': 0.04,       # 4% stop loss
            'take_profit_pct': 0.08,     # 8% take profit
            'confidence_threshold': 0.5  # Lower confidence threshold
        }
    
    def _get_symbols(self) -> List[str]:
        return ["TSLA", "GOOGL", "AAPL", "MSFT", "SPY", "QQQ"]  # Growth stocks
    
    def _analyze_symbol(self, symbol: str, data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Aggressive analysis - more trades, higher risk"""
        if not data or 'price' not in data:
            return None
        
        price = data['price']
        volume = data.get('volume', 0)
        
        # Lower volume threshold for aggressive trading
        if volume < 500000:  # 500K volume threshold
            return None
        
        # Volatility-based analysis
        if 'volatility' in data:
            volatility = data['volatility']
            
            # Trade on high volatility
            if volatility > 0.02:  # 2% volatility threshold
                confidence = min(0.8, volatility * 10)
                if confidence >= self.risk_profile['confidence_threshold']:
                    return {
                        'symbol': symbol,
                        'action': 'BUY',
                        'quantity': self._calculate_position_size(price, confidence),
                        'price': price,
                        'confidence': confidence,
                        'reason': 'Aggressive volatility play',
                        'stop_loss': price * (1 - self.risk_profile['stop_loss_pct']),
                        'take_profit': price * (1 + self.risk_profile['take_profit_pct'])
                    }
        
        return None
    
    def _calculate_position_size(self, price: float, confidence: float) -> float:
        """Calculate position size - more aggressive"""
        base_size = self.risk_profile['max_position_size'] * self.safe_zone_multiplier
        confidence_multiplier = min(1.2, confidence * 1.5)  # More aggressive sizing
        return base_size * confidence_multiplier

class BalancedAgent(EnhancedTradingAgent):
    """Balanced agent - Moderate risk, steady growth"""
    
    def _get_trading_style(self) -> str:
        return "balanced"
    
    def _get_risk_profile(self) -> Dict[str, float]:
        return {
            'max_position_size': 0.05,  # 5% max position
            'max_daily_loss': 0.02,      # 2% max daily loss
            'stop_loss_pct': 0.03,       # 3% stop loss
            'take_profit_pct': 0.06,     # 6% take profit
            'confidence_threshold': 0.6  # Moderate confidence threshold
        }
    
    def _get_symbols(self) -> List[str]:
        return ["AAPL", "MSFT", "GOOGL", "TSLA", "SPY", "QQQ"]
    
    def _analyze_symbol(self, symbol: str, data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Balanced analysis - moderate risk/reward"""
        if not data or 'price' not in data:
            return None
        
        price = data['price']
        volume = data.get('volume', 0)
        
        if volume < 750000:  # 750K volume threshold
            return None
        
        # RSI-based analysis
        if 'rsi' in data:
            rsi = data['rsi']
            
            # Buy on oversold conditions
            if 30 <= rsi <= 40:  # Oversold but not extreme
                confidence = (40 - rsi) / 10 * 0.8  # Scale confidence
                if confidence >= self.risk_profile['confidence_threshold']:
                    return {
                        'symbol': symbol,
                        'action': 'BUY',
                        'quantity': self._calculate_position_size(price, confidence),
                        'price': price,
                        'confidence': confidence,
                        'reason': 'Balanced RSI oversold',
                        'stop_loss': price * (1 - self.risk_profile['stop_loss_pct']),
                        'take_profit': price * (1 + self.risk_profile['take_profit_pct'])
                    }
        
        return None
    
    def _calculate_position_size(self, price: float, confidence: float) -> float:
        """Calculate position size - balanced approach"""
        base_size = self.risk_profile['max_position_size'] * self.safe_zone_multiplier
        confidence_multiplier = min(1.0, confidence)
        return base_size * confidence_multiplier

class ScalpingAgent(EnhancedTradingAgent):
    """Scalping agent - Quick trades, small profits"""
    
    def _get_trading_style(self) -> str:
        return "scalping"
    
    def _get_risk_profile(self) -> Dict[str, float]:
        return {
            'max_position_size': 0.03,  # 3% max position
            'max_daily_loss': 0.015,     # 1.5% max daily loss
            'stop_loss_pct': 0.015,      # 1.5% stop loss
            'take_profit_pct': 0.03,     # 3% take profit
            'confidence_threshold': 0.7  # High confidence for quick trades
        }
    
    def _get_symbols(self) -> List[str]:
        return ["SPY", "QQQ"]  # High liquidity ETFs
    
    def _analyze_symbol(self, symbol: str, data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Scalping analysis - quick momentum trades"""
        if not data or 'price' not in data:
            return None
        
        price = data['price']
        volume = data.get('volume', 0)
        
        if volume < 2000000:  # 2M volume threshold for scalping
            return None
        
        # Price momentum analysis
        if 'price_change' in data and 'volume_change' in data:
            price_change = data['price_change']
            volume_change = data['volume_change']
            
            # Look for strong momentum with volume
            if price_change > 0.01 and volume_change > 0.5:  # 1% price increase, 50% volume increase
                confidence = min(0.9, (price_change * 10 + volume_change * 0.1))
                if confidence >= self.risk_profile['confidence_threshold']:
                    return {
                        'symbol': symbol,
                        'action': 'BUY',
                        'quantity': self._calculate_position_size(price, confidence),
                        'price': price,
                        'confidence': confidence,
                        'reason': 'Scalping momentum',
                        'stop_loss': price * (1 - self.risk_profile['stop_loss_pct']),
                        'take_profit': price * (1 + self.risk_profile['take_profit_pct'])
                    }
        
        return None
    
    def _calculate_position_size(self, price: float, confidence: float) -> float:
        """Calculate position size - smaller for scalping"""
        base_size = self.risk_profile['max_position_size'] * self.safe_zone_multiplier
        confidence_multiplier = min(0.8, confidence)  # Smaller positions for scalping
        return base_size * confidence_multiplier

class CryptoAgent(EnhancedTradingAgent):
    """Crypto agent - Specialized for cryptocurrency trading"""
    
    def _get_trading_style(self) -> str:
        return "crypto"
    
    def _get_risk_profile(self) -> Dict[str, float]:
        return {
            'max_position_size': 0.06,  # 6% max position
            'max_daily_loss': 0.025,     # 2.5% max daily loss
            'stop_loss_pct': 0.05,       # 5% stop loss (higher for crypto)
            'take_profit_pct': 0.10,     # 10% take profit
            'confidence_threshold': 0.6  # Moderate confidence
        }
    
    def _get_symbols(self) -> List[str]:
        return ["BTCUSD", "ETHUSD"]  # Crypto symbols only
    
    def _analyze_symbol(self, symbol: str, data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Crypto analysis - volatility-based trading"""
        if not data or 'price' not in data:
            return None
        
        price = data['price']
        volume = data.get('volume', 0)
        
        if volume < 100000:  # Lower volume threshold for crypto
            return None
        
        # Crypto-specific analysis
        if 'volatility' in data and 'trend' in data:
            volatility = data['volatility']
            trend = data['trend']
            
            # Trade on high volatility with trend
            if volatility > 0.03 and trend > 0:  # 3% volatility, positive trend
                confidence = min(0.8, volatility * 5 + trend * 0.3)
                if confidence >= self.risk_profile['confidence_threshold']:
                    return {
                        'symbol': symbol,
                        'action': 'BUY',
                        'quantity': self._calculate_position_size(price, confidence),
                        'price': price,
                        'confidence': confidence,
                        'reason': 'Crypto volatility trend',
                        'stop_loss': price * (1 - self.risk_profile['stop_loss_pct']),
                        'take_profit': price * (1 + self.risk_profile['take_profit_pct'])
                    }
        
        return None
    
    def _calculate_position_size(self, price: float, confidence: float) -> float:
        """Calculate position size - crypto-specific"""
        base_size = self.risk_profile['max_position_size'] * self.safe_zone_multiplier
        confidence_multiplier = min(1.1, confidence * 1.2)  # Slightly more aggressive for crypto
        return base_size * confidence_multiplier

def create_enhanced_agents(config: Dict[str, Any], logger: logging.Logger) -> Dict[str, EnhancedTradingAgent]:
    """Create enhanced trading agents with distinct styles"""
    agents = {}
    
    # Create agents with distinct styles
    agents['conservative_1'] = ConservativeAgent('conservative_1', config, logger)
    agents['aggressive_1'] = AggressiveAgent('aggressive_1', config, logger)
    agents['balanced_1'] = BalancedAgent('balanced_1', config, logger)
    agents['scalping_1'] = ScalpingAgent('scalping_1', config, logger)
    agents['crypto_1'] = CryptoAgent('crypto_1', config, logger)
    
    return agents

