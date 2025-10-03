#!/usr/bin/env python3
"""
Separated Trading System - Stocks and Crypto
============================================
Separates stock and crypto trading with distinct agents and strategies.
"""

import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
import asyncio

class SeparatedTradingSystem:
    """Trading system that separates stocks and crypto"""
    
    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self.stock_agents = {}
        self.crypto_agents = {}
        self.stock_symbols = ["AAPL", "MSFT", "GOOGL", "TSLA", "SPY", "QQQ"]
        self.crypto_symbols = ["BTCUSD", "ETHUSD"]
        self.trading_enabled = True
        
    def initialize_agents(self):
        """Initialize separated trading agents"""
        # Stock trading agents
        self.stock_agents = {
            'conservative_stock': {
                'type': 'conservative',
                'symbols': ["SPY", "QQQ", "AAPL", "MSFT"],
                'max_position': 0.02,
                'max_daily_loss': 0.01,
                'stop_loss': 0.02,
                'take_profit': 0.04
            },
            'balanced_stock': {
                'type': 'balanced',
                'symbols': ["AAPL", "MSFT", "GOOGL", "TSLA"],
                'max_position': 0.05,
                'max_daily_loss': 0.02,
                'stop_loss': 0.03,
                'take_profit': 0.06
            },
            'aggressive_stock': {
                'type': 'aggressive',
                'symbols': ["TSLA", "GOOGL", "AAPL", "MSFT"],
                'max_position': 0.08,
                'max_daily_loss': 0.03,
                'stop_loss': 0.04,
                'take_profit': 0.08
            },
            'scalping_stock': {
                'type': 'scalping',
                'symbols': ["SPY", "QQQ"],
                'max_position': 0.03,
                'max_daily_loss': 0.015,
                'stop_loss': 0.015,
                'take_profit': 0.03
            }
        }
        
        # Crypto trading agents
        self.crypto_agents = {
            'conservative_crypto': {
                'type': 'conservative',
                'symbols': ["BTCUSD"],
                'max_position': 0.03,
                'max_daily_loss': 0.02,
                'stop_loss': 0.05,
                'take_profit': 0.08
            },
            'balanced_crypto': {
                'type': 'balanced',
                'symbols': ["BTCUSD", "ETHUSD"],
                'max_position': 0.06,
                'max_daily_loss': 0.025,
                'stop_loss': 0.05,
                'take_profit': 0.10
            },
            'aggressive_crypto': {
                'type': 'aggressive',
                'symbols': ["BTCUSD", "ETHUSD"],
                'max_position': 0.10,
                'max_daily_loss': 0.04,
                'stop_loss': 0.08,
                'take_profit': 0.15
            }
        }
        
        self.logger.info(f"Initialized {len(self.stock_agents)} stock agents and {len(self.crypto_agents)} crypto agents")
    
    def analyze_stock_market(self, market_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Analyze stock market and generate trading decisions"""
        decisions = []
        
        for agent_id, agent_config in self.stock_agents.items():
            agent_decisions = self._analyze_with_agent(agent_id, agent_config, market_data, 'stock')
            decisions.extend(agent_decisions)
        
        return decisions
    
    def analyze_crypto_market(self, market_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Analyze crypto market and generate trading decisions"""
        decisions = []
        
        for agent_id, agent_config in self.crypto_agents.items():
            agent_decisions = self._analyze_with_agent(agent_id, agent_config, market_data, 'crypto')
            decisions.extend(agent_decisions)
        
        return decisions
    
    def _analyze_with_agent(self, agent_id: str, agent_config: Dict[str, Any], 
                          market_data: Dict[str, Any], market_type: str) -> List[Dict[str, Any]]:
        """Analyze market with specific agent"""
        decisions = []
        
        for symbol in agent_config['symbols']:
            if symbol in market_data:
                decision = self._create_trading_decision(
                    agent_id, symbol, market_data[symbol], agent_config, market_type
                )
                if decision:
                    decisions.append(decision)
        
        return decisions
    
    def _create_trading_decision(self, agent_id: str, symbol: str, data: Dict[str, Any], 
                               agent_config: Dict[str, Any], market_type: str) -> Optional[Dict[str, Any]]:
        """Create trading decision based on agent configuration"""
        if not data or 'price' not in data:
            return None
        
        price = data['price']
        volume = data.get('volume', 0)
        
        # Different volume thresholds for stocks vs crypto
        if market_type == 'stock':
            min_volume = 500000  # 500K for stocks
        else:  # crypto
            min_volume = 100000  # 100K for crypto
        
        if volume < min_volume:
            return None
        
        # Analyze based on agent type
        if agent_config['type'] == 'conservative':
            decision = self._conservative_analysis(symbol, data, agent_config)
        elif agent_config['type'] == 'balanced':
            decision = self._balanced_analysis(symbol, data, agent_config)
        elif agent_config['type'] == 'aggressive':
            decision = self._aggressive_analysis(symbol, data, agent_config)
        elif agent_config['type'] == 'scalping':
            decision = self._scalping_analysis(symbol, data, agent_config)
        else:
            return None
        
        if decision:
            decision.update({
                'agent_id': agent_id,
                'market_type': market_type,
                'max_position': agent_config['max_position'],
                'max_daily_loss': agent_config['max_daily_loss'],
                'stop_loss': agent_config['stop_loss'],
                'take_profit': agent_config['take_profit']
            })
        
        return decision
    
    def _conservative_analysis(self, symbol: str, data: Dict[str, Any], 
                             agent_config: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Conservative analysis - high confidence required"""
        if 'sma_20' in data and 'sma_50' in data:
            sma_20 = data['sma_20']
            sma_50 = data['sma_50']
            price = data['price']
            
            if sma_20 > sma_50 and price > sma_20:
                confidence = min(0.9, (sma_20 - sma_50) / sma_50 + 0.5)
                if confidence >= 0.8:  # High confidence threshold
                    return {
                        'symbol': symbol,
                        'action': 'BUY',
                        'quantity': agent_config['max_position'] * 0.8,  # 80% of max position
                        'price': price,
                        'confidence': confidence,
                        'reason': 'Conservative momentum',
                        'style': 'conservative'
                    }
        return None
    
    def _balanced_analysis(self, symbol: str, data: Dict[str, Any], 
                         agent_config: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Balanced analysis - moderate confidence"""
        if 'rsi' in data:
            rsi = data['rsi']
            price = data['price']
            
            if 30 <= rsi <= 40:  # Oversold but not extreme
                confidence = (40 - rsi) / 10 * 0.8
                if confidence >= 0.6:  # Moderate confidence threshold
                    return {
                        'symbol': symbol,
                        'action': 'BUY',
                        'quantity': agent_config['max_position'] * 0.9,  # 90% of max position
                        'price': price,
                        'confidence': confidence,
                        'reason': 'Balanced RSI oversold',
                        'style': 'balanced'
                    }
        return None
    
    def _aggressive_analysis(self, symbol: str, data: Dict[str, Any], 
                           agent_config: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Aggressive analysis - lower confidence threshold"""
        if 'volatility' in data:
            volatility = data['volatility']
            price = data['price']
            
            if volatility > 0.02:  # 2% volatility threshold
                confidence = min(0.8, volatility * 10)
                if confidence >= 0.5:  # Lower confidence threshold
                    return {
                        'symbol': symbol,
                        'action': 'BUY',
                        'quantity': agent_config['max_position'] * 1.0,  # Full position
                        'price': price,
                        'confidence': confidence,
                        'reason': 'Aggressive volatility play',
                        'style': 'aggressive'
                    }
        return None
    
    def _scalping_analysis(self, symbol: str, data: Dict[str, Any], 
                         agent_config: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Scalping analysis - quick momentum trades"""
        if 'price_change' in data and 'volume_change' in data:
            price_change = data['price_change']
            volume_change = data['volume_change']
            price = data['price']
            
            if price_change > 0.01 and volume_change > 0.5:  # 1% price increase, 50% volume increase
                confidence = min(0.9, (price_change * 10 + volume_change * 0.1))
                if confidence >= 0.7:  # High confidence for scalping
                    return {
                        'symbol': symbol,
                        'action': 'BUY',
                        'quantity': agent_config['max_position'] * 0.7,  # 70% of max position
                        'price': price,
                        'confidence': confidence,
                        'reason': 'Scalping momentum',
                        'style': 'scalping'
                    }
        return None
    
    def get_trading_summary(self) -> Dict[str, Any]:
        """Get trading system summary"""
        return {
            'stock_agents': len(self.stock_agents),
            'crypto_agents': len(self.crypto_agents),
            'stock_symbols': self.stock_symbols,
            'crypto_symbols': self.crypto_symbols,
            'trading_enabled': self.trading_enabled,
            'agent_types': {
                'stock': list(set(agent['type'] for agent in self.stock_agents.values())),
                'crypto': list(set(agent['type'] for agent in self.crypto_agents.values()))
            }
        }
    
    def update_trading_status(self, enabled: bool):
        """Update trading status"""
        self.trading_enabled = enabled
        self.logger.info(f"Trading {'enabled' if enabled else 'disabled'}")
    
    def get_agent_config(self, agent_id: str) -> Optional[Dict[str, Any]]:
        """Get agent configuration"""
        if agent_id in self.stock_agents:
            return self.stock_agents[agent_id]
        elif agent_id in self.crypto_agents:
            return self.crypto_agents[agent_id]
        return None

