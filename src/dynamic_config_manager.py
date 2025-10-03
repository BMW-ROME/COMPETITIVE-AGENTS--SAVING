#!/usr/bin/env python3
"""
Dynamic Configuration Manager
=============================
Automatically adapts system configuration based on trading mode
and provides safe mode switching capabilities.
"""

import os
import logging
from typing import Dict, Any, Optional
from datetime import datetime
import json

class DynamicConfigManager:
    """Manages dynamic configuration based on trading mode"""
    
    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self.trading_mode_manager = None  # Will be injected
        self.config_cache = {}
        self.last_update = None
        
    def set_trading_mode_manager(self, mode_manager):
        """Set the trading mode manager reference"""
        self.trading_mode_manager = mode_manager
    
    def get_dynamic_config(self) -> Dict[str, Any]:
        """Get dynamically configured settings based on current mode"""
        if not self.trading_mode_manager:
            return self._get_default_config()
        
        mode_info = self.trading_mode_manager.get_mode_info()
        base_config = self._get_base_config()
        
        # Apply mode-specific configurations
        dynamic_config = self._apply_mode_configuration(base_config, mode_info)
        
        # Cache the configuration
        self.config_cache = dynamic_config
        self.last_update = datetime.now()
        
        return dynamic_config
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration when mode manager is not available"""
        return {
            'trading_mode': 'PAPER',
            'alpaca_base_url': 'https://paper-api.alpaca.markets',
            'alpaca_data_url': 'https://data.alpaca.markets',
            'max_position_size': 0.05,
            'max_daily_loss': 0.02,
            'stop_loss_pct': 0.03,
            'trading_enabled': True,
            'real_money': False,
            'safety_level': 'HIGH',
            'risk_multiplier': 1.0,
            'data_sources': ['alpaca', 'mock'],
            'crypto_enabled': False,
            'news_sources': ['mock'],
            'social_sentiment': False
        }
    
    def _get_base_config(self) -> Dict[str, Any]:
        """Get base configuration template"""
        return {
            'trading_mode': 'PAPER',
            'alpaca_base_url': 'https://paper-api.alpaca.markets',
            'alpaca_data_url': 'https://data.alpaca.markets',
            'max_position_size': 0.05,
            'max_daily_loss': 0.02,
            'stop_loss_pct': 0.03,
            'trading_enabled': True,
            'real_money': False,
            'safety_level': 'HIGH',
            'risk_multiplier': 1.0,
            'data_sources': ['alpaca', 'mock'],
            'crypto_enabled': False,
            'news_sources': ['mock'],
            'social_sentiment': False,
            'agent_configs': {},
            'backtesting_enabled': True,
            'real_time_analysis': True,
            'performance_tracking': True,
            'logging_level': 'INFO'
        }
    
    def _apply_mode_configuration(self, base_config: Dict[str, Any], mode_info: Dict[str, Any]) -> Dict[str, Any]:
        """Apply mode-specific configuration overrides"""
        config = base_config.copy()
        
        # Update with mode-specific settings
        config.update({
            'trading_mode': mode_info['current_mode'],
            'alpaca_base_url': mode_info['alpaca_base_url'],
            'max_position_size': mode_info['max_position_size'],
            'max_daily_loss': mode_info['max_daily_loss'],
            'stop_loss_pct': mode_info['stop_loss_pct'],
            'trading_enabled': mode_info['trading_enabled'],
            'real_money': mode_info['real_money'],
            'safety_level': mode_info['safety_level']
        })
        
        # Mode-specific adaptations
        if mode_info['current_mode'] == 'LIVE':
            config.update({
                'risk_multiplier': 0.5,  # More conservative for live trading
                'data_sources': ['alpaca'],  # Only real data sources
                'crypto_enabled': False,  # Disable crypto for safety
                'news_sources': ['alpaca', 'alpha_vantage'],  # Real news sources
                'social_sentiment': True,  # Enable sentiment analysis
                'backtesting_enabled': False,  # Disable backtesting in live mode
                'logging_level': 'WARNING',  # Reduce log noise
                'agent_configs': self._get_live_agent_configs()
            })
            
        elif mode_info['current_mode'] == 'PAPER':
            config.update({
                'risk_multiplier': 1.0,  # Standard risk for paper trading
                'data_sources': ['alpaca', 'mock'],  # Mix of real and mock data
                'crypto_enabled': True,  # Enable crypto for testing
                'news_sources': ['mock', 'alpaca'],  # Mix of sources
                'social_sentiment': True,  # Enable sentiment analysis
                'backtesting_enabled': True,  # Enable backtesting
                'logging_level': 'INFO',  # Standard logging
                'agent_configs': self._get_paper_agent_configs()
            })
            
        elif mode_info['current_mode'] == 'SIMULATION':
            config.update({
                'risk_multiplier': 2.0,  # More aggressive for simulation
                'data_sources': ['mock'],  # Only mock data
                'crypto_enabled': True,  # Enable crypto for testing
                'news_sources': ['mock'],  # Only mock news
                'social_sentiment': True,  # Enable sentiment analysis
                'backtesting_enabled': True,  # Enable backtesting
                'trading_enabled': False,  # No actual trading
                'logging_level': 'DEBUG',  # Detailed logging
                'agent_configs': self._get_simulation_agent_configs()
            })
        
        return config
    
    def _get_live_agent_configs(self) -> Dict[str, Any]:
        """Get agent configurations optimized for live trading"""
        return {
            'conservative_1': {
                'max_position_size': 0.01,  # 1% max position
                'max_daily_loss': 0.005,     # 0.5% max daily loss
                'stop_loss_pct': 0.015,      # 1.5% stop loss
                'take_profit_pct': 0.03,     # 3% take profit
                'confidence_threshold': 0.8,  # High confidence required
                'max_trades_per_day': 3,     # Limit daily trades
                'enabled': True
            },
            'balanced_1': {
                'max_position_size': 0.015,  # 1.5% max position
                'max_daily_loss': 0.008,     # 0.8% max daily loss
                'stop_loss_pct': 0.02,       # 2% stop loss
                'take_profit_pct': 0.04,     # 4% take profit
                'confidence_threshold': 0.75, # High confidence required
                'max_trades_per_day': 5,     # Limit daily trades
                'enabled': True
            },
            'ai_enhanced_1': {
                'max_position_size': 0.02,   # 2% max position
                'max_daily_loss': 0.01,      # 1% max daily loss
                'stop_loss_pct': 0.025,      # 2.5% stop loss
                'take_profit_pct': 0.05,     # 5% take profit
                'confidence_threshold': 0.7,  # High confidence required
                'max_trades_per_day': 8,     # Limit daily trades
                'enabled': True
            }
        }
    
    def _get_paper_agent_configs(self) -> Dict[str, Any]:
        """Get agent configurations for paper trading"""
        return {
            'conservative_1': {
                'max_position_size': 0.05,   # 5% max position
                'max_daily_loss': 0.02,      # 2% max daily loss
                'stop_loss_pct': 0.03,       # 3% stop loss
                'take_profit_pct': 0.06,     # 6% take profit
                'confidence_threshold': 0.6,  # Moderate confidence
                'max_trades_per_day': 10,    # More trades allowed
                'enabled': True
            },
            'balanced_1': {
                'max_position_size': 0.05,    # 5% max position
                'max_daily_loss': 0.02,      # 2% max daily loss
                'stop_loss_pct': 0.03,       # 3% stop loss
                'take_profit_pct': 0.06,     # 6% take profit
                'confidence_threshold': 0.5,  # Moderate confidence
                'max_trades_per_day': 15,    # More trades allowed
                'enabled': True
            },
            'aggressive_1': {
                'max_position_size': 0.05,   # 5% max position
                'max_daily_loss': 0.02,      # 2% max daily loss
                'stop_loss_pct': 0.03,       # 3% stop loss
                'take_profit_pct': 0.06,     # 6% take profit
                'confidence_threshold': 0.4,  # Lower confidence threshold
                'max_trades_per_day': 20,     # More trades allowed
                'enabled': True
            },
            'ai_enhanced_1': {
                'max_position_size': 0.05,   # 5% max position
                'max_daily_loss': 0.02,      # 2% max daily loss
                'stop_loss_pct': 0.03,       # 3% stop loss
                'take_profit_pct': 0.06,     # 6% take profit
                'confidence_threshold': 0.5,  # Moderate confidence
                'max_trades_per_day': 15,     # More trades allowed
                'enabled': True
            }
        }
    
    def _get_simulation_agent_configs(self) -> Dict[str, Any]:
        """Get agent configurations for simulation mode"""
        return {
            'conservative_1': {
                'max_position_size': 0.10,   # 10% max position
                'max_daily_loss': 0.05,      # 5% max daily loss
                'stop_loss_pct': 0.05,       # 5% stop loss
                'take_profit_pct': 0.10,     # 10% take profit
                'confidence_threshold': 0.3,  # Low confidence threshold
                'max_trades_per_day': 50,    # Many trades allowed
                'enabled': True
            },
            'balanced_1': {
                'max_position_size': 0.10,   # 10% max position
                'max_daily_loss': 0.05,      # 5% max daily loss
                'stop_loss_pct': 0.05,       # 5% stop loss
                'take_profit_pct': 0.10,     # 10% take profit
                'confidence_threshold': 0.3,  # Low confidence threshold
                'max_trades_per_day': 50,    # Many trades allowed
                'enabled': True
            },
            'aggressive_1': {
                'max_position_size': 0.10,   # 10% max position
                'max_daily_loss': 0.05,      # 5% max daily loss
                'stop_loss_pct': 0.05,       # 5% stop loss
                'take_profit_pct': 0.10,     # 10% take profit
                'confidence_threshold': 0.2,  # Very low confidence threshold
                'max_trades_per_day': 100,   # Many trades allowed
                'enabled': True
            },
            'ai_enhanced_1': {
                'max_position_size': 0.10,   # 10% max position
                'max_daily_loss': 0.05,      # 5% max daily loss
                'stop_loss_pct': 0.05,       # 5% stop loss
                'take_profit_pct': 0.10,     # 10% take profit
                'confidence_threshold': 0.3,  # Low confidence threshold
                'max_trades_per_day': 50,    # Many trades allowed
                'enabled': True
            }
        }
    
    def get_environment_variables(self) -> Dict[str, str]:
        """Get environment variables based on current configuration"""
        config = self.get_dynamic_config()
        
        env_vars = {
            'TRADING_MODE': config['trading_mode'],
            'ALPACA_BASE_URL': config['alpaca_base_url'],
            'ALPACA_DATA_URL': config['alpaca_data_url'],
            'MAX_POSITION_SIZE': str(config['max_position_size']),
            'MAX_DAILY_LOSS': str(config['max_daily_loss']),
            'STOP_LOSS_PCT': str(config['stop_loss_pct']),
            'TRADING_ENABLED': str(config['trading_enabled']),
            'REAL_MONEY': str(config['real_money']),
            'SAFETY_LEVEL': config['safety_level'],
            'RISK_MULTIPLIER': str(config['risk_multiplier']),
            'CRYPTO_ENABLED': str(config['crypto_enabled']),
            'BACKTESTING_ENABLED': str(config['backtesting_enabled']),
            'REAL_TIME_ANALYSIS': str(config['real_time_analysis']),
            'PERFORMANCE_TRACKING': str(config['performance_tracking']),
            'LOG_LEVEL': config['logging_level']
        }
        
        return env_vars
    
    def validate_configuration(self) -> Dict[str, Any]:
        """Validate current configuration for consistency"""
        result = {
            'valid': True,
            'warnings': [],
            'errors': [],
            'recommendations': []
        }
        
        config = self.get_dynamic_config()
        
        # Check for configuration inconsistencies
        if config['real_money'] and config['max_position_size'] > 0.05:
            result['warnings'].append("High position size in live trading mode")
            result['recommendations'].append("Consider reducing max_position_size for live trading")
        
        if config['trading_enabled'] and config['max_daily_loss'] > 0.05:
            result['warnings'].append("High daily loss limit")
            result['recommendations'].append("Consider reducing max_daily_loss for safety")
        
        if config['trading_mode'] == 'LIVE' and not config['real_money']:
            result['errors'].append("Live mode should have real_money enabled")
            result['valid'] = False
        
        if config['trading_mode'] == 'PAPER' and config['real_money']:
            result['errors'].append("Paper mode should not have real_money enabled")
            result['valid'] = False
        
        return result
    
    def get_mode_switching_guide(self) -> Dict[str, Any]:
        """Get guide for safe mode switching"""
        return {
            'paper_to_simulation': {
                'description': 'Switch from Paper to Simulation mode',
                'safety_level': 'LOW',
                'confirmations_required': 0,
                'changes': [
                    'Trading disabled',
                    'More aggressive parameters',
                    'Mock data only',
                    'Higher position limits'
                ]
            },
            'paper_to_live': {
                'description': 'Switch from Paper to Live mode',
                'safety_level': 'CRITICAL',
                'confirmations_required': 3,
                'changes': [
                    'Real money at risk',
                    'Conservative parameters',
                    'Real data sources only',
                    'Tighter risk controls'
                ],
                'requirements': [
                    'Valid Alpaca credentials',
                    'Risk management review',
                    'Position size verification',
                    'Stop loss confirmation'
                ]
            },
            'live_to_paper': {
                'description': 'Switch from Live to Paper mode',
                'safety_level': 'HIGH',
                'confirmations_required': 1,
                'changes': [
                    'No real money at risk',
                    'Paper trading environment',
                    'Mixed data sources',
                    'Standard risk controls'
                ]
            },
            'any_to_emergency': {
                'description': 'Emergency stop all trading',
                'safety_level': 'MAXIMUM',
                'confirmations_required': 1,
                'changes': [
                    'All trading halted',
                    'Simulation mode activated',
                    'No new positions',
                    'Existing positions monitored'
                ]
            }
        }

