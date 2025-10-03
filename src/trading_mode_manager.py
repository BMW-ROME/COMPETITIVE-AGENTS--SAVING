#!/usr/bin/env python3
"""
Trading Mode Manager - Dynamic Paper/Live Mode Switching
=======================================================
Handles seamless switching between Paper and Live trading modes
with proper isolation, validation, and safety measures.
"""

import os
import logging
from typing import Dict, Any, Optional
from enum import Enum
from datetime import datetime
import json

class TradingMode(Enum):
    """Trading mode enumeration"""
    PAPER = "PAPER"
    LIVE = "LIVE"
    SIMULATION = "SIMULATION"

class TradingModeManager:
    """Manages dynamic switching between Paper and Live trading modes"""
    
    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self.current_mode = self._detect_current_mode()
        self.mode_configs = self._load_mode_configurations()
        self.safety_checks = self._initialize_safety_checks()
        
    def _detect_current_mode(self) -> TradingMode:
        """Detect current trading mode from environment"""
        mode_str = os.getenv('TRADING_MODE', 'PAPER').upper()
        
        if mode_str == 'LIVE':
            # Additional validation for live mode
            if not self._validate_live_mode_credentials():
                self.logger.warning("Live mode requested but credentials invalid, falling back to PAPER")
                return TradingMode.PAPER
            return TradingMode.LIVE
        elif mode_str == 'SIMULATION':
            return TradingMode.SIMULATION
        else:
            return TradingMode.PAPER
    
    def _validate_live_mode_credentials(self) -> bool:
        """Validate that live mode credentials are properly configured"""
        api_key = os.getenv('ALPACA_API_KEY', '')
        secret_key = os.getenv('ALPACA_SECRET_KEY', '')
        
        # Check if credentials are real (not placeholders)
        if (api_key == 'your_alpaca_api_key_here' or 
            secret_key == 'your_alpaca_secret_key_here' or
            len(api_key) < 10 or len(secret_key) < 10):
            return False
            
        return True
    
    def _load_mode_configurations(self) -> Dict[str, Dict[str, Any]]:
        """Load configuration for each trading mode"""
        return {
            TradingMode.PAPER.value: {
                'alpaca_base_url': 'https://paper-api.alpaca.markets',
                'alpaca_data_url': 'https://data.alpaca.markets',
                'max_position_size': 0.05,  # 5% for paper trading
                'max_daily_loss': 0.02,     # 2% max daily loss
                'stop_loss_pct': 0.03,      # 3% stop loss
                'trading_enabled': True,
                'real_money': False,
                'safety_level': 'HIGH',
                'description': 'Paper trading mode - No real money at risk'
            },
            TradingMode.LIVE.value: {
                'alpaca_base_url': 'https://api.alpaca.markets',
                'alpaca_data_url': 'https://data.alpaca.markets',
                'max_position_size': 0.02,  # 2% for live trading (more conservative)
                'max_daily_loss': 0.01,     # 1% max daily loss (very conservative)
                'stop_loss_pct': 0.02,      # 2% stop loss (tighter)
                'trading_enabled': True,
                'real_money': True,
                'safety_level': 'CRITICAL',
                'description': 'LIVE trading mode - REAL MONEY AT RISK'
            },
            TradingMode.SIMULATION.value: {
                'alpaca_base_url': 'https://paper-api.alpaca.markets',
                'alpaca_data_url': 'https://data.alpaca.markets',
                'max_position_size': 0.10,  # 10% for simulation (more aggressive)
                'max_daily_loss': 0.05,     # 5% max daily loss
                'stop_loss_pct': 0.05,      # 5% stop loss
                'trading_enabled': False,    # No actual trading
                'real_money': False,
                'safety_level': 'MAXIMUM',
                'description': 'Simulation mode - Analysis only, no trades'
            }
        }
    
    def _initialize_safety_checks(self) -> Dict[str, Any]:
        """Initialize safety checks for mode switching"""
        return {
            'live_mode_confirmations_required': 3,
            'last_mode_switch': None,
            'mode_switch_cooldown': 300,  # 5 minutes
            'emergency_stop_enabled': True,
            'double_confirmation_required': True
        }
    
    def get_current_config(self) -> Dict[str, Any]:
        """Get current mode configuration"""
        return self.mode_configs[self.current_mode.value]
    
    def get_mode_info(self) -> Dict[str, Any]:
        """Get comprehensive mode information"""
        config = self.get_current_config()
        return {
            'current_mode': self.current_mode.value,
            'description': config['description'],
            'real_money': config['real_money'],
            'safety_level': config['safety_level'],
            'trading_enabled': config['trading_enabled'],
            'max_position_size': config['max_position_size'],
            'max_daily_loss': config['max_daily_loss'],
            'stop_loss_pct': config['stop_loss_pct'],
            'alpaca_base_url': config['alpaca_base_url'],
            'last_updated': datetime.now().isoformat()
        }
    
    def can_switch_to_mode(self, target_mode: TradingMode) -> Dict[str, Any]:
        """Check if mode switch is allowed and safe"""
        result = {
            'allowed': True,
            'warnings': [],
            'errors': [],
            'confirmations_required': 0
        }
        
        # Check cooldown period
        if self.safety_checks['last_mode_switch']:
            time_since_switch = (datetime.now() - self.safety_checks['last_mode_switch']).total_seconds()
            if time_since_switch < self.safety_checks['mode_switch_cooldown']:
                result['errors'].append(f"Mode switch cooldown active. Wait {self.safety_checks['mode_switch_cooldown'] - time_since_switch:.0f} seconds")
                result['allowed'] = False
        
        # Special checks for live mode
        if target_mode == TradingMode.LIVE:
            if not self._validate_live_mode_credentials():
                result['errors'].append("Invalid or missing Alpaca credentials for live trading")
                result['allowed'] = False
            
            result['warnings'].append("LIVE MODE: Real money will be at risk!")
            result['warnings'].append("Ensure you have proper risk management in place")
            result['confirmations_required'] = self.safety_checks['live_mode_confirmations_required']
        
        # Check if switching to simulation from live
        if self.current_mode == TradingMode.LIVE and target_mode == TradingMode.SIMULATION:
            result['warnings'].append("Switching from LIVE to SIMULATION - all trading will stop")
        
        return result
    
    def switch_mode(self, target_mode: TradingMode, confirmations: int = 0) -> Dict[str, Any]:
        """Safely switch trading mode with proper validation"""
        result = {
            'success': False,
            'message': '',
            'new_mode': target_mode.value,
            'warnings': []
        }
        
        # Validate switch is allowed
        validation = self.can_switch_to_mode(target_mode)
        if not validation['allowed']:
            result['message'] = f"Mode switch not allowed: {'; '.join(validation['errors'])}"
            return result
        
        # Check confirmations for live mode
        if target_mode == TradingMode.LIVE:
            if confirmations < validation['confirmations_required']:
                result['message'] = f"Live mode requires {validation['confirmations_required']} confirmations. Received: {confirmations}"
                return result
        
        # Perform the switch
        try:
            old_mode = self.current_mode
            self.current_mode = target_mode
            self.safety_checks['last_mode_switch'] = datetime.now()
            
            # Log the switch
            self.logger.info(f"Trading mode switched: {old_mode.value} -> {target_mode.value}")
            
            # Additional warnings
            if target_mode == TradingMode.LIVE:
                self.logger.warning("🚨 LIVE TRADING MODE ACTIVATED - REAL MONEY AT RISK 🚨")
                result['warnings'].append("LIVE TRADING ACTIVE - Monitor positions carefully")
            elif target_mode == TradingMode.PAPER:
                self.logger.info("📄 PAPER TRADING MODE - Safe testing environment")
            elif target_mode == TradingMode.SIMULATION:
                self.logger.info("🧪 SIMULATION MODE - Analysis only, no trades")
            
            result['success'] = True
            result['message'] = f"Successfully switched to {target_mode.value} mode"
            
        except Exception as e:
            result['message'] = f"Failed to switch mode: {str(e)}"
            self.logger.error(f"Mode switch failed: {e}")
        
        return result
    
    def get_safety_status(self) -> Dict[str, Any]:
        """Get current safety status"""
        config = self.get_current_config()
        return {
            'mode': self.current_mode.value,
            'real_money': config['real_money'],
            'safety_level': config['safety_level'],
            'trading_enabled': config['trading_enabled'],
            'emergency_stop_available': self.safety_checks['emergency_stop_enabled'],
            'last_mode_switch': self.safety_checks['last_mode_switch'].isoformat() if self.safety_checks['last_mode_switch'] else None,
            'cooldown_active': self._is_cooldown_active()
        }
    
    def _is_cooldown_active(self) -> bool:
        """Check if mode switch cooldown is active"""
        if not self.safety_checks['last_mode_switch']:
            return False
        
        time_since_switch = (datetime.now() - self.safety_checks['last_mode_switch']).total_seconds()
        return time_since_switch < self.safety_checks['mode_switch_cooldown']
    
    def emergency_stop(self) -> Dict[str, Any]:
        """Emergency stop all trading"""
        result = {
            'success': False,
            'message': '',
            'actions_taken': []
        }
        
        try:
            # Switch to simulation mode (no trading)
            switch_result = self.switch_mode(TradingMode.SIMULATION)
            if switch_result['success']:
                result['success'] = True
                result['message'] = "Emergency stop activated - All trading halted"
                result['actions_taken'].append("Switched to SIMULATION mode")
                result['actions_taken'].append("Trading disabled")
                self.logger.critical("🚨 EMERGENCY STOP ACTIVATED - ALL TRADING HALTED 🚨")
            else:
                result['message'] = f"Emergency stop failed: {switch_result['message']}"
                
        except Exception as e:
            result['message'] = f"Emergency stop error: {str(e)}"
            self.logger.error(f"Emergency stop failed: {e}")
        
        return result
    
    def validate_trading_action(self, action: str, symbol: str, quantity: float, price: float) -> Dict[str, Any]:
        """Validate trading action based on current mode"""
        result = {
            'allowed': True,
            'warnings': [],
            'modifications': {}
        }
        
        config = self.get_current_config()
        
        # Check if trading is enabled
        if not config['trading_enabled']:
            result['allowed'] = False
            result['warnings'].append(f"Trading disabled in {self.current_mode.value} mode")
            return result
        
        # Check position size limits
        max_position = config['max_position_size']
        if quantity > max_position:
            result['modifications']['quantity'] = max_position
            result['warnings'].append(f"Quantity reduced to {max_position} (mode limit)")
        
        # Live mode additional checks
        if self.current_mode == TradingMode.LIVE:
            result['warnings'].append("LIVE TRADING: Real money transaction")
            result['warnings'].append(f"Position: {symbol} x {quantity} @ ${price}")
        
        return result

