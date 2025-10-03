#!/usr/bin/env python3
"""
Safe Trading Interface - Respects Buying Power Limits
=====================================================
Ensures trading never exceeds safe buying power limits.
"""

import logging
from typing import Dict, Any, Optional, List
from datetime import datetime
import asyncio

class SafeTradingInterface:
    """Safe trading interface with buying power protection"""
    
    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self.safe_zone_multiplier = 0.8  # Only use 80% of available buying power
        self.min_buying_power = 1000.0   # Minimum $1000 buying power required
        self.max_daily_trades = 50       # Maximum trades per day
        self.daily_trade_count = 0
        self.last_reset_date = datetime.now().date()
        
    def validate_trade(self, trade_request: Dict[str, Any], account_info: Dict[str, Any]) -> Dict[str, Any]:
        """Validate trade request against safety limits"""
        result = {
            'valid': False,
            'reason': '',
            'adjusted_quantity': 0,
            'warnings': []
        }
        
        try:
            # Check if it's a new day (reset daily trade count)
            current_date = datetime.now().date()
            if current_date != self.last_reset_date:
                self.daily_trade_count = 0
                self.last_reset_date = current_date
            
            # Get account information
            buying_power = float(account_info.get('buying_power', 0.0) or 0.0)
            cash = float(account_info.get('cash', 0.0) or 0.0)
            portfolio_value = float(account_info.get('portfolio_value', 0.0) or 0.0)
            
            # Check minimum buying power
            if buying_power < self.min_buying_power:
                result['reason'] = f"Insufficient buying power. Required: ${self.min_buying_power:,.2f}, Available: ${buying_power:,.2f}"
                return result
            
            # Check daily trade limit
            if self.daily_trade_count >= self.max_daily_trades:
                result['reason'] = f"Daily trade limit reached. Max: {self.max_daily_trades}, Current: {self.daily_trade_count}"
                return result
            
            # Calculate trade details
            symbol = trade_request.get('symbol', '')
            quantity = float(trade_request.get('quantity', 0.0))
            price = float(trade_request.get('price', 0.0))
            action = trade_request.get('action', 'BUY')
            
            if action != 'BUY':
                result['reason'] = "Only BUY orders supported in paper trading"
                return result
            
            # Calculate required capital
            required_capital = quantity * price
            
            # Apply safe zone multiplier
            safe_buying_power = buying_power * self.safe_zone_multiplier
            
            # Check if trade fits within safe zone
            if required_capital > safe_buying_power:
                # Adjust quantity to fit within safe zone
                adjusted_quantity = (safe_buying_power / price) * 0.95  # 95% of safe zone
                result['adjusted_quantity'] = adjusted_quantity
                result['warnings'].append(f"Quantity reduced to fit safe zone: {quantity:.2f} -> {adjusted_quantity:.2f}")
                quantity = adjusted_quantity
                required_capital = quantity * price
            
            # Final validation
            if required_capital > safe_buying_power:
                result['reason'] = f"Trade too large for safe zone. Required: ${required_capital:,.2f}, Safe zone: ${safe_buying_power:,.2f}"
                return result
            
            # Check position size limits
            max_position_value = portfolio_value * 0.05  # 5% of portfolio max
            if required_capital > max_position_value:
                result['reason'] = f"Position too large. Required: ${required_capital:,.2f}, Max position: ${max_position_value:,.2f}"
                return result
            
            # All checks passed
            result['valid'] = True
            result['adjusted_quantity'] = quantity
            result['required_capital'] = required_capital
            result['safe_buying_power'] = safe_buying_power
            result['remaining_buying_power'] = safe_buying_power - required_capital
            
            # Log the trade
            self.logger.info(f"Safe trade validated: {symbol} {quantity:.2f} @ ${price:.2f} = ${required_capital:,.2f}")
            
        except Exception as e:
            result['reason'] = f"Validation error: {str(e)}"
            self.logger.error(f"Trade validation failed: {e}")
        
        return result
    
    def execute_safe_trade(self, trade_request: Dict[str, Any], account_info: Dict[str, Any]) -> Dict[str, Any]:
        """Execute trade with safety checks"""
        result = {
            'success': False,
            'trade_id': None,
            'message': '',
            'warnings': []
        }
        
        try:
            # Validate trade first
            validation = self.validate_trade(trade_request, account_info)
            
            if not validation['valid']:
                result['message'] = validation['reason']
                return result
            
            # Update quantity if adjusted
            if validation['adjusted_quantity'] > 0:
                trade_request['quantity'] = validation['adjusted_quantity']
                result['warnings'] = validation['warnings']
            
            # Execute the trade (simulated for paper trading)
            trade_id = f"safe_trade_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            # Increment daily trade count
            self.daily_trade_count += 1
            
            result['success'] = True
            result['trade_id'] = trade_id
            result['message'] = f"Safe trade executed: {trade_request['symbol']} {trade_request['quantity']:.2f} @ ${trade_request['price']:.2f}"
            
            self.logger.info(f"Safe trade executed: {trade_id}")
            
        except Exception as e:
            result['message'] = f"Trade execution failed: {str(e)}"
            self.logger.error(f"Safe trade execution failed: {e}")
        
        return result
    
    def get_safety_status(self) -> Dict[str, Any]:
        """Get current safety status"""
        return {
            'safe_zone_multiplier': self.safe_zone_multiplier,
            'min_buying_power': self.min_buying_power,
            'max_daily_trades': self.max_daily_trades,
            'daily_trade_count': self.daily_trade_count,
            'remaining_trades': self.max_daily_trades - self.daily_trade_count,
            'last_reset_date': self.last_reset_date.isoformat()
        }
    
    def reset_daily_limits(self):
        """Reset daily trading limits"""
        self.daily_trade_count = 0
        self.last_reset_date = datetime.now().date()
        self.logger.info("Daily trading limits reset")
    
    def update_safety_parameters(self, safe_zone_multiplier: float = None, 
                                min_buying_power: float = None, 
                                max_daily_trades: int = None):
        """Update safety parameters"""
        if safe_zone_multiplier is not None:
            self.safe_zone_multiplier = max(0.1, min(0.9, safe_zone_multiplier))
            self.logger.info(f"Safe zone multiplier updated: {self.safe_zone_multiplier}")
        
        if min_buying_power is not None:
            self.min_buying_power = max(100.0, min_buying_power)
            self.logger.info(f"Minimum buying power updated: ${self.min_buying_power:,.2f}")
        
        if max_daily_trades is not None:
            self.max_daily_trades = max(1, min(100, max_daily_trades))
            self.logger.info(f"Maximum daily trades updated: {self.max_daily_trades}")
    
    def calculate_safe_position_size(self, symbol: str, price: float, 
                                   account_info: Dict[str, Any], 
                                   max_position_pct: float = 0.05) -> float:
        """Calculate safe position size for a symbol"""
        try:
            buying_power = float(account_info.get('buying_power', 0.0) or 0.0)
            portfolio_value = float(account_info.get('portfolio_value', 0.0) or 0.0)
            
            # Calculate safe buying power
            safe_buying_power = buying_power * self.safe_zone_multiplier
            
            # Calculate max position value
            max_position_value = portfolio_value * max_position_pct
            
            # Use the smaller of the two
            available_capital = min(safe_buying_power, max_position_value)
            
            # Calculate quantity
            if price > 0:
                quantity = (available_capital / price) * 0.95  # 95% of available capital
                return max(0, quantity)
            
            return 0.0
            
        except Exception as e:
            self.logger.error(f"Error calculating safe position size: {e}")
            return 0.0
    
    def check_trading_eligibility(self, account_info: Dict[str, Any]) -> Dict[str, Any]:
        """Check if account is eligible for trading"""
        result = {
            'eligible': False,
            'reason': '',
            'recommendations': []
        }
        
        try:
            buying_power = float(account_info.get('buying_power', 0.0) or 0.0)
            cash = float(account_info.get('cash', 0.0) or 0.0)
            portfolio_value = float(account_info.get('portfolio_value', 0.0) or 0.0)
            
            # Check minimum requirements
            if buying_power < self.min_buying_power:
                result['reason'] = f"Insufficient buying power. Required: ${self.min_buying_power:,.2f}, Available: ${buying_power:,.2f}"
                result['recommendations'].append("Add funds to account")
                return result
            
            if cash < 100.0:
                result['reason'] = f"Insufficient cash. Required: $100.00, Available: ${cash:,.2f}"
                result['recommendations'].append("Add cash to account")
                return result
            
            if portfolio_value < 1000.0:
                result['reason'] = f"Portfolio too small. Required: $1,000.00, Available: ${portfolio_value:,.2f}"
                result['recommendations'].append("Build portfolio value")
                return result
            
            # Check daily limits
            if self.daily_trade_count >= self.max_daily_trades:
                result['reason'] = f"Daily trade limit reached. Max: {self.max_daily_trades}, Current: {self.daily_trade_count}"
                result['recommendations'].append("Wait for daily reset")
                return result
            
            # All checks passed
            result['eligible'] = True
            result['reason'] = "Account eligible for trading"
            result['recommendations'].append("Monitor position sizes")
            result['recommendations'].append("Set stop losses")
            
        except Exception as e:
            result['reason'] = f"Eligibility check failed: {str(e)}"
            self.logger.error(f"Trading eligibility check failed: {e}")
        
        return result

