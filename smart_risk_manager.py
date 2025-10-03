#!/usr/bin/env python3
"""
Smart Risk Manager - Adaptive Trading Constraints
Dynamically adjusts trading parameters based on account conditions
"""

import json
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class SmartRiskManager:
    def __init__(self, account_info):
        self.account = account_info
        self.buying_power = float(account_info.buying_power)
        self.equity = float(account_info.equity)
        self.day_trading_power = float(account_info.daytrading_buying_power)
        self.pattern_day_trader = account_info.pattern_day_trader
        
    def calculate_risk_profile(self):
        """Calculate current risk profile based on account metrics"""
        
        # Buying power ratio (key metric)
        bp_ratio = self.buying_power / self.equity if self.equity > 0 else 0
        
        # Risk levels based on buying power ratio
        if bp_ratio >= 0.10:  # 10%+ buying power
            risk_level = "AGGRESSIVE"
        elif bp_ratio >= 0.05:  # 5-10% buying power
            risk_level = "BALANCED"
        elif bp_ratio >= 0.02:  # 2-5% buying power
            risk_level = "CONSERVATIVE"
        else:  # <2% buying power
            risk_level = "MINIMAL"
            
        return {
            'risk_level': risk_level,
            'buying_power_ratio': bp_ratio,
            'buying_power': self.buying_power,
            'equity': self.equity,
            'is_pattern_day_trader': self.pattern_day_trader
        }
    
    def get_trading_parameters(self):
        """Get adaptive trading parameters based on risk profile"""
        
        risk_profile = self.calculate_risk_profile()
        risk_level = risk_profile['risk_level']
        
        # Base parameters by risk level
        parameters = {
            'AGGRESSIVE': {
                'max_trade_value': min(self.buying_power * 0.05, 500),  # 5% of BP, max $500
                'max_position_pct': 0.15,  # 15% of portfolio per position
                'max_total_exposure': 0.8,  # 80% of portfolio in positions
                'min_execution_threshold': 0.25,  # 25% execution probability
                'position_size_multiplier': 1.0,
                'aggressive_exits': False
            },
            'BALANCED': {
                'max_trade_value': min(self.buying_power * 0.03, 250),  # 3% of BP, max $250
                'max_position_pct': 0.12,  # 12% of portfolio per position
                'max_total_exposure': 0.7,  # 70% of portfolio in positions
                'min_execution_threshold': 0.35,  # 35% execution probability
                'position_size_multiplier': 0.8,
                'aggressive_exits': False
            },
            'CONSERVATIVE': {
                'max_trade_value': min(self.buying_power * 0.02, 100),  # 2% of BP, max $100
                'max_position_pct': 0.08,  # 8% of portfolio per position
                'max_total_exposure': 0.6,  # 60% of portfolio in positions
                'min_execution_threshold': 0.45,  # 45% execution probability
                'position_size_multiplier': 0.6,
                'aggressive_exits': True
            },
            'MINIMAL': {
                'max_trade_value': min(self.buying_power * 0.01, 50),   # 1% of BP, max $50
                'max_position_pct': 0.05,  # 5% of portfolio per position
                'max_total_exposure': 0.4,  # 40% of portfolio in positions
                'min_execution_threshold': 0.60,  # 60% execution probability
                'position_size_multiplier': 0.4,
                'aggressive_exits': True
            }
        }
        
        config = parameters[risk_level]
        
        # Add account-specific adjustments
        config.update({
            'risk_level': risk_level,
            'buying_power_ratio': risk_profile['buying_power_ratio'],
            'timestamp': datetime.now().isoformat(),
            'account_equity': self.equity,
            'available_buying_power': self.buying_power
        })
        
        return config
    
    def should_allow_new_trades(self, positions=None):
        """Determine if new trades should be allowed"""
        
        risk_profile = self.calculate_risk_profile()
        
        # Never trade if buying power is extremely low
        if self.buying_power < 50:
            return False, "Insufficient buying power (<$50)"
        
        # Check if we're over-leveraged
        if risk_profile['buying_power_ratio'] < 0.01:  # Less than 1%
            return False, f"Over-leveraged (BP ratio: {risk_profile['buying_power_ratio']:.1%})"
        
        # Pattern day trader restrictions
        if not self.pattern_day_trader and self.equity < 25000:
            # More restrictive for non-PDT accounts
            if risk_profile['buying_power_ratio'] < 0.02:
                return False, "Non-PDT account needs higher buying power ratio"
        
        return True, f"Approved for {risk_profile['risk_level']} trading"
    
    def optimize_trade_size(self, proposed_quantity, symbol_price, confidence=0.5):
        """Optimize trade size based on current risk parameters"""
        
        params = self.get_trading_parameters()
        
        # Calculate proposed trade value
        proposed_value = proposed_quantity * symbol_price
        
        # Apply maximum trade value limit
        max_allowed_value = params['max_trade_value']
        
        if proposed_value > max_allowed_value:
            # Scale down the quantity
            optimized_quantity = (max_allowed_value / symbol_price) * 0.95  # 5% buffer
            
            logger.info(f"🔧 Trade size reduced: ${proposed_value:.2f} → ${optimized_quantity * symbol_price:.2f}")
        else:
            optimized_quantity = proposed_quantity
        
        # Apply position size multiplier based on risk level
        optimized_quantity *= params['position_size_multiplier']
        
        # Apply confidence-based adjustment
        confidence_multiplier = 0.5 + (confidence * 0.5)  # 0.5 to 1.0 range
        optimized_quantity *= confidence_multiplier
        
        return max(optimized_quantity, 0.001)  # Minimum position size
    
    def save_risk_config(self, filepath="risk_config.json"):
        """Save current risk configuration to file"""
        
        config = {
            'trading_parameters': self.get_trading_parameters(),
            'risk_profile': self.calculate_risk_profile(),
            'account_snapshot': {
                'equity': self.equity,
                'buying_power': self.buying_power,
                'day_trading_power': self.day_trading_power,
                'pattern_day_trader': self.pattern_day_trader
            }
        }
        
        with open(filepath, 'w') as f:
            json.dump(config, f, indent=2)
        
        logger.info(f"💾 Risk configuration saved to {filepath}")
        return filepath

def load_risk_config(filepath="risk_config.json"):
    """Load risk configuration from file"""
    try:
        with open(filepath, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return None

if __name__ == "__main__":
    # Example usage
    import alpaca_trade_api as tradeapi
    import os
    from dotenv import load_dotenv
    
    load_dotenv()
    api = tradeapi.REST(
        os.getenv('APCA_API_KEY_ID'),
        os.getenv('APCA_API_SECRET_KEY'), 
        'https://paper-api.alpaca.markets'
    )
    
    account = api.get_account()
    risk_manager = SmartRiskManager(account)
    
    print("🎯 SMART RISK MANAGER ANALYSIS")
    print("=" * 40)
    
    # Get current parameters
    params = risk_manager.get_trading_parameters()
    print(f"Risk Level: {params['risk_level']}")
    print(f"Max Trade Value: ${params['max_trade_value']:.2f}")
    print(f"Execution Threshold: {params['min_execution_threshold']:.0%}")
    print(f"Position Size Multiplier: {params['position_size_multiplier']:.1f}x")
    
    # Check trading approval
    can_trade, reason = risk_manager.should_allow_new_trades()
    print(f"Trading Status: {'✅ APPROVED' if can_trade else '❌ RESTRICTED'}")
    print(f"Reason: {reason}")
    
    # Save configuration
    risk_manager.save_risk_config()