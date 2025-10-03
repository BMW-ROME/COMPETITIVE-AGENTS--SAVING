#!/usr/bin/env python3
"""
Adaptive Risk Manager - Adjust trading parameters based on account constraints
"""
import alpaca_trade_api as tradeapi
import os
from dotenv import load_dotenv

class AdaptiveRiskManager:
    def __init__(self):
        load_dotenv()
        self.api = tradeapi.REST(
            os.getenv('APCA_API_KEY_ID'),
            os.getenv('APCA_API_SECRET_KEY'), 
            'https://paper-api.alpaca.markets'
        )
        self.min_buying_power = 5000  # Minimum required
        self.max_position_pct = 0.02  # 2% max per position
        
    def get_account_constraints(self):
        """Get current account constraints"""
        account = self.api.get_account()
        positions = self.api.list_positions()
        
        buying_power = float(account.buying_power)
        equity = float(account.equity)
        position_count = len(positions)
        
        return {
            'buying_power': buying_power,
            'equity': equity,
            'position_count': position_count,
            'available_pct': buying_power / equity if equity > 0 else 0,
            'can_trade': buying_power > self.min_buying_power
        }
    
    def calculate_max_trade_size(self, symbol_price=None):
        """Calculate maximum trade size based on constraints"""
        constraints = self.get_account_constraints()
        
        if not constraints['can_trade']:
            return 0, "Insufficient buying power"
        
        # Conservative sizing based on available capital
        max_dollar_amount = min(
            constraints['buying_power'] * 0.1,  # 10% of buying power
            constraints['equity'] * self.max_position_pct,  # 2% of equity
            2000  # Hard cap at $2000 per trade
        )
        
        if symbol_price:
            max_shares = max_dollar_amount / symbol_price
            return max_shares, f"Max ${max_dollar_amount:.0f} (${symbol_price:.2f}/share)"
        
        return max_dollar_amount, "Max dollar amount"
    
    def should_allow_trade(self, symbol, quantity, price):
        """Check if trade should be allowed"""
        constraints = self.get_account_constraints()
        
        if not constraints['can_trade']:
            return False, "insufficient_buying_power"
        
        trade_value = quantity * price
        max_trade, _ = self.calculate_max_trade_size()
        
        if trade_value > max_trade:
            return False, "position_size_limit"
        
        # Check position concentration
        if constraints['position_count'] >= 15:
            return False, "max_positions_reached"
        
        return True, "approved"
    
    def get_recommended_settings(self):
        """Get recommended trading system settings"""
        constraints = self.get_account_constraints()
        
        if constraints['buying_power'] < 1000:
            return {
                'max_trade_value': 50,
                'cycle_interval': 120,  # Slower trading
                'enabled_agents': 2,
                'risk_multiplier': 0.1
            }
        elif constraints['buying_power'] < 5000:
            return {
                'max_trade_value': 200,
                'cycle_interval': 90,
                'enabled_agents': 4,
                'risk_multiplier': 0.3
            }
        else:
            return {
                'max_trade_value': min(2000, constraints['buying_power'] * 0.1),
                'cycle_interval': 45,
                'enabled_agents': 12,
                'risk_multiplier': 1.0
            }

def main():
    rm = AdaptiveRiskManager()
    
    print("🔒 ADAPTIVE RISK MANAGER")
    print("=" * 40)
    
    constraints = rm.get_account_constraints()
    print(f"💰 Buying Power: ${constraints['buying_power']:,.2f}")
    print(f"📊 Equity: ${constraints['equity']:,.2f}")
    print(f"📈 Positions: {constraints['position_count']}")
    print(f"🟢 Can Trade: {constraints['can_trade']}")
    
    settings = rm.get_recommended_settings()
    print(f"\n⚙️ RECOMMENDED SETTINGS:")
    for key, value in settings.items():
        print(f"   {key}: {value}")
    
    # Test some trade scenarios
    test_symbols = [('AAPL', 254), ('SPY', 577), ('QQQ', 597)]
    
    print(f"\n🧪 TRADE APPROVAL TESTS:")
    for symbol, price in test_symbols:
        max_shares, reason = rm.calculate_max_trade_size(price)
        print(f"   {symbol} @ ${price}: Max {max_shares:.4f} shares ({reason})")
        
        # Test small trade
        test_qty = max_shares * 0.5 if max_shares > 0 else 0.01
        approved, reason = rm.should_allow_trade(symbol, test_qty, price)
        status = "✅" if approved else "❌"
        print(f"   {status} Test trade {test_qty:.4f} shares: {reason}")

if __name__ == "__main__":
    main()