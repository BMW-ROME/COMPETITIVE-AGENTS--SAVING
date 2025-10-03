#!/usr/bin/env python3
"""
Pre-Risk Filter: Eliminates trades that will be rejected by risk limits
This prevents attempting trades that will fail, achieving zero warnings
"""

import asyncio
import alpaca_trade_api as tradeapi
import os
from dotenv import load_dotenv
from typing import Dict, List
import logging

class PreRiskFilter:
    """Pre-filters trades to prevent risk_limits rejections"""
    
    def __init__(self, api):
        self.api = api
        self.logger = logging.getLogger('PreRiskFilter')
        
    def check_account_constraints(self) -> Dict:
        """Check current account constraints"""
        try:
            account = self.api.get_account()
            positions = self.api.list_positions()
            
            return {
                'buying_power': float(account.buying_power),
                'portfolio_value': float(account.portfolio_value),
                'day_trade_count': getattr(account, 'daytrade_buying_power', 0),
                'position_count': len(positions),
                'equity': float(account.equity),
                'status': getattr(account, 'status', 'ACTIVE')
            }
        except Exception as e:
            self.logger.warning(f"Failed to get account constraints: {e}")
            return {}
    
    def will_pass_risk_check(self, trade_candidate: Dict, constraints: Dict) -> tuple:
        """
        Predict if trade will pass risk management checks
        Returns (will_pass: bool, reason: str)
        """
        if not constraints:
            return False, "no_account_data"
            
        symbol = trade_candidate['symbol']
        trade_value = trade_candidate.get('trade_value', 0)
        
        # Check buying power - much more lenient
        if trade_value > constraints['buying_power'] * 0.95:  # 95% threshold
            return False, "insufficient_buying_power"
            
        # Check if account is restricted
        if constraints['status'] != 'ACTIVE':
            return False, "account_restricted"
            
        # Check position limits - more lenient
        if constraints['position_count'] >= 15:  # Higher limit
            return False, "too_many_positions"
            
        # Check minimum trade size only (remove PDT limit for paper trading)
        if trade_value < 1.0:  # Minimum $1 trades
            return False, "trade_too_small"
            
        # Check maximum single trade size
        if trade_value > 50.0:  # Maximum $50 per trade for safety
            return False, "trade_too_large"
            
        return True, "passed"
    
    def filter_trades_for_zero_rejections(self, trade_candidates: List[Dict]) -> List[Dict]:
        """Filter trades to ensure zero rejections"""
        constraints = self.check_account_constraints()
        
        if not constraints:
            self.logger.warning("No account constraints available - skipping all trades")
            return []
            
        passed_trades = []
        filtered_count = 0
        
        for candidate in trade_candidates:
            will_pass, reason = self.will_pass_risk_check(candidate, constraints)
            
            if will_pass:
                passed_trades.append(candidate)
            else:
                filtered_count += 1
                self.logger.debug(f"Pre-filtered {candidate['symbol']}: {reason}")
        
        if passed_trades:
            self.logger.info(f"✅ Pre-Risk Filter: {len(passed_trades)} trades approved, "
                           f"{filtered_count} filtered out")
        else:
            self.logger.info("🔒 Pre-Risk Filter: All trades filtered - protecting capital")
            
        return passed_trades

def get_pre_risk_filter(api):
    """Get pre-risk filter instance"""
    return PreRiskFilter(api)