#!/usr/bin/env python3
"""
Comprehensive Trading System - Addresses All Issues
===================================================
- Separate stock and crypto trading
- Distinct agent trading styles
- Safe buying power management
- Stop-loss and take-profit orders
- No trading until $0 buying power
"""

import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
import asyncio

class ComprehensiveTradingSystem:
    """Comprehensive trading system addressing all requirements"""
    
    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self.safe_zone_multiplier = 0.8  # Only use 80% of buying power
        self.min_buying_power = 1000.0   # Minimum $1000 required
        self.max_daily_trades = 50        # Maximum trades per day
        self.daily_trade_count = 0
        self.last_reset_date = datetime.now().date()
        
        # Initialize separated systems
        self.stock_system = SeparatedTradingSystem(logger)
        self.crypto_system = SeparatedTradingSystem(logger)
        self.safe_interface = SafeTradingInterface(logger)
        
        # Initialize systems
        self.stock_system.initialize_agents()
        self.crypto_system.initialize_agents()
        
    def analyze_markets(self, stock_data: Dict[str, Any], crypto_data: Dict[str, Any]) -> Dict[str, List[Dict[str, Any]]]:
        """Analyze both stock and crypto markets separately"""
        results = {
            'stock_decisions': [],
            'crypto_decisions': [],
            'combined_decisions': []
        }
        
        try:
            # Analyze stock market
            if stock_data:
                stock_decisions = self.stock_system.analyze_stock_market(stock_data)
                results['stock_decisions'] = stock_decisions
                self.logger.info(f"Generated {len(stock_decisions)} stock trading decisions")
            
            # Analyze crypto market
            if crypto_data:
                crypto_decisions = self.crypto_system.analyze_crypto_market(crypto_data)
                results['crypto_decisions'] = crypto_decisions
                self.logger.info(f"Generated {len(crypto_decisions)} crypto trading decisions")
            
            # Combine decisions
            results['combined_decisions'] = stock_decisions + crypto_decisions
            
            # Log summary
            self.logger.info(f"Total decisions: {len(results['combined_decisions'])} "
                           f"(Stocks: {len(stock_decisions)}, Crypto: {len(crypto_decisions)})")
            
        except Exception as e:
            self.logger.error(f"Market analysis failed: {e}")
        
        return results
    
    def execute_safe_trades(self, decisions: List[Dict[str, Any]], account_info: Dict[str, Any]) -> Dict[str, Any]:
        """Execute trades with safety checks"""
        results = {
            'executed_trades': [],
            'failed_trades': [],
            'warnings': [],
            'summary': {}
        }
        
        try:
            # Check trading eligibility
            eligibility = self.safe_interface.check_trading_eligibility(account_info)
            if not eligibility['eligible']:
                results['warnings'].append(f"Trading not eligible: {eligibility['reason']}")
                return results
            
            # Process each decision
            for decision in decisions:
                try:
                    # Create trade request
                    trade_request = {
                        'symbol': decision['symbol'],
                        'action': decision['action'],
                        'quantity': decision['quantity'],
                        'price': decision['price']
                    }
                    
                    # Execute safe trade
                    trade_result = self.safe_interface.execute_safe_trade(trade_request, account_info)
                    
                    if trade_result['success']:
                        # Add stop-loss and take-profit
                        trade_result.update({
                            'stop_loss': decision.get('stop_loss', decision['price'] * 0.97),
                            'take_profit': decision.get('take_profit', decision['price'] * 1.06),
                            'agent_id': decision.get('agent_id', 'unknown'),
                            'style': decision.get('style', 'unknown'),
                            'market_type': decision.get('market_type', 'unknown')
                        })
                        
                        results['executed_trades'].append(trade_result)
                        self.logger.info(f"Trade executed: {decision['symbol']} {decision['quantity']:.2f} @ ${decision['price']:.2f}")
                    else:
                        results['failed_trades'].append({
                            'decision': decision,
                            'reason': trade_result['message']
                        })
                        self.logger.warning(f"Trade failed: {trade_result['message']}")
                
                except Exception as e:
                    results['failed_trades'].append({
                        'decision': decision,
                        'reason': f"Execution error: {str(e)}"
                    })
                    self.logger.error(f"Trade execution error: {e}")
            
            # Create summary
            results['summary'] = {
                'total_decisions': len(decisions),
                'executed_trades': len(results['executed_trades']),
                'failed_trades': len(results['failed_trades']),
                'success_rate': len(results['executed_trades']) / len(decisions) if decisions else 0,
                'daily_trade_count': self.safe_interface.daily_trade_count,
                'remaining_trades': self.safe_interface.max_daily_trades - self.safe_interface.daily_trade_count
            }
            
        except Exception as e:
            self.logger.error(f"Trade execution failed: {e}")
            results['warnings'].append(f"System error: {str(e)}")
        
        return results
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        return {
            'trading_enabled': True,
            'safe_zone_multiplier': self.safe_zone_multiplier,
            'min_buying_power': self.min_buying_power,
            'max_daily_trades': self.max_daily_trades,
            'daily_trade_count': self.daily_trade_count,
            'remaining_trades': self.max_daily_trades - self.daily_trade_count,
            'stock_system': self.stock_system.get_trading_summary(),
            'crypto_system': self.crypto_system.get_trading_summary(),
            'safety_status': self.safe_interface.get_safety_status()
        }
    
    def update_safety_parameters(self, safe_zone_multiplier: float = None,
                               min_buying_power: float = None,
                               max_daily_trades: int = None):
        """Update safety parameters"""
        if safe_zone_multiplier is not None:
            self.safe_zone_multiplier = max(0.1, min(0.9, safe_zone_multiplier))
            self.safe_interface.update_safety_parameters(safe_zone_multiplier=safe_zone_multiplier)
        
        if min_buying_power is not None:
            self.min_buying_power = max(100.0, min_buying_power)
            self.safe_interface.update_safety_parameters(min_buying_power=min_buying_power)
        
        if max_daily_trades is not None:
            self.max_daily_trades = max(1, min(100, max_daily_trades))
            self.safe_interface.update_safety_parameters(max_daily_trades=max_daily_trades)
        
        self.logger.info(f"Safety parameters updated: safe_zone={self.safe_zone_multiplier}, "
                        f"min_buying_power=${self.min_buying_power:,.2f}, "
                        f"max_daily_trades={self.max_daily_trades}")
    
    def reset_daily_limits(self):
        """Reset daily trading limits"""
        self.daily_trade_count = 0
        self.last_reset_date = datetime.now().date()
        self.safe_interface.reset_daily_limits()
        self.logger.info("Daily trading limits reset")
    
    def get_agent_performance(self) -> Dict[str, Any]:
        """Get agent performance summary"""
        return {
            'stock_agents': {
                'count': len(self.stock_system.stock_agents),
                'types': list(set(agent['type'] for agent in self.stock_system.stock_agents.values())),
                'symbols': self.stock_system.stock_symbols
            },
            'crypto_agents': {
                'count': len(self.crypto_system.crypto_agents),
                'types': list(set(agent['type'] for agent in self.crypto_system.crypto_agents.values())),
                'symbols': self.crypto_system.crypto_symbols
            }
        }
    
    def validate_system_health(self) -> Dict[str, Any]:
        """Validate system health"""
        health = {
            'healthy': True,
            'issues': [],
            'recommendations': []
        }
        
        try:
            # Check safety parameters
            if self.safe_zone_multiplier < 0.5:
                health['issues'].append("Safe zone multiplier too low")
                health['recommendations'].append("Increase safe zone multiplier to 0.8")
            
            if self.min_buying_power < 500:
                health['issues'].append("Minimum buying power too low")
                health['recommendations'].append("Increase minimum buying power to $1000")
            
            if self.max_daily_trades > 100:
                health['issues'].append("Daily trade limit too high")
                health['recommendations'].append("Reduce daily trade limit to 50")
            
            # Check agent configurations
            stock_agents = len(self.stock_system.stock_agents)
            crypto_agents = len(self.crypto_system.crypto_agents)
            
            if stock_agents == 0:
                health['issues'].append("No stock agents configured")
                health['recommendations'].append("Initialize stock agents")
            
            if crypto_agents == 0:
                health['issues'].append("No crypto agents configured")
                health['recommendations'].append("Initialize crypto agents")
            
            # Overall health
            if health['issues']:
                health['healthy'] = False
            
        except Exception as e:
            health['healthy'] = False
            health['issues'].append(f"Health check failed: {str(e)}")
        
        return health

