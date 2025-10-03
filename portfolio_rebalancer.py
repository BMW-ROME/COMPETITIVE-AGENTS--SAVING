#!/usr/bin/env python3
"""
Portfolio Rebalancer - Strategic Position Reduction
Intelligently reduces oversized positions to free up buying power
"""

import alpaca_trade_api as tradeapi
import os
from dotenv import load_dotenv
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PortfolioRebalancer:
    def __init__(self):
        load_dotenv()
        self.api = tradeapi.REST(
            os.getenv('APCA_API_KEY_ID'),
            os.getenv('APCA_API_SECRET_KEY'), 
            'https://paper-api.alpaca.markets'
        )
        
    def analyze_portfolio(self):
        """Analyze current portfolio and identify rebalancing opportunities"""
        account = self.api.get_account()
        positions = self.api.list_positions()
        
        portfolio_value = float(account.equity)
        buying_power = float(account.buying_power)
        
        logger.info(f"💰 Portfolio Value: ${portfolio_value:,.2f}")
        logger.info(f"💳 Buying Power: ${buying_power:,.2f}")
        logger.info(f"📊 Buying Power Ratio: {(buying_power/portfolio_value)*100:.1f}%")
        
        position_analysis = []
        total_unrealized_pnl = 0
        
        for pos in positions:
            symbol = pos.symbol
            qty = float(pos.qty)
            market_value = float(pos.market_value)
            unrealized_pnl = float(pos.unrealized_pl or 0.0)
            pnl_pct = float(pos.unrealized_plpc or 0.0) * 100
            
            position_pct = (market_value / portfolio_value) * 100
            
            position_analysis.append({
                'symbol': symbol,
                'quantity': qty,
                'market_value': market_value,
                'unrealized_pnl': unrealized_pnl,
                'pnl_pct': pnl_pct,
                'portfolio_pct': position_pct,
                'side': pos.side
            })
            
            total_unrealized_pnl += unrealized_pnl
            
        # Sort by position size (largest first)
        position_analysis.sort(key=lambda x: x['market_value'], reverse=True)
        
        logger.info(f"\n📈 Total Unrealized P&L: ${total_unrealized_pnl:,.2f}")
        logger.info("🎯 POSITION ANALYSIS:")
        for pos in position_analysis:
            logger.info(f"  {pos['symbol']}: ${pos['market_value']:,.0f} "
                       f"({pos['portfolio_pct']:.1f}%) | "
                       f"P&L: ${pos['unrealized_pnl']:,.2f} ({pos['pnl_pct']:+.1f}%)")
        
        return position_analysis, buying_power, portfolio_value
    
    def create_rebalancing_plan(self, positions, buying_power, portfolio_value):
        """Create intelligent rebalancing plan"""
        target_buying_power = portfolio_value * 0.05  # Target 5% buying power
        needed_reduction = target_buying_power - buying_power
        
        logger.info(f"\n🎯 REBALANCING PLAN:")
        logger.info(f"  Target Buying Power: ${target_buying_power:,.2f} (5%)")
        logger.info(f"  Current Buying Power: ${buying_power:,.2f}")
        logger.info(f"  Needed Reduction: ${needed_reduction:,.2f}")
        
        if needed_reduction <= 0:
            logger.info("✅ No rebalancing needed - sufficient buying power")
            return []
        
        rebalancing_actions = []
        remaining_reduction = needed_reduction
        
        # Strategy: Reduce largest positions first, prioritize losers
        for pos in positions:
            if remaining_reduction <= 0:
                break
                
            # Skip very small positions
            if pos['market_value'] < 1000:
                continue
                
            # Calculate reduction percentage based on position characteristics
            reduction_pct = 0.0
            
            # Large positions (>10% of portfolio) - reduce by 25-50%
            if pos['portfolio_pct'] > 10:
                reduction_pct = 0.4 if pos['unrealized_pnl'] < 0 else 0.25
            
            # Medium positions (5-10% of portfolio) - reduce by 15-30%
            elif pos['portfolio_pct'] > 5:
                reduction_pct = 0.25 if pos['unrealized_pnl'] < 0 else 0.15
            
            # Smaller positions - reduce by 10-20% only if losing
            elif pos['unrealized_pnl'] < -100:
                reduction_pct = 0.15
            
            if reduction_pct > 0:
                reduction_qty = pos['quantity'] * reduction_pct
                reduction_value = pos['market_value'] * reduction_pct
                
                reason = f"Reduce {pos['portfolio_pct']:.1f}% position by {reduction_pct*100:.0f}%"
                
                rebalancing_actions.append({
                    'symbol': pos['symbol'],
                    'action': 'sell' if pos['side'] == 'long' else 'buy',
                    'quantity': abs(reduction_qty),
                    'estimated_value': reduction_value,
                    'reason': reason
                })
                
                remaining_reduction -= reduction_value
                
                logger.info(f"  📉 {pos['symbol']}: Sell {reduction_qty:.4f} shares "
                           f"(~${reduction_value:,.0f}) - {reason}")
        
        total_estimated_reduction = sum(action['estimated_value'] for action in rebalancing_actions)
        logger.info(f"\n💡 Total Estimated Reduction: ${total_estimated_reduction:,.2f}")
        logger.info(f"🎯 New Estimated Buying Power: ${buying_power + total_estimated_reduction:,.2f}")
        
        return rebalancing_actions
    
    def execute_rebalancing(self, actions, dry_run=True):
        """Execute rebalancing plan"""
        if dry_run:
            logger.info("\n🔍 DRY RUN MODE - No actual trades will be executed")
        else:
            logger.info("\n🚀 EXECUTING REBALANCING PLAN")
        
        executed_actions = []
        
        for action in actions:
            try:
                if not dry_run:
                    order = self.api.submit_order(
                        symbol=action['symbol'],
                        qty=action['quantity'],
                        side=action['action'],
                        type='market',
                        time_in_force='day'
                    )
                    
                    if order and hasattr(order, 'id'):
                        logger.info(f"✅ EXECUTED: {action['symbol']} {action['action'].upper()} "
                                   f"{action['quantity']:.4f} | Order ID: {order.id}")
                        executed_actions.append({**action, 'order_id': order.id, 'status': 'submitted'})
                    else:
                        logger.warning(f"❌ FAILED: {action['symbol']} - Order not submitted")
                else:
                    logger.info(f"🔍 WOULD EXECUTE: {action['symbol']} {action['action'].upper()} "
                               f"{action['quantity']:.4f} (~${action['estimated_value']:,.0f})")
                    executed_actions.append({**action, 'status': 'dry_run'})
                    
            except Exception as e:
                logger.error(f"❌ ERROR executing {action['symbol']}: {e}")
                
        return executed_actions
    
    def run_rebalancing(self, dry_run=True):
        """Run complete rebalancing process"""
        logger.info("🚀 PORTFOLIO REBALANCER STARTING")
        logger.info("=" * 50)
        
        # Analyze current portfolio
        positions, buying_power, portfolio_value = self.analyze_portfolio()
        
        # Create rebalancing plan
        actions = self.create_rebalancing_plan(positions, buying_power, portfolio_value)
        
        if not actions:
            logger.info("✅ No rebalancing actions needed")
            return []
        
        # Execute rebalancing
        executed = self.execute_rebalancing(actions, dry_run=dry_run)
        
        logger.info("\n📊 REBALANCING COMPLETE")
        logger.info("=" * 50)
        
        return executed

if __name__ == "__main__":
    rebalancer = PortfolioRebalancer()
    
    # Run in dry-run mode first
    print("🔍 Running Portfolio Analysis & Rebalancing Plan...")
    executed_actions = rebalancer.run_rebalancing(dry_run=True)
    
    if executed_actions:
        print(f"\n💡 Generated {len(executed_actions)} rebalancing actions")
        print("\n🚀 To execute for real, run:")
        print("python3 portfolio_rebalancer.py --execute")
        
        # Ask for confirmation if running interactively
        try:
            response = input("\n❓ Execute rebalancing plan now? (yes/no): ").strip().lower()
            if response in ['yes', 'y']:
                print("\n🚀 EXECUTING REBALANCING PLAN...")
                rebalancer.run_rebalancing(dry_run=False)
        except (EOFError, KeyboardInterrupt):
            print("\n👋 Exiting...")