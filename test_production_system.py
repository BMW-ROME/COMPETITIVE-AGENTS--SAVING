#!/usr/bin/env python3
"""
Production System Test - Final Validation
========================================
Test the production-ready system to ensure everything works perfectly.
"""

import asyncio
import logging
import sys
import os
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)

logger = logging.getLogger("ProductionTest")

async def test_production_system():
    """Test the production system"""
    try:
        logger.info("🏭 TESTING PRODUCTION SYSTEM")
        logger.info("=" * 50)
        
        # Import production system
        from production_trading_system import ProductionTradingSystem
        
        # Create system
        logger.info("Creating production trading system...")
        system = ProductionTradingSystem()
        
        # Run test cycle
        logger.info("Running production test cycle...")
        cycle_result = await system.run_competitive_cycle()
        
        # Analyze results
        account_info = cycle_result.get('account_info', {})
        total_decisions = cycle_result.get('total_decisions', 0)
        selected_trades = len(cycle_result.get('selected_trades', []))
        executed_trades = len(cycle_result.get('executed_trades', []))
        
        logger.info("=" * 50)
        logger.info("🏭 PRODUCTION TEST RESULTS:")
        logger.info(f"   💰 Buying Power: ${account_info.get('buying_power', 0):,.2f}")
        logger.info(f"   📊 Portfolio Value: ${account_info.get('portfolio_value', 0):,.2f}")
        logger.info(f"   🎯 Agent Decisions: {total_decisions}/12")
        logger.info(f"   ✅ Trades Selected: {selected_trades}")
        logger.info(f"   🚀 Trades Executed: {executed_trades}")
        logger.info(f"   📊 Current Positions: {account_info.get('positions_count', 0)}")
        logger.info(f"   ⏱️ Cycle Duration: {cycle_result.get('cycle_duration', 0):.2f}s")
        
        # Success criteria
        success_score = 0
        if total_decisions >= 3:  # At least 25% agents active
            success_score += 2
            logger.info("   ✅ Agent activity: GOOD")
        elif total_decisions >= 1:
            success_score += 1
            logger.info("   ⚠️ Agent activity: ACCEPTABLE")
        else:
            logger.info("   ❌ Agent activity: POOR")
        
        if selected_trades > 0:
            success_score += 2
            logger.info("   ✅ Trade selection: WORKING")
        else:
            logger.info("   ❌ Trade selection: NOT WORKING")
        
        if executed_trades > 0:
            success_score += 3
            logger.info("   ✅ Trade execution: SUCCESS")
        else:
            logger.info("   ❌ Trade execution: FAILED")
        
        if account_info.get('buying_power', 0) > 0:
            success_score += 1
            logger.info("   ✅ Account connection: ACTIVE")
        else:
            logger.info("   ❌ Account connection: INACTIVE")
        
        # Overall assessment
        logger.info("=" * 50)
        if success_score >= 7:
            logger.info("🏆 PRODUCTION SYSTEM: EXCELLENT - READY FOR LIVE TRADING!")
        elif success_score >= 5:
            logger.info("✅ PRODUCTION SYSTEM: GOOD - MINOR OPTIMIZATIONS NEEDED")
        elif success_score >= 3:
            logger.info("⚠️ PRODUCTION SYSTEM: ACCEPTABLE - NEEDS IMPROVEMENT")
        else:
            logger.info("❌ PRODUCTION SYSTEM: NEEDS WORK")
        
        logger.info(f"   Overall Score: {success_score}/8")
        
        # Show trade details
        executed_trades_list = cycle_result.get('executed_trades', [])
        if executed_trades_list:
            logger.info("\n💰 EXECUTED TRADES:")
            for i, trade in enumerate(executed_trades_list[:5], 1):
                logger.info(f"   {i}. {trade['agent_id']}: {trade['symbol']} {trade['action']} {trade['quantity']} @ ${trade['price']:.2f}")
        
        logger.info("=" * 50)
        logger.info("Production test completed!")
        
    except Exception as e:
        logger.error(f"❌ Production test failed: {e}")
        import traceback
        logger.error(traceback.format_exc())

if __name__ == "__main__":
    # Create logs directory
    os.makedirs('logs', exist_ok=True)
    
    # Run test
    asyncio.run(test_production_system())