#!/usr/bin/env python3
"""
Quick System Test - Validate Trading System Fixes
=================================================
This script will run a quick test to ensure agents are making decisions.
"""

import asyncio
import logging
import sys
import os
from datetime import datetime

# Add src to path
sys.path.insert(0, 'src')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)

logger = logging.getLogger("QuickTest")

async def test_system():
    """Quick test of the competitive trading system"""
    try:
        logger.info("🚀 STARTING SYSTEM VALIDATION TEST")
        logger.info("=" * 50)
        
        # Import the fixed system
        from run_real_competitive_trading import RealCompetitiveTradingSystem
        
        # Create system
        logger.info("Creating trading system...")
        system = RealCompetitiveTradingSystem()
        
        # Run one cycle to test
        logger.info("Running test cycle...")
        cycle_result = await system.run_competitive_cycle()
        
        # Analyze results
        total_decisions = cycle_result.get('total_decisions', 0)
        total_trades = len(cycle_result.get('selected_trades', []))
        
        logger.info("=" * 50)
        logger.info("🎯 TEST RESULTS:")
        logger.info(f"   Agents Making Decisions: {total_decisions}/12")
        logger.info(f"   Trades Selected: {total_trades}")
        logger.info(f"   Cycle Duration: {cycle_result.get('cycle_duration', 0):.2f}s")
        
        if total_decisions > 0:
            logger.info("✅ SUCCESS: Agents are now making decisions!")
            logger.info("🎉 System fixes are working!")
            
            # Show agent details
            agent_decisions = cycle_result.get('agent_decisions', {})
            for agent_id, decision in agent_decisions.items():
                if decision:
                    logger.info(f"   {agent_id}: {decision.get('symbol')} {decision.get('action')} {decision.get('quantity'):.1f}")
        else:
            logger.warning("❌ ISSUE: No agents made decisions")
            logger.warning("   Check market data and decision logic")
        
        logger.info("=" * 50)
        logger.info("Test completed successfully!")
        
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        import traceback
        logger.error(traceback.format_exc())

if __name__ == "__main__":
    # Create logs directory
    os.makedirs('logs', exist_ok=True)
    
    # Run test
    asyncio.run(test_system())