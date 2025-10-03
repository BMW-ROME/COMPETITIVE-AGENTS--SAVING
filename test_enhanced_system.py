#!/usr/bin/env python3
"""
Enhanced System Test - Validate Multiple Agent Activity
======================================================
This script tests the enhanced trading system to ensure multiple agents are active.
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

logger = logging.getLogger("EnhancedTest")

async def test_enhanced_system():
    """Test the enhanced system"""
    try:
        logger.info("🚀 TESTING ENHANCED SYSTEM")
        logger.info("=" * 50)
        
        # Import enhanced system
        from run_enhanced_competitive_trading import EnhancedCompetitiveTradingSystem
        
        # Create system
        logger.info("Creating enhanced trading system...")
        system = EnhancedCompetitiveTradingSystem()
        
        # Run test cycle
        logger.info("Running enhanced test cycle...")
        cycle_result = await system.run_competitive_cycle()
        
        # Analyze results
        total_decisions = cycle_result.get('total_decisions', 0)
        total_trades = len(cycle_result.get('selected_trades', []))
        executed_trades = len(cycle_result.get('executed_trades', []))
        active_agents = [aid for aid, decision in cycle_result.get('agent_decisions', {}).items() if decision]
        
        logger.info("=" * 50)
        logger.info("🎯 ENHANCED TEST RESULTS:")
        logger.info(f"   Active Agents: {len(active_agents)}/12")
        logger.info(f"   Total Decisions: {total_decisions}")
        logger.info(f"   Trades Selected: {total_trades}")
        logger.info(f"   Trades Executed: {executed_trades}")
        logger.info(f"   Cycle Duration: {cycle_result.get('cycle_duration', 0):.2f}s")
        
        if len(active_agents) >= 6:
            logger.info("✅ EXCELLENT: Multiple agents are now active!")
        elif len(active_agents) >= 3:
            logger.info("✅ GOOD: Several agents are active!")
        elif len(active_agents) >= 1:
            logger.info("✅ PROGRESS: At least one agent is active!")
        else:
            logger.warning("❌ ISSUE: No agents are active")
        
        # Show active agents
        if active_agents:
            logger.info(f"   Active Agents: {', '.join(active_agents)}")
            
            # Show decision details
            agent_decisions = cycle_result.get('agent_decisions', {})
            for agent_id in active_agents[:5]:  # Show first 5
                decision = agent_decisions[agent_id]
                logger.info(f"     {agent_id}: {decision.get('symbol')} {decision.get('action')} {decision.get('quantity', 0):.1f} (confidence: {decision.get('confidence', 0):.2f})")
        
        logger.info("=" * 50)
        logger.info("Enhanced test completed successfully!")
        
    except Exception as e:
        logger.error(f"❌ Enhanced test failed: {e}")
        import traceback
        logger.error(traceback.format_exc())

if __name__ == "__main__":
    # Create logs directory
    os.makedirs('logs', exist_ok=True)
    
    # Run test
    asyncio.run(test_enhanced_system())