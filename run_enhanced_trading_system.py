#!/usr/bin/env python3
"""
Enhanced Trading System - Competitive Agents
============================================
Implements competitive trading with:
- All agents making decisions every cycle
- Continuous learning and reflections
- Quick trades (scalping)
- Hierarchy-based agent selection
- Performance-based agent scaling
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
    handlers=[
        logging.FileHandler('logs/enhanced_trading_system.log', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger("EnhancedTradingSystem")

async def main():
    try:
        logger.info("=" * 60)
        logger.info("ENHANCED COMPETITIVE TRADING SYSTEM STARTING")
        logger.info("=" * 60)
        
        # Import enhanced system components
        from enhanced_system_orchestrator import EnhancedSystemOrchestrator
        from config.settings import SystemConfig
        
        # Create enhanced system configuration
        system_config = SystemConfig(
            trading_symbols=["AAPL", "MSFT", "GOOGL", "TSLA", "SPY", "QQQ"],
            agent_configs=None  # Use enhanced agents
        )
        
        # Initialize enhanced orchestrator
        orchestrator = EnhancedSystemOrchestrator(system_config)
        
        # Initialize system
        logger.info("Initializing enhanced competitive trading system...")
        success = await orchestrator.initialize()
        
        if not success:
            logger.error("Failed to initialize enhanced trading system")
            return
        
        logger.info("Enhanced trading system initialized successfully!")
        logger.info("Starting competitive trading operations...")
        logger.info("All agents will now make decisions and reflections every cycle!")
        
        # Run enhanced system
        await orchestrator.run_system()
        
    except KeyboardInterrupt:
        logger.info("Enhanced system stopped by user")
    except Exception as e:
        logger.error(f"Fatal error in enhanced system: {e}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)

if __name__ == "__main__":
    # Create logs directory
    os.makedirs('logs', exist_ok=True)
    
    # Run the enhanced system
    asyncio.run(main())

