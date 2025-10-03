#!/usr/bin/env python3
"""
Ultimate Trading System - Simplified Version (No MCP)
====================================================
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
        logging.FileHandler('logs/trading_system.log', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger("UltimateTradingSystem")

async def main():
    try:
        logger.info("=" * 60)
        logger.info("ULTIMATE TRADING SYSTEM STARTING (SIMPLIFIED)")
        logger.info("=" * 60)
        
        # Import system components
        from system_orchestrator import TradingSystemOrchestrator
        from config.settings import SystemConfig
        
        # Create system configuration
        system_config = SystemConfig(
            trading_symbols=["AAPL", "MSFT", "GOOGL", "TSLA", "SPY", "QQQ"],
            agent_configs=None  # Use default agents
        )
        
        # Initialize orchestrator
        orchestrator = TradingSystemOrchestrator(system_config)
        
        # Initialize system
        logger.info("Initializing trading system...")
        success = await orchestrator.initialize()
        
        if not success:
            logger.error("Failed to initialize trading system")
            return
        
        logger.info("Trading system initialized successfully!")
        logger.info("Starting trading operations...")
        
        # Run system
        await orchestrator.run_system()
        
    except KeyboardInterrupt:
        logger.info("System stopped by user")
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)

if __name__ == "__main__":
    # Create logs directory
    os.makedirs('logs', exist_ok=True)
    
    # Run the system
    asyncio.run(main())

