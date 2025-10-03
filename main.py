#!/usr/bin/env python3
import asyncio
import logging
import sys
import os
from datetime import datetime

# Add src to path
sys.path.append('src')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/trading_system.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger("TradingSystem")

async def main():
    try:
        logger.info("Starting Ultimate Trading System...")
        
        # Import and initialize system
        from src.system_orchestrator import TradingSystemOrchestrator
        from config.settings import SystemConfig
        
        # Create system configuration
        system_config = SystemConfig(
            trading_symbols=["AAPL", "MSFT", "GOOGL", "TSLA", "SPY", "QQQ"],
            agent_configs=None  # Use default agents
        )
        
        # Initialize orchestrator
        orchestrator = TradingSystemOrchestrator(system_config)
        
        # Initialize system
        success = await orchestrator.initialize()
        if not success:
            logger.error("Failed to initialize trading system")
            return
        
        logger.info("Trading system initialized successfully")
        
        # Run system
        await orchestrator.run_system()
        
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())
