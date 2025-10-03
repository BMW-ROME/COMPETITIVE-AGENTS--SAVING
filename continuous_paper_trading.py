#!/usr/bin/env python3
"""
Continuous Paper Trading
========================

This script runs paper trading continuously until stopped.
Designed to run in a Docker container.
"""

import asyncio
import logging
import os
import signal
import sys
import threading
from datetime import datetime
from logging.handlers import RotatingFileHandler
from dotenv import load_dotenv
from rl_100_percent_execution import get_100_percent_execution_optimizer

from src.system_orchestrator import TradingSystemOrchestrator
from config.settings import SystemConfig, AgentConfig, AgentType
from typing import Optional

# Load environment variables
load_dotenv()

# Configure logging (env-configurable directory with rotation)
log_dir = os.getenv('LOG_DIR', 'logs')
try:
    os.makedirs(log_dir, exist_ok=True)
except Exception:
    pass

file_handler = RotatingFileHandler(
    os.path.join(log_dir, 'paper_trading.log'),
    maxBytes=int(os.getenv('LOG_MAX_BYTES', '10000000')),
    backupCount=int(os.getenv('LOG_BACKUP_COUNT', '5')),
    encoding='utf-8'
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        file_handler,
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)
class ContinuousPaperTrading:
    """Continuous paper trading system."""
    
    def __init__(self):
        self.orchestrator: Optional[TradingSystemOrchestrator] = None
        self.is_running = False
        self.cycle_count = 0
        self.dashboard_thread: Optional[threading.Thread] = None
        self.dashboard_thread = None
        
    async def initialize_system(self):
        """Initialize the trading system."""
        try:
            logger.info("Initializing Continuous Paper Trading System")
            
            # Create system configuration
            system_config = SystemConfig(
                trading_symbols=[
                    # Stocks - Major tech and blue-chip
                    "AAPL", "GOOGL", "MSFT", "TSLA", "AMZN", "NVDA", "META", "NFLX", "ADBE", "CRM",
                    "JPM", "BAC", "WFC", "GS", "MS", "C", "AXP", "BLK", "SPGI", "V",
                    
                    # FOREX - Major pairs
                    "EURUSD", "GBPUSD", "USDJPY", "USDCHF", "AUDUSD", "USDCAD", "NZDUSD",
                    "EURGBP", "EURJPY", "GBPJPY", "CHFJPY", "AUDJPY", "CADJPY",
                    
                    # Crypto - Major coins and DeFi
                    "BTCUSD", "ETHUSD", "ADAUSD", "SOLUSD", "DOTUSD", "MATICUSD", "AVAXUSD",
                    "UNIUSD", "AAVEUSD", "SUSHIUSD", "CRVUSD", "COMPUSD", "MKRUSD",
                ],
                agent_configs=[
                    # Stock Agents
                    AgentConfig("paper_stock_conservative", AgentType.CONSERVATIVE, 
                               initial_capital=15000.0, risk_tolerance=0.02, max_position_size=0.05),
                    AgentConfig("paper_stock_balanced", AgentType.BALANCED,
                               initial_capital=12000.0, risk_tolerance=0.03, max_position_size=0.08),
                    AgentConfig("paper_stock_quantitative", AgentType.QUANTITATIVE_PATTERN,
                               initial_capital=10000.0, risk_tolerance=0.04, max_position_size=0.06),
                    
                    # FOREX Agents
                    AgentConfig("paper_forex_major", AgentType.FOREX_SPECIALIST,
                               initial_capital=10000.0, risk_tolerance=0.02, max_position_size=0.08),
                    AgentConfig("paper_forex_minor", AgentType.FOREX_SPECIALIST,
                               initial_capital=8000.0, risk_tolerance=0.03, max_position_size=0.06),
                    
                    # Crypto Agents
                    AgentConfig("paper_crypto_major", AgentType.CRYPTO_SPECIALIST,
                               initial_capital=12000.0, risk_tolerance=0.05, max_position_size=0.08),
                    AgentConfig("paper_crypto_defi", AgentType.CRYPTO_SPECIALIST,
                               initial_capital=6000.0, risk_tolerance=0.08, max_position_size=0.05),
                    
                    # Multi-Asset Arbitrage
                    AgentConfig("paper_arbitrage", AgentType.MULTI_ASSET_ARBITRAGE,
                               initial_capital=15000.0, risk_tolerance=0.03, max_position_size=0.10),
                ]
            )
            
            # Initialize orchestrator
            self.orchestrator = TradingSystemOrchestrator(system_config)
            await self.orchestrator.initialize()
            
            logger.info(f"System initialized with {len(self.orchestrator.agents)} agents")
            logger.info(f"Trading {len(system_config.trading_symbols)} symbols")
            logger.info(f"Total capital: ${sum(a.initial_capital for a in system_config.agent_configs):,.0f}")
            
            # Start monitoring dashboard in background
            self.start_dashboard()
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize system: {e}")
            return False
    
    async def run_continuous_trading(self):
        """Run continuous trading cycles."""
        logger.info("Starting continuous paper trading...")
        self.is_running = True
        
        try:
            while self.is_running:
                self.cycle_count += 1
                cycle_start = datetime.now()
                logger.info(f"Starting trading cycle {self.cycle_count}")
                
                # Run one trading cycle
                if self.orchestrator is None:
                    logger.error("Orchestrator is not initialized; stopping continuous trading")
                    self.is_running = False
                    break
                await self.orchestrator._run_cycle()
                
                # Log cycle performance
                cycle_duration = (datetime.now() - cycle_start).total_seconds()
                performance = self.orchestrator.system_performance
                performance = self.orchestrator.system_performance
                
                logger.info(f"Cycle {self.cycle_count} completed in {cycle_duration:.2f}s")
                logger.info(f"Total trades: {performance.get('successful_trades', 0)} successful, {performance.get('failed_trades', 0)} failed")
                logger.info(f"Total return: {performance.get('total_return', 0.0):.4f}")
                
                # Wait before next cycle (30 seconds)
                logger.info("Waiting 30 seconds before next cycle...")
                await asyncio.sleep(30)
                
        except Exception as e:
            logger.error(f"Error in continuous trading: {e}")
        finally:
            await self.cleanup()
    
    async def cleanup(self):
        """Clean up resources."""
        if self.orchestrator:
            await self.orchestrator.shutdown()
            logger.info("System cleanup completed")
    
    def start_dashboard(self):
        """Start the monitoring dashboard in a background thread."""
        try:
            from monitoring_dashboard import app
            self.dashboard_thread = threading.Thread(
                target=lambda: app.run(host='0.0.0.0', port=8000, debug=False, use_reloader=False),
                daemon=True
            )
            self.dashboard_thread.start()
            logger.info("Monitoring dashboard started at http://localhost:8000")
        except Exception as e:
            logger.warning(f"Could not start dashboard: {e}")
    
    def stop(self):
        """Stop the trading system."""
        logger.info("Stopping continuous paper trading...")
        self.is_running = False

async def main():
    """Main function."""
    # Set up signal handlers for graceful shutdown
    trading_system = ContinuousPaperTrading()
    
    def signal_handler(signum, frame):
        logger.info(f"Received signal {signum}")
        trading_system.stop()
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Initialize and run
    if await trading_system.initialize_system():
        await trading_system.run_continuous_trading()
    else:
        logger.error("Failed to initialize system")
        sys.exit(1)

if __name__ == "__main__":
    print("📄 Continuous Paper Trading System")
    print("=" * 50)
    print("This system will run paper trading continuously")
    print("until stopped with Ctrl+C or Docker stop command.")
    print("=" * 50)
    
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Paper trading stopped by user")
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        sys.exit(1)
