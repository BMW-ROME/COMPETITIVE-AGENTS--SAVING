#!/usr/bin/env python3
"""
Continuous Live Trading - Windows Compatible Version
===================================================

This script runs live trading continuously until stopped.
Designed for Windows compatibility (no emojis).
"""

import asyncio
import logging
import signal
import sys
import threading
import os
import inspect
from datetime import datetime
from dotenv import load_dotenv
from logging.handlers import RotatingFileHandler
from src.system_orchestrator import TradingSystemOrchestrator
from config.settings import SystemConfig, AgentConfig, AgentType

# Load environment variables
load_dotenv()

# Ensure logs directory exists
os.makedirs('logs', exist_ok=True)

# Configure logging for Windows compatibility with a rotating file handler
file_handler = RotatingFileHandler('logs/live_trading.log', maxBytes=10 * 1024 * 1024, backupCount=5, encoding='utf-8')
file_handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)

stream_handler = logging.StreamHandler(sys.stdout)
stream_handler.setFormatter(formatter)

logging.basicConfig(
    level=logging.INFO,
    handlers=[file_handler, stream_handler]
)
logger = logging.getLogger(__name__)

class ContinuousLiveTrading:
    """Continuous live trading system."""
    
    def __init__(self):
        self.orchestrator = None
        self.is_running = False
        self.cycle_count = 0
        self.dashboard_thread = None
        
    async def initialize_system(self):
        """Initialize the trading system."""
        try:
            logger.info("Initializing Continuous LIVE Trading System")
            logger.warning("WARNING: This is LIVE TRADING with REAL MONEY!")
            
            # Verify we're in live mode
            from config.settings import AlpacaConfig
            alpaca_config = AlpacaConfig()
            
            if 'paper-api' in alpaca_config.base_url:
                logger.error("ERROR: System is configured for paper trading!")
                logger.error("Please update config/settings.py to use live trading URL")
                return False
            
            logger.info("Confirmed: Live trading mode enabled")
            
            # Create system configuration with conservative settings for live trading
            system_config = SystemConfig(
                trading_symbols=[
                    # Stocks - Conservative selection for live trading
                    "AAPL", "GOOGL", "MSFT", "TSLA", "AMZN", "NVDA", "META", "NFLX",
                    "JPM", "BAC", "WFC", "GS", "SPY", "QQQ", "IWM", "VTI",
                    
                    # FOREX - Major pairs only for live trading
                    "EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCAD",
                    
                    # Crypto - Major coins only for live trading
                    "BTCUSD", "ETHUSD", "ADAUSD", "SOLUSD", "DOTUSD",
                ],
                agent_configs=[
                    # Conservative stock agent (largest allocation)
                    AgentConfig("live_stock_conservative", AgentType.CONSERVATIVE, 
                               initial_capital=20000.0, risk_tolerance=0.015, max_position_size=0.03),
                    
                    # Balanced stock agent
                    AgentConfig("live_stock_balanced", AgentType.BALANCED,
                               initial_capital=15000.0, risk_tolerance=0.02, max_position_size=0.04),
                    
                    # Quantitative pattern agent
                    AgentConfig("live_stock_quantitative", AgentType.QUANTITATIVE_PATTERN,
                               initial_capital=10000.0, risk_tolerance=0.025, max_position_size=0.035),
                ]
            )
            
            # Initialize orchestrator
            self.orchestrator = TradingSystemOrchestrator(system_config)
            await self.orchestrator.initialize()
            
            total_capital = sum(a.initial_capital for a in system_config.agent_configs)
            
            logger.info(f"Live system initialized with {len(self.orchestrator.agents)} agents")
            logger.info(f"Trading {len(system_config.trading_symbols)} symbols")
            logger.info(f"Total capital: ${total_capital:,.0f}")
            logger.warning(f"RISK WARNING: ${total_capital:,.0f} at risk in live trading!")
            
            # Start monitoring dashboard in background
            self.start_dashboard()
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize live system: {e}")
            return False
    
    async def run_continuous_trading(self):
        """Run continuous trading cycles."""
        logger.info("Starting continuous LIVE trading...")
        logger.warning("WARNING: Real money is at risk!")
        self.is_running = True

        # Ensure the orchestrator is initialized and provides a runnable cycle coroutine
        if not self.orchestrator:
            logger.error("Orchestrator is not initialized. Aborting continuous trading.")
            self.is_running = False
            return

        # Prefer a public run_cycle method, fallback to _run_cycle if present
        cycle_runner = getattr(self.orchestrator, "run_cycle", None) or getattr(self.orchestrator, "_run_cycle", None)
        if not callable(cycle_runner):
            logger.error("Orchestrator does not provide a run_cycle or _run_cycle coroutine method. Aborting.")
            self.is_running = False
            return
        
        try:
            while self.is_running:
                self.cycle_count += 1
                cycle_start = datetime.now()
                
                logger.info(f"Starting LIVE trading cycle {self.cycle_count}")
                # Run one trading cycle using the available runner (supports coroutine functions,
                # functions that return awaitables, and synchronous functions).
                if asyncio.iscoroutinefunction(cycle_runner):
                    # runner is an async def -> await it directly
                    await cycle_runner()
                else:
                    # runner is synchronous (or may return an awaitable) -> run in executor
                    loop = asyncio.get_running_loop()
                    result = await loop.run_in_executor(None, cycle_runner)
                    # if the sync runner returned an awaitable, await it now
                    if inspect.isawaitable(result):
                        await result

                # Log cycle performance
                cycle_duration = (datetime.now() - cycle_start).total_seconds()
                performance = self.orchestrator.system_performance
                performance = self.orchestrator.system_performance
                
                logger.info(f"LIVE Cycle {self.cycle_count} completed in {cycle_duration:.2f}s")
                logger.info(f"Total trades: {performance.get('successful_trades', 0)} successful, {performance.get('failed_trades', 0)} failed")
                logger.info(f"Total return: {performance.get('total_return', 0.0):.4f}")
                
                # More frequent cycles for live trading (15 seconds)
                logger.info("Waiting 15 seconds before next LIVE cycle...")
                await asyncio.sleep(15)
                
        except Exception as e:
            logger.error(f"Error in continuous LIVE trading: {e}")
        finally:
            await self.cleanup()
    
    async def cleanup(self):
        """Clean up resources."""
        if self.orchestrator:
            await self.orchestrator.shutdown()
            logger.info("Live system cleanup completed")
    
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
        logger.info("Stopping continuous LIVE trading...")
        self.is_running = False

async def main():
    """Main function."""
    # Set up signal handlers for graceful shutdown
    trading_system = ContinuousLiveTrading()
    
    def signal_handler(signum, frame):
        logger.info(f"Received signal {signum}")
        trading_system.stop()
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Initialize and run
    if await trading_system.initialize_system():
        await trading_system.run_continuous_trading()
    else:
        logger.error("Failed to initialize live system")
        sys.exit(1)

if __name__ == "__main__":
    print("Continuous LIVE Trading System")
    print("=" * 50)
    print("WARNING: This is LIVE TRADING with REAL MONEY!")
    print("This system will run live trading continuously")
    print("until stopped with Ctrl+C or Docker stop command.")
    print("=" * 50)
    
    # Final confirmation
    response = input("Are you sure you want to start LIVE trading? (type 'YES' to confirm): ")
    if response != "YES":
        print("Live trading cancelled")
        sys.exit(0)
    
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Live trading stopped by user")
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        sys.exit(1)
