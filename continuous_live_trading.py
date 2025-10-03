#!/usr/bin/env python3
"""
Continuous Live Trading
=======================

This script runs live trading continuously until stopped.
Designed to run in a Docker container with live API keys.
"""

import asyncio
import logging
import signal
import sys
from datetime import datetime
from dotenv import load_dotenv
from rl_100_percent_execution import get_100_percent_execution_optimizer

from src.system_orchestrator import TradingSystemOrchestrator
from config.settings import SystemConfig, AgentConfig, AgentType

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/live_trading.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class ContinuousLiveTrading:
    """Continuous live trading system."""
    
    def __init__(self):
        self.orchestrator = None
        self.is_running = False
        self.cycle_count = 0
        
    async def initialize_system(self):
        """Initialize the trading system."""
        try:
            logger.info("🚨 Initializing Continuous LIVE Trading System")
            logger.warning("⚠️  WARNING: This is LIVE TRADING with REAL MONEY!")
            
            # Verify we're in live mode
            from config.settings import AlpacaConfig
            alpaca_config = AlpacaConfig()
            
            if 'paper-api' in alpaca_config.base_url:
                logger.error("❌ ERROR: System is configured for paper trading!")
                logger.error("Please update config/settings.py to use live trading URL")
                return False
            
            logger.info("✅ Confirmed: Live trading mode enabled")
            
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
                    
                    # FOREX specialist (conservative)
                    AgentConfig("live_forex_major", AgentType.FOREX_SPECIALIST,
                               initial_capital=10000.0, risk_tolerance=0.015, max_position_size=0.05),
                    
                    # Crypto specialist (very conservative)
                    AgentConfig("live_crypto_major", AgentType.CRYPTO_SPECIALIST,
                               initial_capital=8000.0, risk_tolerance=0.03, max_position_size=0.04),
                    
                    # Multi-Asset Arbitrage (conservative)
                    AgentConfig("live_arbitrage", AgentType.MULTI_ASSET_ARBITRAGE,
                               initial_capital=12000.0, risk_tolerance=0.02, max_position_size=0.06),
                ]
            )
            
            # Initialize orchestrator
            self.orchestrator = TradingSystemOrchestrator(system_config)
            await self.orchestrator.initialize()
            
            total_capital = sum(a.initial_capital for a in system_config.agent_configs)
            
            logger.info(f"✅ Live system initialized with {len(self.orchestrator.agents)} agents")
            logger.info(f"📊 Trading {len(system_config.trading_symbols)} symbols")
            logger.info(f"💰 Total capital: ${total_capital:,.0f}")
            logger.warning(f"⚠️  RISK WARNING: ${total_capital:,.0f} at risk in live trading!")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize live system: {e}")
            return False
    
    async def run_continuous_trading(self):
        """Run continuous trading cycles."""
        logger.info("🔄 Starting continuous LIVE trading...")
        logger.warning("⚠️  WARNING: Real money is at risk!")
        self.is_running = True
        
        # Ensure orchestrator is initialized before starting cycles
        if self.orchestrator is None:
            logger.error("❌ Orchestrator is not initialized; aborting continuous trading")
            await self.cleanup()
            return
        
        try:
            while self.is_running:
                self.cycle_count += 1
                cycle_start = datetime.now()
                
                logger.info(f"📈 Starting LIVE trading cycle {self.cycle_count}")
                
                # Run one trading cycle (use public API if available; fall back to private)
                run_cycle_coro = None
                if hasattr(self.orchestrator, "run_cycle"):
                    run_cycle_coro = getattr(self.orchestrator, "run_cycle")
                elif hasattr(self.orchestrator, "_run_cycle"):
                    run_cycle_coro = getattr(self.orchestrator, "_run_cycle")
                else:
                    logger.error("❌ Orchestrator does not expose a run cycle coroutine; aborting")
                    self.is_running = False
                    break
                
                await run_cycle_coro()
                
                # Log cycle performance
                cycle_duration = (datetime.now() - cycle_start).total_seconds()
                performance = getattr(self.orchestrator, "system_performance", {}) or {}
                
                logger.info(f"✅ LIVE Cycle {self.cycle_count} completed in {cycle_duration:.2f}s")
                logger.info(f"📊 Total trades: {performance.get('successful_trades', 0)} successful, {performance.get('failed_trades', 0)} failed")
                logger.info(f"💰 Total return: {performance.get('total_return', 0.0):.4f}")
                
                # More frequent cycles for live trading (15 seconds)
                logger.info("⏳ Waiting 15 seconds before next LIVE cycle...")
                await asyncio.sleep(15)
                
        except Exception as e:
            logger.error(f"❌ Error in continuous LIVE trading: {e}")
        finally:
            await self.cleanup()
    
    async def cleanup(self):
        """Clean up resources."""
        if self.orchestrator:
            await self.orchestrator.shutdown()
            logger.info("🧹 Live system cleanup completed")
    
    def stop(self):
        """Stop the trading system."""
        logger.info("🛑 Stopping continuous LIVE trading...")
        self.is_running = False

async def main():
    """Main function."""
    # Set up signal handlers for graceful shutdown
    trading_system = ContinuousLiveTrading()
    
    def signal_handler(signum, frame):
        logger.info(f"📡 Received signal {signum}")
        trading_system.stop()
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Initialize and run
    if await trading_system.initialize_system():
        await trading_system.run_continuous_trading()
    else:
        logger.error("❌ Failed to initialize live system")
        sys.exit(1)

if __name__ == "__main__":
    print("🚨 Continuous LIVE Trading System")
    print("=" * 50)
    print("⚠️  WARNING: This is LIVE TRADING with REAL MONEY!")
    print("This system will run live trading continuously")
    print("until stopped with Ctrl+C or Docker stop command.")
    print("=" * 50)
    
    # Final confirmation
    response = input("🚨 Are you sure you want to start LIVE trading? (type 'YES' to confirm): ")
    if response != "YES":
        print("👋 Live trading cancelled")
        sys.exit(0)
    
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("👋 Live trading stopped by user")
    except Exception as e:
        logger.error(f"❌ Fatal error: {e}")
        sys.exit(1)
