#!/usr/bin/env python3
"""
Continuous Real Alpaca Trading
==============================

This script runs paper trading with REAL Alpaca API calls and returns.
"""

import asyncio
import logging
import signal
import sys
import threading
from datetime import datetime
from dotenv import load_dotenv
from src.system_orchestrator import TradingSystemOrchestrator
from config.settings import SystemConfig, AgentConfig, AgentType
import json
import os

# Load environment variables
load_dotenv()

# Configure logging for Windows compatibility
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/real_alpaca_trading.log', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class ContinuousRealAlpacaTrading:
    """Continuous real Alpaca trading system."""
    
    def __init__(self):
        self.orchestrator = None
        self.is_running = False
        self.cycle_count = 0
        self.dashboard_thread = None
        
    async def initialize_system(self):
        """Initialize the trading system."""
        try:
            logger.info("Initializing Continuous REAL Alpaca Trading System")
            
            # Create system configuration
            system_config = SystemConfig(
                trading_symbols=[
                    # Start with just a few symbols for testing
                    "AAPL", "MSFT", "GOOGL", "TSLA", "SPY", "QQQ",
                    # Add crypto for testing
                    "BTCUSD", "ETHUSD",
                ],
                # Use default agent configuration (all 11 agents)
                agent_configs=None  # This will use the default 11 agents from SystemConfig
            )
            
            # Initialize orchestrator
            self.orchestrator = TradingSystemOrchestrator(system_config)
            await self.orchestrator.initialize()
            
            logger.info(f"System initialized with {len(self.orchestrator.agents)} agents")
            logger.info(f"Trading {len(system_config.trading_symbols)} symbols")
            logger.info(f"Total capital: ${sum(a.initial_capital for a in system_config.agent_configs):,.0f}")
            
            # Show real account info
            if self.orchestrator.alpaca_interface:
                account_info = self.orchestrator.alpaca_interface.account_info
                logger.info(f"Real Alpaca Account: {account_info['account_number']}")
                logger.info(f"Real Portfolio Value: ${account_info['portfolio_value']:,.2f}")
                logger.info(f"Real Buying Power: ${account_info['buying_power']:,.2f}")
                logger.info(f"Real Cash: ${account_info['cash']:,.2f}")
            
            # Start monitoring dashboard in background
            self.start_dashboard()
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize system: {e}")
            return False
    
    async def run_continuous_trading(self):
        """Run continuous trading cycles."""
        logger.info("Starting continuous REAL Alpaca trading...")
        self.is_running = True
        
        try:
            while self.is_running:
                self.cycle_count += 1
                cycle_start = datetime.now()
                
                logger.info(f"Starting trading cycle {self.cycle_count}")
                
                # Run one trading cycle
                if not self.orchestrator:
                    logger.error("Orchestrator is not initialized. Stopping trading.")
                    self.is_running = False
                    break
                # Prefer a public 'run_cycle' coroutine if available, otherwise fall back to protected '_run_cycle'
                run_cycle = getattr(self.orchestrator, "run_cycle", None)
                if callable(run_cycle):
                    result = run_cycle()
                    if asyncio.iscoroutine(result):
                        await result
                else:
                    await self.orchestrator._run_cycle()
                
                # Log cycle performance
                cycle_duration = (datetime.now() - cycle_start).total_seconds()
                performance = self.orchestrator.system_performance
                
                # Get real portfolio performance
                real_performance = None
                if self.orchestrator.alpaca_interface:
                    real_performance = await self.orchestrator.alpaca_interface.get_portfolio_performance()
                
                logger.info(f"Cycle {self.cycle_count} completed in {cycle_duration:.2f}s")
                logger.info(f"Total trades: {performance.get('successful_trades', 0)} successful, {performance.get('failed_trades', 0)} failed")
                
                if real_performance:
                    logger.info(f"REAL Portfolio Value: ${real_performance['portfolio_value']:,.2f}")
                    logger.info(f"REAL Total Return: ${real_performance['total_return']:,.2f}")
                    logger.info(f"REAL Return %: {real_performance['return_percentage']:.2f}%")
                    logger.info(f"REAL Positions: {real_performance['positions']}")
                    # Write dashboard performance file (paper)
                    try:
                        os.makedirs('logs', exist_ok=True)
                        with open('logs/paper_performance.json', 'w', encoding='utf-8') as f:
                            json.dump({
                                'total_cycles': self.cycle_count,
                                'successful_trades': performance.get('successful_trades', 0),
                                'failed_trades': performance.get('failed_trades', 0),
                                'total_return': real_performance.get('total_return', 0.0),
                                'portfolio_value': real_performance.get('portfolio_value', 0.0),
                                'timestamp': datetime.now().isoformat()
                            }, f)
                    except Exception as e:
                        logger.warning(f"Could not write paper_performance.json: {e}")
                else:
                    logger.info(f"System return: {performance.get('total_return', 0.0):.4f}")
                
                # Wait before next cycle (60 seconds for real trading)
                logger.info("Waiting 60 seconds before next cycle...")
                await asyncio.sleep(60)
                
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
        logger.info("Stopping continuous real Alpaca trading...")
        self.is_running = False

async def main():
    """Main function."""
    # Set up signal handlers for graceful shutdown
    trading_system = ContinuousRealAlpacaTrading()
    
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
    print("Continuous REAL Alpaca Trading System")
    print("=" * 50)
    print("This system will run REAL paper trading with Alpaca")
    print("and show actual returns and portfolio values.")
    print("=" * 50)
    
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Real Alpaca trading stopped by user")
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        sys.exit(1)
