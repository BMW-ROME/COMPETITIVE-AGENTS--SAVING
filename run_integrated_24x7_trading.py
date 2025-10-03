#!/usr/bin/env python3
"""
Launch 24/7 Crypto + Paper Trading System
Implements "A with every single module already integrated"
Combines crypto trading, paper trading, RL, risk management, and all enhancements
"""

import asyncio
import os
import sys
import json
import time
import signal
from datetime import datetime
from dotenv import load_dotenv
import logging

# Import all integrated modules
sys.path.append('/workspaces/competitive-trading-agents')
from crypto_24x7_trading import Crypto24x7TradingSystem
from continuous_competitive_trading import ContinuousCompetitiveTradingSystem
from rl_100_percent_execution import get_100_percent_execution_optimizer

class IntegratedTradingOrchestrator:
    """
    Master orchestrator for 24/7 trading across all markets and systems
    Integrates: Crypto 24/7 + US Paper Trading + RL + Risk Management + All Modules
    """
    
    def __init__(self):
        load_dotenv()
        
        self.running = False
        self.systems = {}
        
        # Initialize logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('logs/integrated_trading.log'),
                logging.StreamHandler(sys.stdout)
            ]
        )
        self.logger = logging.getLogger('IntegratedTrading')
        
        # Signal handling
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        # System metrics
        self.start_time = datetime.now()
        self.total_systems = 0
        self.active_systems = 0
        
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        self.logger.info(f"📡 Received signal {signum}, shutting down all trading systems...")
        self.running = False
        
        # Stop all systems gracefully
        for system_name, system in self.systems.items():
            if hasattr(system, 'running'):
                system.running = False
    
    async def start_crypto_trading(self):
        """Start 24/7 crypto trading system"""
        try:
            self.logger.info("🚀 Starting 24/7 Crypto Trading System...")
            crypto_system = Crypto24x7TradingSystem()
            self.systems['crypto_24x7'] = crypto_system
            
            # Run crypto trading in background
            crypto_task = asyncio.create_task(crypto_system.start_crypto_trading())
            self.total_systems += 1
            self.active_systems += 1
            
            return crypto_task
            
        except Exception as e:
            self.logger.error(f"❌ Failed to start crypto trading: {e}")
            return None
    
    async def start_paper_trading(self):
        """Start US paper trading system"""
        try:
            self.logger.info("📈 Starting US Paper Trading System...")
            paper_system = ContinuousCompetitiveTradingSystem()
            self.systems['paper_trading'] = paper_system
            
            # Run paper trading in background
            paper_task = asyncio.create_task(paper_system.start_continuous_trading())
            self.total_systems += 1
            self.active_systems += 1
            
            return paper_task
            
        except Exception as e:
            self.logger.error(f"❌ Failed to start paper trading: {e}")
            return None
    
    async def monitor_systems(self):
        """Monitor all trading systems"""
        monitor_count = 0
        
        while self.running:
            monitor_count += 1
            runtime = (datetime.now() - self.start_time).total_seconds() / 3600
            
            # Log system status
            if monitor_count % 10 == 0:  # Every 10 minutes
                active_count = sum(1 for name, system in self.systems.items() 
                                 if hasattr(system, 'running') and system.running)
                
                self.logger.info("🌐 INTEGRATED TRADING SYSTEMS STATUS")
                self.logger.info(f"⏱️ Runtime: {runtime:.1f} hours")
                self.logger.info(f"🔧 Total Systems: {self.total_systems}")
                self.logger.info(f"✅ Active Systems: {active_count}")
                
                for system_name, system in self.systems.items():
                    if hasattr(system, 'running'):
                        status = "🟢 RUNNING" if system.running else "🔴 STOPPED"
                        self.logger.info(f"  {system_name}: {status}")
                    
                    # System-specific stats
                    if hasattr(system, 'stats'):
                        stats = system.stats
                        if hasattr(stats, 'total_trades_executed'):
                            self.logger.info(f"    Trades: {stats.total_trades_executed}")
                        if hasattr(stats, 'total_pnl'):
                            self.logger.info(f"    P&L: ${stats.total_pnl:.2f}")
            
            await asyncio.sleep(60)  # Check every minute
    
    async def run_integrated_systems(self):
        """Run all integrated trading systems"""
        self.running = True
        
        self.logger.info("🌟 LAUNCHING INTEGRATED 24/7 TRADING PLATFORM")
        self.logger.info("=" * 60)
        
        # CRITICAL: Environment Safety Check
        alpaca_url = os.getenv("ALPACA_BASE_URL") or os.getenv("APCA_API_BASE_URL", "NOT_SET")
        trading_mode = os.getenv("TRADING_MODE", "NOT_SET")
        if "paper" in alpaca_url.lower():
            self.logger.info("🟡 TRADING ENVIRONMENT: PAPER (SAFE MODE)")
        elif "live" in alpaca_url.lower():
            self.logger.info("🔴 TRADING ENVIRONMENT: LIVE (REAL MONEY)")
        else:
            self.logger.warning("⚠️ TRADING ENVIRONMENT: UNKNOWN - VERIFY .ENV")
        self.logger.info(f"🔗 Alpaca API: {alpaca_url}")
        self.logger.info(f"📋 Trading Mode: {trading_mode}")
        self.logger.info("=" * 60)
        
        self.logger.info("🪙 Crypto Trading: 24/7 Always Open")
        self.logger.info("📈 Paper Trading: US Market Hours + Extended")
        self.logger.info("🧠 RL Optimization: 100% Execution Targeting")
        self.logger.info("🛡️ Risk Management: Multi-Level Protection")
        self.logger.info("⚙️ All Modules: Fully Integrated")
        self.logger.info("=" * 60)
        
        tasks = []
        
        try:
            # Start crypto trading (24/7)
            crypto_task = await self.start_crypto_trading()
            if crypto_task:
                tasks.append(crypto_task)
            
            # Start paper trading (market hours)
            paper_task = await self.start_paper_trading()
            if paper_task:
                tasks.append(paper_task)
            
            # Start system monitoring
            monitor_task = asyncio.create_task(self.monitor_systems())
            tasks.append(monitor_task)
            
            if not tasks:
                self.logger.error("❌ No systems started successfully")
                return
            
            self.logger.info(f"✅ {len(tasks)} systems launched successfully")
            self.logger.info("💎 Press Ctrl+C to shutdown all systems gracefully")
            
            # Wait for all tasks
            await asyncio.gather(*tasks, return_exceptions=True)
            
        except Exception as e:
            self.logger.error(f"💥 Critical error in integrated systems: {e}")
        finally:
            await self._shutdown_all_systems()
    
    async def _shutdown_all_systems(self):
        """Shutdown all systems gracefully"""
        self.logger.info("🔄 Shutting down all trading systems...")
        
        # Stop all systems
        for system_name, system in self.systems.items():
            if hasattr(system, 'running'):
                system.running = False
                self.logger.info(f"🔴 Stopped {system_name}")
        
        # Generate final report
        runtime = (datetime.now() - self.start_time).total_seconds() / 3600
        
        final_report = {
            'timestamp': datetime.now().isoformat(),
            'runtime_hours': runtime,
            'total_systems': self.total_systems,
            'systems_launched': list(self.systems.keys()),
            'integration_status': 'complete'
        }
        
        # Save final report
        filename = f"data/integrated_session_{datetime.now().strftime('%Y%m%d_%H%M')}.json"
        os.makedirs('data', exist_ok=True)
        with open(filename, 'w') as f:
            json.dump(final_report, f, indent=2)
        
        self.logger.info(f"💾 Final session report saved: {filename}")
        self.logger.info(f"⏱️ Total runtime: {runtime:.1f} hours")
        self.logger.info("✅ All systems shutdown complete")
        self.logger.info("🏁 Integrated trading platform stopped")

async def main():
    """Main function"""
    # Create directories
    os.makedirs('logs', exist_ok=True)
    os.makedirs('data', exist_ok=True)
    
    # Launch integrated trading platform
    orchestrator = IntegratedTradingOrchestrator()
    await orchestrator.run_integrated_systems()

if __name__ == "__main__":
    print("🌟 Integrated 24/7 Trading Platform")
    print("🪙 Crypto + Paper Trading + RL + Risk Management")
    print("🚀 Starting all systems...")
    print()
    
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n🔴 Shutdown requested by user")
    except Exception as e:
        print(f"\n💥 Critical error: {e}")