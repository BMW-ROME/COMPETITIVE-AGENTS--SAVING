#!/usr/bin/env python3
"""
Production 24/7 Crypto + Paper Trading System
Clean, efficient, minimal logging for production deployment
"""

import sys
import os
import asyncio
import logging
from datetime import datetime

# Add path
sys.path.append('/workspaces/competitive-trading-agents')

def setup_production_logging():
    """Setup clean production logging"""
    # Set up clean logging - only important messages
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('logs/production_trading.log'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    # Reduce noise from specific modules
    logging.getLogger('RL_100_Percent').setLevel(logging.INFO)
    logging.getLogger('RL_Optimizer').setLevel(logging.WARNING)
    logging.getLogger('ContinuousTrading').setLevel(logging.INFO)
    logging.getLogger('Crypto24x7Trading').setLevel(logging.INFO)

def print_startup_banner():
    """Print clean startup information"""
    print("🌟 Production 24/7 Trading System")
    print("=" * 45)
    print("🪙 Crypto Trading: 24/7 Markets")
    print("📈 Paper Trading: Market Hours")
    print("🧠 RL Optimization: Active")
    print("🛡️ Risk Management: Enabled")
    print("🔇 Clean Mode: Production")
    print("=" * 45)
    print(f"🕐 Launch: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("💎 Press Ctrl+C to shutdown gracefully")
    print()

async def main():
    """Main production launcher"""
    setup_production_logging()
    print_startup_banner()
    
    # Import and run the integrated system
    from run_integrated_24x7_trading import IntegratedTradingOrchestrator
    
    orchestrator = IntegratedTradingOrchestrator()
    await orchestrator.run_integrated_systems()

if __name__ == "__main__":
    try:
        # Create necessary directories
        os.makedirs('logs', exist_ok=True)
        os.makedirs('data', exist_ok=True)
        
        # Run the production system
        asyncio.run(main())
        
    except KeyboardInterrupt:
        print("\n🔴 Production system shutdown requested")
        print("✅ All systems stopped gracefully")
    except Exception as e:
        print(f"\n💥 Production error: {e}")
        print("📋 Check logs/production_trading.log for details")