#!/usr/bin/env python3
"""
Production Paper Trading System - Docker Ready
Fixed environment loading and clean process management
"""

import os
import sys
import asyncio
import logging
from datetime import datetime

def load_env_file(env_file_path):
    """Load environment variables from .env file manually"""
    env_vars = {}
    
    if not os.path.exists(env_file_path):
        return env_vars
    
    try:
        with open(env_file_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    key = key.strip()
                    value = value.strip()
                    
                    # Set in os.environ for child processes
                    os.environ[key] = value
                    env_vars[key] = value
    except Exception as e:
        print(f"Error loading .env file: {e}")
    
    return env_vars

def verify_paper_trading_environment():
    """Verify we're in PAPER trading mode"""
    print("🔒 TRADING ENVIRONMENT VERIFICATION")
    print("=" * 60)
    
    # Load environment
    env_file = "/workspaces/competitive-trading-agents/.env"
    env_vars = load_env_file(env_file)
    
    # Check Alpaca configuration
    alpaca_key = env_vars.get('ALPACA_API_KEY') or env_vars.get('APCA_API_KEY_ID', '')
    alpaca_secret = env_vars.get('ALPACA_SECRET_KEY') or env_vars.get('APCA_API_SECRET_KEY', '')
    alpaca_base_url = env_vars.get('ALPACA_BASE_URL') or env_vars.get('APCA_API_BASE_URL', '')
    trading_mode = env_vars.get('TRADING_MODE', 'UNKNOWN')
    
    print(f"🔗 Base URL: {alpaca_base_url}")
    print(f"📋 Trading Mode: {trading_mode}")
    
    # Verify PAPER environment
    is_paper = 'paper-api.alpaca.markets' in alpaca_base_url.lower()
    
    if is_paper:
        print("🟢 CONFIRMED: PAPER TRADING ENVIRONMENT")
        print("🛡️  STATUS: SAFE - NO REAL MONEY AT RISK")
        if alpaca_key:
            masked_key = alpaca_key[:6] + "..." + alpaca_key[-4:] if len(alpaca_key) > 10 else "***"
            print(f"🔑 API Key: {masked_key}")
        print("=" * 60)
        return True
    else:
        print("🔴 WARNING: NOT CONFIRMED AS PAPER TRADING")
        print("🛑 SAFETY CHECK FAILED")
        print("=" * 60)
        return False

def setup_production_logging():
    """Setup clean production logging"""
    # Create logs directory
    os.makedirs('logs', exist_ok=True)
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('logs/production_trading.log'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    # Set specific log levels for clean operation
    logging.getLogger('alpaca').setLevel(logging.WARNING)
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    logging.getLogger('requests').setLevel(logging.WARNING)

async def run_production_trading():
    """Run the production trading system"""
    try:
        print("\n🚀 LAUNCHING PRODUCTION TRADING SYSTEM")
        print("=" * 60)
        print("📈 Paper Trading: Active")
        print("🪙 Crypto Trading: 24/7")
        print("🧠 RL Optimization: Enabled")
        print("🛡️ Risk Management: Active")
        print("=" * 60)
        
        # Import the working integrated system
        sys.path.append('/workspaces/competitive-trading-agents')
        from run_integrated_24x7_trading import IntegratedTradingOrchestrator
        
        # Create and run orchestrator
        orchestrator = IntegratedTradingOrchestrator()
        
        print("⏰ System started at:", datetime.now().strftime('%H:%M:%S'))
        print("💎 Press Ctrl+C to shutdown gracefully")
        print()
        
        # Run the integrated systems
        await orchestrator.run_integrated_systems()
        
    except KeyboardInterrupt:
        print("\n⏸️  Graceful shutdown requested...")
        return True
    except Exception as e:
        print(f"\n❌ System error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main production launcher"""
    print("🏁 PRODUCTION PAPER TRADING SYSTEM")
    print(f"📅 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Verify environment safety
    if not verify_paper_trading_environment():
        print("🛑 Environment verification failed - exiting for safety")
        sys.exit(1)
    
    # Setup logging
    setup_production_logging()
    
    # Create necessary directories
    os.makedirs('data', exist_ok=True)
    
    try:
        # Run the trading system
        asyncio.run(run_production_trading())
        print("✅ System shutdown completed successfully")
        
    except Exception as e:
        print(f"❌ Critical error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()