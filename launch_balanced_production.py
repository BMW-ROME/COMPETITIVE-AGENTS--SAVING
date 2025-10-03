#!/usr/bin/env python3
"""
Balanced Production System - Clean with Smart Execution
Allows trades while maintaining production-level cleanliness
"""

import sys
import os
import asyncio
import logging
from datetime import datetime
from dotenv import load_dotenv

# Add path
sys.path.append('/workspaces/competitive-trading-agents')

def check_environment_safety():
    """Check and display environment configuration for safety"""
    print("=" * 80)
    print("🔒 TRADING ENVIRONMENT SAFETY CHECK")
    print("=" * 80)
    
    # Load environment
    env_file = "/workspaces/competitive-trading-agents/.env"
    load_dotenv(env_file)
    
    # Check environment variables
    alpaca_key = os.getenv('ALPACA_API_KEY', '') or os.getenv('APCA_API_KEY_ID', '')
    alpaca_secret = os.getenv('ALPACA_SECRET_KEY', '') or os.getenv('APCA_API_SECRET_KEY', '')
    alpaca_base_url = os.getenv('ALPACA_BASE_URL', '') or os.getenv('APCA_API_BASE_URL', '')
    
    print(f"📊 ALPACA_BASE_URL: {alpaca_base_url}")
    
    # Determine environment type
    if 'paper-api.alpaca.markets' in alpaca_base_url:
        env_type = "PAPER TRADING"
        env_color = "🟢"
        safety_status = "SAFE"
    elif 'api.alpaca.markets' in alpaca_base_url:
        env_type = "LIVE TRADING"
        env_color = "🔴"
        safety_status = "LIVE MONEY AT RISK"
    else:
        env_type = "UNKNOWN"
        env_color = "🟡"
        safety_status = "VERIFY CONFIGURATION"
    
    print(f"{env_color} ENVIRONMENT TYPE: {env_type}")
    print(f"🛡️  SAFETY STATUS: {safety_status}")
    
    if alpaca_key:
        masked_key = alpaca_key[:6] + "..." + alpaca_key[-4:] if len(alpaca_key) > 10 else "***"
        print(f"🔑 API Key: {masked_key}")
    
    if alpaca_secret:
        print(f"🔐 Secret Key: {'*' * 10} (configured)")
    
    print("=" * 80)
    print(f"✅ Environment verified: {env_type}")
    return env_type == "PAPER TRADING"

def setup_balanced_logging():
    """Setup balanced logging - clean but informative"""
    
    # Set up clean logging with key information
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('logs/balanced_production.log'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    # Reduce noise but keep essential info
    logging.getLogger('RL_100_Percent').setLevel(logging.WARNING)
    logging.getLogger('RL_Optimizer').setLevel(logging.WARNING)  
    logging.getLogger('PreRiskFilter').setLevel(logging.INFO)  # Show filter activity
    logging.getLogger('ContinuousTrading').setLevel(logging.INFO)
    logging.getLogger('Crypto24x7Trading').setLevel(logging.INFO)
    logging.getLogger('IntegratedTrading').setLevel(logging.INFO)

def print_balanced_startup():
    """Print balanced startup information"""
    print("⚡ Balanced Production System")
    print("=" * 40)
    print("✅ Systems: Active")
    print("🎯 Trading: Enabled") 
    print("🛡️ Risk Filter: Balanced")
    print("📊 Monitoring: Enhanced")
    
    # CRITICAL: Show trading environment for safety
    alpaca_url = os.getenv("ALPACA_BASE_URL", "NOT_SET")
    trading_mode = os.getenv("TRADING_MODE", "NOT_SET")
    if "paper" in alpaca_url.lower():
        print("🟡 ENVIRONMENT: PAPER TRADING (SAFE)")
    elif "live" in alpaca_url.lower():
        print("🔴 ENVIRONMENT: LIVE TRADING (REAL MONEY)")
    else:
        print("⚠️ ENVIRONMENT: UNKNOWN - CHECK .ENV")
    print(f"🔗 API: {alpaca_url}")
    print(f"📋 Mode: {trading_mode}")
    
    print("=" * 40)
    print(f"🕐 {datetime.now().strftime('%H:%M:%S')} - System Active")
    print("💎 Ctrl+C to shutdown")
    print()

async def main():
    """Main balanced production launcher"""
    setup_balanced_logging()
    print_balanced_startup()
    
    try:
        # Import and run the integrated system
        from run_integrated_24x7_trading import IntegratedTradingOrchestrator
        
        orchestrator = IntegratedTradingOrchestrator()
        await orchestrator.run_integrated_systems()
        
    except Exception as e:
        print(f"System Error: {e}")
        print("Check logs/balanced_production.log for details")

if __name__ == "__main__":
    try:
        # Create necessary directories
        os.makedirs('logs', exist_ok=True)
        os.makedirs('data', exist_ok=True)
        
        print("🚀 Starting Balanced Production System...")
        
        # CRITICAL: Check environment safety first
        if not check_environment_safety():
            print("⚠️ WARNING: Not confirmed as PAPER trading")
            print("🛑 Exiting for safety")
            sys.exit(1)
        
        print("🔄 Initializing trading engines...")
        print()
        
        # Run the balanced system
        asyncio.run(main())
        
    except KeyboardInterrupt:
        print("\n🔴 Shutdown requested - stopping all systems")
        print("✅ Balanced shutdown complete")
    except Exception as e:
        print(f"\nCritical Error: {e}")
        print("Details in logs/balanced_production.log")