#!/usr/bin/env python3
"""
Enhanced Production Trading System with Smart Buying Power Management
Detects current buying power and manages funds intelligently
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

def check_account_buying_power():
    """Check current account buying power before starting"""
    try:
        print("💰 CHECKING CURRENT BUYING POWER")
        print("=" * 60)
        
        # Import alpaca for quick check
        sys.path.append('/workspaces/competitive-trading-agents')
        import alpaca_trade_api as tradeapi
        
        # Create API connection
        api = tradeapi.REST(
            key_id=os.environ.get('APCA_API_KEY_ID'),
            secret_key=os.environ.get('APCA_API_SECRET_KEY'),
            base_url=os.environ.get('APCA_API_BASE_URL', 'https://paper-api.alpaca.markets'),
            api_version='v2'
        )
        
        # Get account info
        account = api.get_account()
        
        # Extract buying power
        buying_power = 0.0
        if hasattr(account, 'buying_power'):
            buying_power = float(account.buying_power)
        elif hasattr(account, 'cash'):
            buying_power = float(account.cash)
        
        print(f"🏦 Current Buying Power: ${buying_power:,.2f}")
        
        # Analyze status
        if buying_power >= 1000:
            print("✅ EXCELLENT: Substantial buying power available")
            status = "excellent"
        elif buying_power >= 100:
            print("✅ GOOD: Adequate buying power for trading")
            status = "good"
        elif buying_power >= 25:
            print("⚠️ LIMITED: Minimal buying power - small trades only")
            status = "limited"
        else:
            print("❌ INSUFFICIENT: Very low buying power")
            status = "insufficient"
        
        # Calculate safe trading amounts
        if buying_power > 0:
            reserve_amount = max(buying_power * 0.25, 25.0)  # Keep 25% or $25 minimum
            available = buying_power - reserve_amount
            max_trade = min(available * 0.1, 50.0)  # 10% of available, max $50
            
            print(f"🛡️ Safety Reserve: ${reserve_amount:.2f}")
            print(f"💵 Available for Trading: ${available:.2f}")
            print(f"📏 Max Single Trade: ${max_trade:.2f}")
        
        print("=" * 60)
        
        return {
            'buying_power': buying_power,
            'status': status,
            'sufficient': buying_power >= 25.0
        }
        
    except Exception as e:
        print(f"❌ Error checking buying power: {e}")
        print("⚠️ Proceeding with caution - will check again during trading")
        print("=" * 60)
        return {
            'buying_power': 0.0,
            'status': 'unknown',
            'sufficient': False
        }

def setup_production_logging():
    """Setup clean production logging"""
    # Create logs directory
    os.makedirs('logs', exist_ok=True)
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('logs/enhanced_production.log'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    # Set specific log levels for clean operation
    logging.getLogger('alpaca').setLevel(logging.WARNING)
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    logging.getLogger('requests').setLevel(logging.WARNING)

async def run_enhanced_trading():
    """Run the enhanced trading system with smart buying power management"""
    try:
        print("\n🚀 LAUNCHING ENHANCED PRODUCTION SYSTEM")
        print("=" * 60)
        print("📈 Paper Trading: Active with Smart Fund Management")
        print("🪙 Crypto Trading: 24/7 with Reserve Protection")
        print("🧠 RL Optimization: Enabled")
        print("🛡️ Risk Management: Multi-Level + Buying Power Protection")
        print("=" * 60)
        
        # Import the working integrated system
        sys.path.append('/workspaces/competitive-trading-agents')
        from run_integrated_24x7_trading import IntegratedTradingOrchestrator
        
        # Create and run orchestrator
        orchestrator = IntegratedTradingOrchestrator()
        
        print("⏰ Enhanced system started at:", datetime.now().strftime('%H:%M:%S'))
        print("💎 Press Ctrl+C to shutdown gracefully")
        print("🔄 System will refresh buying power every 5 minutes")
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
    """Main enhanced launcher"""
    print("🏁 ENHANCED PRODUCTION PAPER TRADING SYSTEM")
    print(f"📅 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Verify environment safety
    if not verify_paper_trading_environment():
        print("🛑 Environment verification failed - exiting for safety")
        sys.exit(1)
    
    # Check current buying power
    bp_status = check_account_buying_power()
    
    if not bp_status['sufficient']:
        print("⚠️ WARNING: Low buying power detected")
        print("💡 Consider adding funds to your PAPER account for better trading")
        
        user_choice = input("\n🤔 Continue with limited funds? (y/N): ").lower().strip()
        if user_choice != 'y':
            print("🛑 Exiting - Add funds and restart when ready")
            sys.exit(1)
        
        print("⚠️ Continuing with limited fund management...")
    
    # Setup logging
    setup_production_logging()
    
    # Create necessary directories
    os.makedirs('data', exist_ok=True)
    
    try:
        # Run the enhanced trading system
        asyncio.run(run_enhanced_trading())
        print("✅ Enhanced system shutdown completed successfully")
        
    except Exception as e:
        print(f"❌ Critical error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()