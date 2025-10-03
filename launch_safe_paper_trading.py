#!/usr/bin/env python3
"""
Safe Paper Trading Launcher with Clear Environment Acknowledgment
Explicitly shows PAPER vs LIVE environment for safety
"""

import os
import sys
import subprocess
import signal
import time
from datetime import datetime

def check_environment_safety():
    """Check and display environment configuration for safety"""
    print("=" * 80)
    print("🔒 TRADING ENVIRONMENT SAFETY CHECK")
    print("=" * 80)
    
    # Check .env file exists
    env_file = "/workspaces/competitive-trading-agents/.env"
    if not os.path.exists(env_file):
        print("❌ ERROR: .env file not found!")
        return False
    
    # Load and check environment variables
    from dotenv import load_dotenv
    load_dotenv(env_file)  # Explicitly load from .env file
    
    # Critical safety checks - check both ALPACA_ and APCA_ prefixes
    alpaca_key = os.getenv('ALPACA_API_KEY', '') or os.getenv('APCA_API_KEY_ID', '')
    alpaca_secret = os.getenv('ALPACA_SECRET_KEY', '') or os.getenv('APCA_API_SECRET_KEY', '')
    alpaca_base_url = os.getenv('ALPACA_BASE_URL', '') or os.getenv('APCA_API_BASE_URL', '')
    
    print(f"🔍 Loading from: {env_file}")
    print(f"🔍 Found API Key: {'Yes' if alpaca_key else 'No'}")
    print(f"🔍 Found Secret: {'Yes' if alpaca_secret else 'No'}")
    print(f"🔍 Base URL: {alpaca_base_url}")
    
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
    
    # Show key configuration (masked for security)
    if alpaca_key:
        masked_key = alpaca_key[:6] + "..." + alpaca_key[-4:] if len(alpaca_key) > 10 else "***"
        print(f"🔑 API Key: {masked_key}")
    else:
        print("❌ No API Key found!")
        return False
    
    if alpaca_secret:
        print(f"🔐 Secret Key: {'*' * 10} (configured)")
    else:
        print("❌ No Secret Key found!")
        return False
    
    print("=" * 80)
    
    # Safety confirmation for LIVE trading
    if env_type == "LIVE TRADING":
        print("⚠️  WARNING: LIVE TRADING DETECTED!")
        print("⚠️  REAL MONEY WILL BE AT RISK!")
        confirmation = input("Type 'LIVE TRADING CONFIRMED' to proceed: ")
        if confirmation != "LIVE TRADING CONFIRMED":
            print("❌ Live trading not confirmed. Exiting for safety.")
            return False
    
    print(f"✅ Environment verified: {env_type}")
    return True

def launch_integrated_system():
    """Launch the integrated trading system"""
    try:
        print("\n🚀 Launching Integrated Trading System...")
        print(f"⏰ Start Time: {datetime.now().strftime('%H:%M:%S')}")
        
        # Import and run the integrated system
        sys.path.append('/workspaces/competitive-trading-agents')
        
        print("🔧 Initializing Enhanced Trading System...")
        print("▶️  Starting continuous trading loop...")
        print("💡 Press Ctrl+C to stop trading safely")
        print("-" * 60)
        
        # Import and run the async main function
        import asyncio
        from run_enhanced_trading_system import main
        
        # Run the async main function
        asyncio.run(main())
        
    except KeyboardInterrupt:
        print("\n⏸️  Graceful shutdown requested...")
        return True
    except Exception as e:
        print(f"\n❌ System error: {str(e)}")
        return False

def main():
    """Main launcher with safety checks"""
    print("🏁 Starting Safe Paper Trading System")
    print(f"📅 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Perform safety checks first
    if not check_environment_safety():
        print("🛑 Safety checks failed. System not started.")
        sys.exit(1)
    
    # Add delay for user to review environment
    print("\n⏳ Starting system in 3 seconds...")
    time.sleep(1)
    print("⏳ Starting system in 2 seconds...")
    time.sleep(1)
    print("⏳ Starting system in 1 second...")
    time.sleep(1)
    
    # Launch the system
    if launch_integrated_system():
        print("\n✅ System shutdown completed successfully")
    else:
        print("\n❌ System encountered an error")
        sys.exit(1)

if __name__ == "__main__":
    main()