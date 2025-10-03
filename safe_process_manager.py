#!/usr/bin/env python3
"""
Safe Process Manager for Trading System
Handles process cleanup without hanging pkill commands
"""

import os
import psutil
import time
import signal
import subprocess
from datetime import datetime

def find_trading_processes():
    """Find all trading-related processes safely"""
    trading_processes = []
    
    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            if proc.info['cmdline']:
                cmdline = ' '.join(proc.info['cmdline'])
                
                # Look for trading system processes
                if any(keyword in cmdline.lower() for keyword in [
                    'launch_', 'trading', 'crypto_24x7', 'competitive',
                    'continuous_', 'paper_trading', 'alpaca'
                ]):
                    # Exclude this script itself
                    if 'safe_process_manager' not in cmdline:
                        trading_processes.append({
                            'pid': proc.info['pid'],
                            'name': proc.info['name'],
                            'cmdline': cmdline[:100] + ('...' if len(cmdline) > 100 else '')
                        })
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    
    return trading_processes

def stop_trading_processes_safely():
    """Stop trading processes safely without hanging"""
    print("🔍 Finding trading processes...")
    
    processes = find_trading_processes()
    
    if not processes:
        print("✅ No trading processes found")
        return True
    
    print(f"📋 Found {len(processes)} trading processes:")
    for proc in processes:
        print(f"  PID {proc['pid']}: {proc['cmdline']}")
    
    # Send TERM signal first (graceful)
    print("\n🔄 Sending TERM signal for graceful shutdown...")
    for proc in processes:
        try:
            os.kill(proc['pid'], signal.SIGTERM)
            print(f"  ✅ TERM sent to PID {proc['pid']}")
        except (OSError, ProcessLookupError):
            print(f"  ⚠️ PID {proc['pid']} already gone")
    
    # Wait for graceful shutdown
    print("\n⏳ Waiting 5 seconds for graceful shutdown...")
    time.sleep(5)
    
    # Check what's still running
    remaining = find_trading_processes()
    
    if not remaining:
        print("✅ All processes stopped gracefully")
        return True
    
    # Force kill remaining processes
    print(f"\n🔨 Force killing {len(remaining)} remaining processes...")
    for proc in remaining:
        try:
            os.kill(proc['pid'], signal.SIGKILL)
            print(f"  💀 KILL sent to PID {proc['pid']}")
        except (OSError, ProcessLookupError):
            print(f"  ⚠️ PID {proc['pid']} already gone")
    
    # Final check
    time.sleep(2)
    final_check = find_trading_processes()
    
    if final_check:
        print(f"❌ {len(final_check)} processes still running - may need manual intervention")
        for proc in final_check:
            print(f"  Stuck: PID {proc['pid']}")
        return False
    
    print("✅ All trading processes stopped successfully")
    return True

def check_environment_safely():
    """Check environment variables with proper loading"""
    print("🔒 ENVIRONMENT SAFETY CHECK")
    print("=" * 50)
    
    # Load environment from .env file
    env_file = "/workspaces/competitive-trading-agents/.env"
    
    if not os.path.exists(env_file):
        print("❌ .env file not found!")
        return False
    
    # Parse .env file manually (more reliable than python-dotenv in some cases)
    env_vars = {}
    try:
        with open(env_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    env_vars[key.strip()] = value.strip()
    except Exception as e:
        print(f"❌ Error reading .env file: {e}")
        return False
    
    # Check critical environment variables
    alpaca_key = env_vars.get('ALPACA_API_KEY') or env_vars.get('APCA_API_KEY_ID', '')
    alpaca_secret = env_vars.get('ALPACA_SECRET_KEY') or env_vars.get('APCA_API_SECRET_KEY', '')
    alpaca_base_url = env_vars.get('ALPACA_BASE_URL') or env_vars.get('APCA_API_BASE_URL', '')
    
    print(f"🔍 Environment Variables Found:")
    print(f"  API Key: {'✅' if alpaca_key else '❌'}")
    print(f"  Secret: {'✅' if alpaca_secret else '❌'}")
    print(f"  Base URL: {alpaca_base_url}")
    
    # Determine environment type
    if 'paper-api.alpaca.markets' in alpaca_base_url:
        env_type = "PAPER TRADING"
        env_color = "🟢"
        safety_status = "SAFE"
    elif 'api.alpaca.markets' in alpaca_base_url:
        env_type = "LIVE TRADING"
        env_color = "🔴"
        safety_status = "⚠️ LIVE MONEY AT RISK ⚠️"
    else:
        env_type = "UNKNOWN"
        env_color = "🟡"
        safety_status = "VERIFY CONFIGURATION"
    
    print(f"\n{env_color} ENVIRONMENT: {env_type}")
    print(f"🛡️  STATUS: {safety_status}")
    
    if alpaca_key:
        masked_key = alpaca_key[:6] + "..." + alpaca_key[-4:] if len(alpaca_key) > 10 else "***"
        print(f"🔑 API Key: {masked_key}")
    
    print("=" * 50)
    
    return env_type == "PAPER TRADING"

def main():
    """Main process manager"""
    print(f"🛡️  Safe Process Manager - {datetime.now().strftime('%H:%M:%S')}")
    print("=" * 60)
    
    # Check environment first
    if not check_environment_safely():
        print("❌ Environment not confirmed as PAPER trading")
        print("🛑 Exiting for safety")
        return False
    
    # Stop any existing processes
    if not stop_trading_processes_safely():
        print("⚠️ Some processes may still be running")
        return False
    
    print("\n✅ System ready for clean startup")
    return True

if __name__ == "__main__":
    main()