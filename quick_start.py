#!/usr/bin/env python3
"""
Quick Start - Continuous Trading
================================
One-click launch for your competitive trading system
"""

import subprocess
import sys
import os
from datetime import datetime

def setup_environment():
    """Setup Python environment and dependencies"""
    print("🔧 Setting up environment...")
    
    # Activate virtual environment and install packages
    try:
        venv_python = "/workspaces/competitive-trading-agents/.venv/bin/python"
        if os.path.exists(venv_python):
            print("✅ Virtual environment found")
            return venv_python
        else:
            print("❌ Virtual environment not found")
            return "python3"
    except Exception as e:
        print(f"⚠️ Environment setup warning: {e}")
        return "python3"

def check_requirements():
    """Check if all requirements are met"""
    print("📋 Checking requirements...")
    
    # Check .env file
    if os.path.exists('.env'):
        print("✅ Environment file found")
    else:
        print("❌ .env file not found - please ensure API keys are configured")
        return False
    
    # Check logs directory
    os.makedirs('logs', exist_ok=True)
    os.makedirs('data', exist_ok=True)
    print("✅ Directories created")
    
    return True

def launch_continuous_trading():
    """Launch the continuous trading system"""
    print("\n🚀 LAUNCHING CONTINUOUS COMPETITIVE TRADING SYSTEM")
    print("=" * 55)
    print("📅 Start Time:", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    print("🎯 Mode: 24/7 Multi-Agent Paper Trading")
    print("💎 Press Ctrl+C to stop gracefully")
    print("📊 Logs will be saved to logs/ directory")
    print("=" * 55)
    print()
    
    # Get Python executable
    python_exe = setup_environment()
    
    # Check requirements
    if not check_requirements():
        print("❌ Requirements check failed")
        return
    
    try:
        # Launch continuous trading
        cmd = [python_exe, "continuous_competitive_trading.py"]
        process = subprocess.run(cmd, cwd="/workspaces/competitive-trading-agents")
        
    except KeyboardInterrupt:
        print("\n👋 Graceful shutdown completed")
    except Exception as e:
        print(f"\n💥 Error launching system: {e}")

if __name__ == "__main__":
    print("🎯 COMPETITIVE TRADING SYSTEM - QUICK START")
    print("=" * 50)
    
    launch_continuous_trading()