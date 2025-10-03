#!/usr/bin/env python3
"""
Quick Launch: 24/7 Crypto + Paper Trading Integration
Execute immediately - all modules integrated
"""

import sys
import os
import asyncio
from datetime import datetime

# Add path
sys.path.append('/workspaces/competitive-trading-agents')

print("🌟 24/7 Crypto + Paper Trading System")
print("=" * 50)
print("🪙 Crypto: Always Open Markets")
print("📈 Paper: US Market Hours")
print("🧠 RL: 100% Execution Optimization")
print("⚙️ Integration: ALL MODULES")
print("=" * 50)
print(f"🕐 Launch Time: {datetime.now()}")
print()

# Import and run
from run_integrated_24x7_trading import main

if __name__ == "__main__":
    try:
        print("🚀 Starting integrated trading platform...")
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n🔴 Graceful shutdown requested")
    except Exception as e:
        print(f"\n💥 Error: {e}")
        print("📋 Check logs for details")