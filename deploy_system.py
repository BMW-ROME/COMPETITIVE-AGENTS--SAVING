#!/usr/bin/env python3
"""
FINAL DEPLOYMENT - Competitive Trading System
=============================================
Your ultimate competitive trading system ready for continuous operation
"""

import subprocess
import sys
import os
import time
from datetime import datetime

def print_banner():
    """Print deployment banner"""
    print("🚀" * 25)
    print("🎯 COMPETITIVE TRADING SYSTEM DEPLOYMENT")
    print("🏆 Your Multi-Agent Trading Platform is Ready!")
    print("💎 Paper Trading Mode - Safe for Testing")
    print("🚀" * 25)
    print()

def check_system():
    """Check system requirements"""
    print("🔧 System Check:")
    
    # Check Python environment
    venv_python = "/workspaces/competitive-trading-agents/.venv/bin/python"
    if os.path.exists(venv_python):
        print("  ✅ Python environment: Ready")
    else:
        print("  ❌ Python environment: Missing")
        return False
    
    # Check environment file
    if os.path.exists('.env'):
        print("  ✅ Configuration: Ready")
    else:
        print("  ❌ Configuration: Missing .env file")
        return False
    
    # Check trading script
    if os.path.exists('ultra_light_trading.py'):
        print("  ✅ Trading system: Ready")
    else:
        print("  ❌ Trading system: Missing script")
        return False
    
    # Create directories
    os.makedirs('logs', exist_ok=True)
    os.makedirs('data', exist_ok=True)
    print("  ✅ Directories: Ready")
    
    print()
    return True

def show_features():
    """Show system features"""
    print("🎯 SYSTEM FEATURES:")
    print("  • 12 Competitive Trading Agents")
    print("  • Real-time Decision Making")
    print("  • Alpaca Paper Trading Integration")
    print("  • Risk Management & Position Sizing")
    print("  • Continuous 24/7 Operation")
    print("  • Comprehensive Logging")
    print("  • Graceful Shutdown (Ctrl+C)")
    print("  • Performance Tracking")
    print()

def show_status():
    """Show current system status"""
    print("📊 SYSTEM STATUS:")
    print(f"  • Deployment Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("  • Mode: Paper Trading (Safe)")
    print("  • API: Alpaca Markets")
    print("  • Agents: 12 Active")
    print("  • Cycle Interval: 90 seconds")
    print()

def get_user_choice():
    """Get user deployment choice"""
    print("🚀 DEPLOYMENT OPTIONS:")
    print("  1. Start Continuous Trading (Recommended)")
    print("  2. Run Quick Test (5 minutes)")
    print("  3. View System Information")
    print("  4. Exit")
    print()
    
    while True:
        try:
            choice = input("Select option (1-4): ").strip()
            if choice in ['1', '2', '3', '4']:
                return int(choice)
            else:
                print("Please enter 1, 2, 3, or 4")
        except KeyboardInterrupt:
            return 4

def run_continuous_trading():
    """Run continuous trading system"""
    print("\n🎯 LAUNCHING CONTINUOUS TRADING SYSTEM")
    print("=" * 50)
    print("💡 The system will run continuously until you stop it")
    print("📊 Progress updates every 5 cycles")
    print("💾 Logs saved to logs/ directory")
    print("🔄 Press Ctrl+C to stop gracefully")
    print("=" * 50)
    print()
    
    try:
        venv_python = "/workspaces/competitive-trading-agents/.venv/bin/python"
        cmd = [venv_python, "ultra_light_trading.py"]
        subprocess.run(cmd, cwd="/workspaces/competitive-trading-agents")
    except KeyboardInterrupt:
        print("\n👋 Trading system stopped gracefully")
    except Exception as e:
        print(f"\n❌ Error: {e}")

def run_quick_test():
    """Run quick test"""
    print("\n🧪 RUNNING QUICK TEST (5 MINUTES)")
    print("=" * 40)
    print("⏱️ System will run for 5 minutes then stop automatically")
    print("📊 You'll see real-time trading activity")
    print("=" * 40)
    print()
    
    try:
        venv_python = "/workspaces/competitive-trading-agents/.venv/bin/python"
        cmd = ["timeout", "300", venv_python, "ultra_light_trading.py"]
        result = subprocess.run(cmd, cwd="/workspaces/competitive-trading-agents")
        
        if result.returncode == 124:  # Timeout
            print("\n✅ Quick test completed successfully!")
        else:
            print("\n⚠️ Test ended early")
            
    except Exception as e:
        print(f"\n❌ Test error: {e}")

def show_system_info():
    """Show detailed system information"""
    print("\n📋 DETAILED SYSTEM INFORMATION")
    print("=" * 45)
    print()
    
    print("🏗️ ARCHITECTURE:")
    print("  • Multi-agent competitive framework")
    print("  • Event-driven decision making")
    print("  • REST API integration with Alpaca")
    print("  • Asynchronous processing")
    print("  • Signal-based graceful shutdown")
    print()
    
    print("🤖 AGENT TYPES:")
    print("  • Conservative traders (low risk)")
    print("  • Balanced traders (medium risk)")
    print("  • Aggressive traders (high risk)")
    print("  • Momentum followers")
    print("  • Scalping specialists")
    print("  • Opportunity hunters")
    print()
    
    print("💰 RISK MANAGEMENT:")
    print("  • Maximum 2% of buying power per trade")
    print("  • Minimum $5 trade size")
    print("  • Maximum 8% total exposure per cycle")
    print("  • Automatic position sizing")
    print("  • Paper trading safety")
    print()
    
    print("📊 MONITORING:")
    print("  • Real-time performance tracking")
    print("  • Detailed logging to files")
    print("  • Agent performance metrics")
    print("  • P&L calculation")
    print("  • Trade execution statistics")
    print()

def main():
    """Main deployment function"""
    print_banner()
    
    # Check system
    if not check_system():
        print("❌ System check failed. Please fix issues and try again.")
        return
    
    show_features()
    show_status()
    
    while True:
        choice = get_user_choice()
        
        if choice == 1:
            run_continuous_trading()
        elif choice == 2:
            run_quick_test()
        elif choice == 3:
            show_system_info()
            input("\nPress Enter to continue...")
        elif choice == 4:
            print("\n👋 Goodbye! Your trading system is ready when you are.")
            break

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 Deployment cancelled by user")
    except Exception as e:
        print(f"\n❌ Deployment error: {e}")