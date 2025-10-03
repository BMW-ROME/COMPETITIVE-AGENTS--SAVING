#!/usr/bin/env python3
"""
Safe Mode Switcher - Interactive Trading Mode Management
=======================================================
Provides a safe, interactive interface for switching between
Paper, Live, and Simulation trading modes.
"""

import sys
import os
import logging
from datetime import datetime
from typing import Dict, Any

# Add src to path
sys.path.insert(0, 'src')

from trading_mode_manager import TradingModeManager, TradingMode
from dynamic_config_manager import DynamicConfigManager

class SafeModeSwitcher:
    """Interactive mode switching with safety checks"""
    
    def __init__(self):
        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger("SafeModeSwitcher")
        
        # Initialize managers
        self.mode_manager = TradingModeManager(self.logger)
        self.config_manager = DynamicConfigManager(self.logger)
        self.config_manager.set_trading_mode_manager(self.mode_manager)
        
    def display_current_status(self):
        """Display current system status"""
        print("\n" + "="*60)
        print("🎯 ULTIMATE TRADING SYSTEM - MODE STATUS")
        print("="*60)
        
        mode_info = self.mode_manager.get_mode_info()
        safety_status = self.mode_manager.get_safety_status()
        
        print(f"📊 Current Mode: {mode_info['current_mode']}")
        print(f"💰 Real Money: {'YES' if mode_info['real_money'] else 'NO'}")
        print(f"🛡️  Safety Level: {mode_info['safety_level']}")
        print(f"📈 Trading Enabled: {'YES' if mode_info['trading_enabled'] else 'NO'}")
        print(f"📏 Max Position: {mode_info['max_position_size']*100:.1f}%")
        print(f"📉 Max Daily Loss: {mode_info['max_daily_loss']*100:.1f}%")
        print(f"🛑 Stop Loss: {mode_info['stop_loss_pct']*100:.1f}%")
        print(f"🔄 Last Mode Switch: {safety_status['last_mode_switch'] or 'Never'}")
        print(f"⏰ Cooldown Active: {'YES' if safety_status['cooldown_active'] else 'NO'}")
        
        if mode_info['real_money']:
            print("\n🚨 WARNING: LIVE TRADING MODE - REAL MONEY AT RISK! 🚨")
        
        print("="*60)
    
    def display_mode_options(self):
        """Display available mode switching options"""
        print("\n📋 AVAILABLE MODE SWITCHES:")
        print("-" * 40)
        
        guide = self.config_manager.get_mode_switching_guide()
        
        for switch_type, info in guide.items():
            print(f"\n🔄 {switch_type.upper().replace('_', ' ')}")
            print(f"   Description: {info['description']}")
            print(f"   Safety Level: {info['safety_level']}")
            print(f"   Confirmations Required: {info['confirmations_required']}")
            
            if 'changes' in info:
                print("   Changes:")
                for change in info['changes']:
                    print(f"     • {change}")
            
            if 'requirements' in info:
                print("   Requirements:")
                for req in info['requirements']:
                    print(f"     • {req}")
    
    def get_user_choice(self) -> str:
        """Get user's mode switching choice"""
        print("\n🎯 MODE SWITCHING OPTIONS:")
        print("1. Switch to PAPER mode (Safe testing)")
        print("2. Switch to SIMULATION mode (Analysis only)")
        print("3. Switch to LIVE mode (Real money - DANGER)")
        print("4. Emergency stop (Halt all trading)")
        print("5. View current configuration")
        print("6. Validate configuration")
        print("7. Exit")
        
        while True:
            try:
                choice = input("\nEnter your choice (1-7): ").strip()
                if choice in ['1', '2', '3', '4', '5', '6', '7']:
                    return choice
                else:
                    print("❌ Invalid choice. Please enter 1-7.")
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                sys.exit(0)
    
    def handle_paper_mode_switch(self):
        """Handle switching to paper mode"""
        print("\n📄 SWITCHING TO PAPER MODE")
        print("-" * 30)
        
        result = self.mode_manager.switch_mode(TradingMode.PAPER)
        
        if result['success']:
            print("✅ Successfully switched to PAPER mode")
            print("📄 Safe testing environment activated")
            print("💰 No real money at risk")
        else:
            print(f"❌ Failed to switch: {result['message']}")
        
        if result['warnings']:
            print("\n⚠️  Warnings:")
            for warning in result['warnings']:
                print(f"   • {warning}")
    
    def handle_simulation_mode_switch(self):
        """Handle switching to simulation mode"""
        print("\n🧪 SWITCHING TO SIMULATION MODE")
        print("-" * 35)
        
        result = self.mode_manager.switch_mode(TradingMode.SIMULATION)
        
        if result['success']:
            print("✅ Successfully switched to SIMULATION mode")
            print("🧪 Analysis-only mode activated")
            print("📊 No trading, analysis only")
        else:
            print(f"❌ Failed to switch: {result['message']}")
        
        if result['warnings']:
            print("\n⚠️  Warnings:")
            for warning in result['warnings']:
                print(f"   • {warning}")
    
    def handle_live_mode_switch(self):
        """Handle switching to live mode with safety checks"""
        print("\n🚨 SWITCHING TO LIVE MODE")
        print("-" * 30)
        print("⚠️  WARNING: This will enable REAL MONEY TRADING!")
        print("💰 Real money will be at risk!")
        print("🛡️  Ensure you have proper risk management in place!")
        
        # Check if credentials are valid
        if not self.mode_manager._validate_live_mode_credentials():
            print("\n❌ LIVE MODE BLOCKED")
            print("🔑 Invalid or missing Alpaca credentials")
            print("📝 Please set valid ALPACA_API_KEY and ALPACA_SECRET_KEY")
            print("💡 Use environment variables or .env file")
            return
        
        # Get confirmations
        confirmations = 0
        required_confirmations = 3
        
        print(f"\n🔐 LIVE MODE REQUIRES {required_confirmations} CONFIRMATIONS")
        
        for i in range(required_confirmations):
            try:
                confirm = input(f"Confirmation {i+1}/{required_confirmations}: Type 'LIVE' to confirm: ").strip()
                if confirm == 'LIVE':
                    confirmations += 1
                    print(f"✅ Confirmation {i+1} received")
                else:
                    print("❌ Invalid confirmation. Live mode switch cancelled.")
                    return
            except KeyboardInterrupt:
                print("\n❌ Live mode switch cancelled.")
                return
        
        # Perform the switch
        result = self.mode_manager.switch_mode(TradingMode.LIVE, confirmations)
        
        if result['success']:
            print("\n🚨 LIVE TRADING MODE ACTIVATED! 🚨")
            print("💰 REAL MONEY IS NOW AT RISK!")
            print("🛡️  Monitor your positions carefully!")
        else:
            print(f"❌ Failed to switch: {result['message']}")
        
        if result['warnings']:
            print("\n⚠️  Warnings:")
            for warning in result['warnings']:
                print(f"   • {warning}")
    
    def handle_emergency_stop(self):
        """Handle emergency stop"""
        print("\n🚨 EMERGENCY STOP")
        print("-" * 20)
        print("⚠️  This will halt ALL TRADING immediately!")
        
        try:
            confirm = input("Type 'STOP' to confirm emergency stop: ").strip()
            if confirm == 'STOP':
                result = self.mode_manager.emergency_stop()
                
                if result['success']:
                    print("🚨 EMERGENCY STOP ACTIVATED!")
                    print("🛑 All trading halted")
                    print("📊 System switched to simulation mode")
                else:
                    print(f"❌ Emergency stop failed: {result['message']}")
            else:
                print("❌ Emergency stop cancelled.")
        except KeyboardInterrupt:
            print("\n❌ Emergency stop cancelled.")
    
    def display_configuration(self):
        """Display current configuration"""
        print("\n📋 CURRENT CONFIGURATION:")
        print("-" * 30)
        
        config = self.config_manager.get_dynamic_config()
        
        for key, value in config.items():
            if isinstance(value, dict):
                print(f"{key}:")
                for sub_key, sub_value in value.items():
                    print(f"  {sub_key}: {sub_value}")
            else:
                print(f"{key}: {value}")
    
    def validate_configuration(self):
        """Validate current configuration"""
        print("\n🔍 VALIDATING CONFIGURATION:")
        print("-" * 35)
        
        validation = self.config_manager.validate_configuration()
        
        if validation['valid']:
            print("✅ Configuration is valid")
        else:
            print("❌ Configuration has errors")
        
        if validation['warnings']:
            print("\n⚠️  Warnings:")
            for warning in validation['warnings']:
                print(f"   • {warning}")
        
        if validation['errors']:
            print("\n❌ Errors:")
            for error in validation['errors']:
                print(f"   • {error}")
        
        if validation['recommendations']:
            print("\n💡 Recommendations:")
            for rec in validation['recommendations']:
                print(f"   • {rec}")
    
    def run(self):
        """Main interactive loop"""
        print("🎯 ULTIMATE TRADING SYSTEM - SAFE MODE SWITCHER")
        print("=" * 60)
        
        while True:
            try:
                self.display_current_status()
                choice = self.get_user_choice()
                
                if choice == '1':
                    self.handle_paper_mode_switch()
                elif choice == '2':
                    self.handle_simulation_mode_switch()
                elif choice == '3':
                    self.handle_live_mode_switch()
                elif choice == '4':
                    self.handle_emergency_stop()
                elif choice == '5':
                    self.display_configuration()
                elif choice == '6':
                    self.validate_configuration()
                elif choice == '7':
                    print("\n👋 Goodbye!")
                    break
                
                input("\nPress Enter to continue...")
                
            except KeyboardInterrupt:
                print("\n\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"\n❌ Error: {e}")
                input("Press Enter to continue...")

if __name__ == "__main__":
    switcher = SafeModeSwitcher()
    switcher.run()

