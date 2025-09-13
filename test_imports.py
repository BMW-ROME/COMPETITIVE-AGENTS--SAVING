#!/usr/bin/env python3
"""
Test script to verify all imports work correctly.
"""

import sys
import os

# Add the current directory to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_imports():
    """Test all imports to ensure they work."""
    try:
        print("Testing imports...")
        
        # Test config imports
        print("  ✓ Testing config imports...")
        from config.settings import SystemConfig, TradingMode, AgentConfig, AgentType
        print("  ✓ Config imports successful")
        
        # Test src imports
        print("  ✓ Testing src imports...")
        from src.base_agent import BaseTradingAgent, TradeDecision, PerformanceMetrics
        from src.trading_agents import ConservativeTradingAgent, AggressiveTradingAgent, BalancedTradingAgent
        from src.hierarchy_manager import HierarchyManager, AgentReport
        from src.data_sources import DataAggregator, MarketDataProvider, NewsProvider, SocialMediaProvider
        from src.alpaca_integration import AlpacaTradingInterface
        from src.system_orchestrator import TradingSystemOrchestrator
        print("  ✓ Src imports successful")
        
        # Test creating basic objects
        print("  ✓ Testing object creation...")
        config = SystemConfig()
        print(f"  ✓ SystemConfig created: {config.trading_mode}")
        
        agent_config = AgentConfig("test_agent", AgentType.CONSERVATIVE)
        print(f"  ✓ AgentConfig created: {agent_config.agent_id}")
        
        print("\n🎉 All imports and basic object creation successful!")
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

if __name__ == "__main__":
    success = test_imports()
    if success:
        print("\n✅ Ready to run the trading system!")
    else:
        print("\n❌ Please fix the import issues before running the system.")
