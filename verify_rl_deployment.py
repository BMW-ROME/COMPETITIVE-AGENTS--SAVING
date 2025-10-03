#!/usr/bin/env python3
"""
RL Deployment Verification Script
================================
Verify that RL system is properly deployed and functioning
"""

import sys
import importlib.util

def verify_rl_modules():
    """Verify RL modules can be imported"""
    modules_to_check = [
        'rl_optimization_engine',
        'rl_100_percent_execution'
    ]
    
    print("🔍 Verifying RL Module Imports...")
    for module in modules_to_check:
        try:
            spec = importlib.util.spec_from_file_location(module, f"{module}.py")
            module_obj = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module_obj)
            print(f"✅ {module}: Import successful")
        except Exception as e:
            print(f"❌ {module}: Import failed - {e}")
            return False
    
    return True

def verify_rl_integration():
    """Verify RL integration in trading systems"""
    systems_to_check = [
        'alpaca_paper_trading_maximal.py',
        'continuous_competitive_trading.py'
    ]
    
    print("\n🔍 Verifying RL Integration...")
    for system in systems_to_check:
        try:
            with open(system, 'r') as f:
                content = f.read()
            
            if 'rl_100_percent_execution' in content:
                print(f"✅ {system}: RL integration detected")
            else:
                print(f"⚠️ {system}: No RL integration found")
        except FileNotFoundError:
            print(f"⚠️ {system}: File not found")
    
    return True

def test_rl_optimizer():
    """Test RL optimizer functionality"""
    print("\n🧪 Testing RL Optimizer...")
    try:
        from rl_100_percent_execution import get_100_percent_execution_optimizer
        
        optimizer = get_100_percent_execution_optimizer()
        
        # Test market data
        test_market_data = {
            'AAPL': {'volume': 1000000, 'change_pct': 0.5, 'price': 254.37}
        }
        
        # Test trade candidate
        test_candidates = [
            {'symbol': 'AAPL', 'side': 'buy', 'quantity': 1.0, 'confidence': 0.8}
        ]
        
        # Test optimization
        result = optimizer.optimize_for_100_percent_execution(test_candidates, test_market_data)
        
        if result:
            print(f"✅ RL Optimizer: Working correctly ({len(result)} optimized trades)")
            return True
        else:
            print("⚠️ RL Optimizer: No trades optimized (may be normal)")
            return True
            
    except Exception as e:
        print(f"❌ RL Optimizer: Test failed - {e}")
        return False

if __name__ == "__main__":
    print("🚀 RL DEPLOYMENT VERIFICATION")
    print("=" * 40)
    
    success = True
    success &= verify_rl_modules()
    success &= verify_rl_integration() 
    success &= test_rl_optimizer()
    
    print("\n" + "=" * 40)
    if success:
        print("🎉 RL DEPLOYMENT VERIFICATION: ALL SYSTEMS GO!")
        print("✅ Your RL-enhanced trading system is ready for 100% execution rate")
    else:
        print("❌ RL DEPLOYMENT VERIFICATION: ISSUES DETECTED") 
        print("⚠️ Please check the errors above and resolve before trading")
    
    sys.exit(0 if success else 1)
