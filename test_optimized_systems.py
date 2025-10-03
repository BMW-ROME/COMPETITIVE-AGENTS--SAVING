#!/usr/bin/env python3
"""
Quick Test Script for Optimized Trading Systems
==============================================
Tests both optimized systems to ensure they work correctly.
"""

import asyncio
import logging
import sys
import os
import time
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/system_test.log', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger("SystemTest")

async def test_smart_trading_imports():
    """Test smart trading system imports"""
    try:
        # Add src to path
        sys.path.insert(0, '.')
        
        # Test imports
        import run_optimized_smart_trading
        logger.info("✅ Smart trading imports successful")
        return True
    except Exception as e:
        logger.error(f"❌ Smart trading import failed: {e}")
        return False

async def test_ultra_aggressive_imports():
    """Test ultra aggressive system imports"""
    try:
        # Add src to path
        sys.path.insert(0, '.')
        
        # Test imports
        import run_optimized_ultra_aggressive
        logger.info("✅ Ultra aggressive imports successful")
        return True
    except Exception as e:
        logger.error(f"❌ Ultra aggressive import failed: {e}")
        return False

async def test_alpaca_connection():
    """Test Alpaca API connection"""
    try:
        import alpaca_trade_api as tradeapi
        
        api = tradeapi.REST(
            key_id="PKK43GTIACJNUPGZPCPF",
            secret_key="CuQWde4QtPHAtwuMfxaQhB8njmrcJDq0YK4Oz9Rw",
            base_url="https://paper-api.alpaca.markets",
            api_version='v2'
        )
        
        # Test connection
        account = api.get_account()
        logger.info(f"✅ Alpaca connection successful - Account: {account.id}")
        return True
    except Exception as e:
        logger.error(f"❌ Alpaca connection failed: {e}")
        return False

async def test_docker_availability():
    """Test Docker availability"""
    try:
        import subprocess
        
        result = subprocess.run(['docker', '--version'], capture_output=True, text=True)
        if result.returncode == 0:
            logger.info("✅ Docker available")
            return True
        else:
            logger.error("❌ Docker not available")
            return False
    except Exception as e:
        logger.error(f"❌ Docker test failed: {e}")
        return False

async def test_deployment_scripts():
    """Test deployment scripts exist"""
    scripts = [
        'deploy_optimized_smart.bat',
        'deploy_optimized_ultra.bat'
    ]
    
    all_exist = True
    for script in scripts:
        if os.path.exists(script):
            logger.info(f"✅ {script} exists")
        else:
            logger.error(f"❌ {script} missing")
            all_exist = False
    
    return all_exist

async def run_comprehensive_test():
    """Run comprehensive system test"""
    logger.info("=" * 60)
    logger.info("OPTIMIZED SYSTEMS COMPREHENSIVE TEST")
    logger.info("=" * 60)
    
    # Create logs directory
    os.makedirs('logs', exist_ok=True)
    
    tests = [
        ("Smart Trading Imports", test_smart_trading_imports),
        ("Ultra Aggressive Imports", test_ultra_aggressive_imports),
        ("Alpaca Connection", test_alpaca_connection),
        ("Docker Availability", test_docker_availability),
        ("Deployment Scripts", test_deployment_scripts)
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        logger.info(f"Running {test_name}...")
        try:
            result = await test_func()
            if result:
                passed += 1
                logger.info(f"✅ {test_name} PASSED")
            else:
                failed += 1
                logger.error(f"❌ {test_name} FAILED")
        except Exception as e:
            failed += 1
            logger.error(f"❌ {test_name} FAILED with exception: {e}")
        
        logger.info("")
    
    # Summary
    logger.info("=" * 60)
    logger.info("TEST SUMMARY")
    logger.info("=" * 60)
    logger.info(f"✅ Passed: {passed}")
    logger.info(f"❌ Failed: {failed}")
    
    if failed == 0:
        logger.info("🎉 ALL TESTS PASSED! Systems are ready for deployment!")
        logger.info("")
        logger.info("DEPLOYMENT COMMANDS:")
        logger.info("  Smart Trading:    .\\deploy_optimized_smart.bat")
        logger.info("  Ultra Aggressive: .\\deploy_optimized_ultra.bat")
    else:
        logger.warning(f"⚠️ {failed} tests failed. Please fix issues before deployment.")
    
    return failed == 0

async def main():
    """Main test function"""
    success = await run_comprehensive_test()
    return success

if __name__ == "__main__":
    asyncio.run(main())

