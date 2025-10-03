#!/usr/bin/env python3
"""
Comprehensive System Check
=========================

Check all critical files and dependencies to ensure the system is properly aligned.
"""

import sys
import os
import importlib
import logging
from typing import List, Dict, Any

# Add src to path
sys.path.append('src')

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def check_file_exists(file_path: str) -> bool:
    """Check if a file exists."""
    return os.path.exists(file_path)

def check_import(module_name: str, class_name: str = None) -> bool:
    """Check if a module can be imported."""
    try:
        module = importlib.import_module(module_name)
        if class_name:
            return hasattr(module, class_name)
        return True
    except ImportError as e:
        logger.error(f"Import error for {module_name}: {e}")
        return False
    except Exception as e:
        logger.error(f"Error checking {module_name}: {e}")
        return False

def check_critical_files() -> Dict[str, bool]:
    """Check all critical files exist."""
    critical_files = {
        # Core system files
        "src/system_orchestrator.py": check_file_exists("src/system_orchestrator.py"),
        "src/base_agent.py": check_file_exists("src/base_agent.py"),
        "src/trading_agents.py": check_file_exists("src/trading_agents.py"),
        "forex_crypto_agents.py": check_file_exists("forex_crypto_agents.py"),
        
        # Configuration
        "config/settings.py": check_file_exists("config/settings.py"),
        
        # Data sources
        "src/data_sources.py": check_file_exists("src/data_sources.py"),
        "src/mock_data_provider.py": check_file_exists("src/mock_data_provider.py"),
        
        # Alpaca integration
        "src/real_alpaca_integration.py": check_file_exists("src/real_alpaca_integration.py"),
        "src/alpaca_http_integration.py": check_file_exists("src/alpaca_http_integration.py"),
        "src/alpaca_integration.py": check_file_exists("src/alpaca_integration.py"),
        
        # Advanced systems
        "src/backtesting_engine.py": check_file_exists("src/backtesting_engine.py"),
        "src/sentiment_analyzer.py": check_file_exists("src/sentiment_analyzer.py"),
        "src/advanced_risk_manager.py": check_file_exists("src/advanced_risk_manager.py"),
        "src/ml_enhancement.py": check_file_exists("src/ml_enhancement.py"),
        "src/multi_timeframe_analyzer.py": check_file_exists("src/multi_timeframe_analyzer.py"),
        "src/correlation_analyzer.py": check_file_exists("src/correlation_analyzer.py"),
        "src/market_hours_optimizer.py": check_file_exists("src/market_hours_optimizer.py"),
        "src/performance_analytics.py": check_file_exists("src/performance_analytics.py"),
        
        # System management
        "src/hierarchy_manager.py": check_file_exists("src/hierarchy_manager.py"),
        "src/persistence.py": check_file_exists("src/persistence.py"),
        
        # Main entry points
        "continuous_real_alpaca_trading.py": check_file_exists("continuous_real_alpaca_trading.py"),
        "monitoring_dashboard.py": check_file_exists("monitoring_dashboard.py"),
        
        # Docker files
        "Dockerfile": check_file_exists("Dockerfile"),
        "docker-compose.yml": check_file_exists("docker-compose.yml"),
        "requirements.txt": check_file_exists("requirements.txt"),
    }
    
    return critical_files

def check_critical_imports() -> Dict[str, bool]:
    """Check all critical imports work."""
    critical_imports = {
        # Core system imports
        "src.system_orchestrator": check_import("src.system_orchestrator"),
        "src.base_agent": check_import("src.base_agent"),
        "src.trading_agents": check_import("src.trading_agents"),
        "forex_crypto_agents": check_import("forex_crypto_agents"),
        
        # Configuration imports
        "config.settings": check_import("config.settings"),
        
        # Data source imports
        "src.data_sources": check_import("src.data_sources"),
        "src.mock_data_provider": check_import("src.mock_data_provider"),
        
        # Alpaca integration imports
        "src.real_alpaca_integration": check_import("src.real_alpaca_integration"),
        "src.alpaca_http_integration": check_import("src.alpaca_http_integration"),
        "src.alpaca_integration": check_import("src.alpaca_integration"),
        
        # Advanced system imports
        "src.backtesting_engine": check_import("src.backtesting_engine"),
        "src.sentiment_analyzer": check_import("src.sentiment_analyzer"),
        "src.advanced_risk_manager": check_import("src.advanced_risk_manager"),
        "src.ml_enhancement": check_import("src.ml_enhancement"),
        "src.multi_timeframe_analyzer": check_import("src.multi_timeframe_analyzer"),
        "src.correlation_analyzer": check_import("src.correlation_analyzer"),
        "src.market_hours_optimizer": check_import("src.market_hours_optimizer"),
        "src.performance_analytics": check_import("src.performance_analytics"),
        
        # System management imports
        "src.hierarchy_manager": check_import("src.hierarchy_manager"),
        "src.persistence": check_import("src.persistence"),
    }
    
    return critical_imports

def check_agent_classes() -> Dict[str, bool]:
    """Check all agent classes can be imported."""
    agent_classes = {
        "ConservativeTradingAgent": check_import("src.trading_agents", "ConservativeTradingAgent"),
        "AggressiveTradingAgent": check_import("src.trading_agents", "AggressiveTradingAgent"),
        "BalancedTradingAgent": check_import("src.trading_agents", "BalancedTradingAgent"),
        "FractalAnalysisAgent": check_import("src.trading_agents", "FractalAnalysisAgent"),
        "CandleRangeTheoryAgent": check_import("src.trading_agents", "CandleRangeTheoryAgent"),
        "QuantitativePatternAgent": check_import("src.trading_agents", "QuantitativePatternAgent"),
        "ForexSpecialistAgent": check_import("forex_crypto_agents", "ForexSpecialistAgent"),
        "CryptoSpecialistAgent": check_import("forex_crypto_agents", "CryptoSpecialistAgent"),
        "MultiAssetArbitrageAgent": check_import("forex_crypto_agents", "MultiAssetArbitrageAgent"),
    }
    
    return agent_classes

def check_system_config() -> bool:
    """Check if system configuration can be loaded."""
    try:
        from config.settings import SystemConfig, AgentConfig, AgentType
        config = SystemConfig()
        logger.info(f"System config loaded: {len(config.agent_configs)} agents")
        return True
    except Exception as e:
        logger.error(f"System config error: {e}")
        return False

def main():
    """Run comprehensive system check."""
    logger.info("🔍 Starting comprehensive system check...")
    
    # Check critical files
    logger.info("\n📁 Checking critical files...")
    file_results = check_critical_files()
    file_failures = [f for f, exists in file_results.items() if not exists]
    
    if file_failures:
        logger.error(f"❌ Missing files: {file_failures}")
    else:
        logger.info("✅ All critical files present")
    
    # Check critical imports
    logger.info("\n📦 Checking critical imports...")
    import_results = check_critical_imports()
    import_failures = [f for f, works in import_results.items() if not works]
    
    if import_failures:
        logger.error(f"❌ Import failures: {import_failures}")
    else:
        logger.info("✅ All critical imports working")
    
    # Check agent classes
    logger.info("\n🤖 Checking agent classes...")
    agent_results = check_agent_classes()
    agent_failures = [f for f, works in agent_results.items() if not works]
    
    if agent_failures:
        logger.error(f"❌ Agent class failures: {agent_failures}")
    else:
        logger.info("✅ All agent classes working")
    
    # Check system configuration
    logger.info("\n⚙️ Checking system configuration...")
    config_works = check_system_config()
    if not config_works:
        logger.error("❌ System configuration failed")
    else:
        logger.info("✅ System configuration working")
    
    # Summary
    logger.info("\n📋 SYSTEM CHECK SUMMARY:")
    total_checks = len(file_results) + len(import_results) + len(agent_results) + 1
    total_failures = len(file_failures) + len(import_failures) + len(agent_failures) + (0 if config_works else 1)
    total_successes = total_checks - total_failures
    
    logger.info(f"  Total checks: {total_checks}")
    logger.info(f"  ✅ Successes: {total_successes}")
    logger.info(f"  ❌ Failures: {total_failures}")
    
    if total_failures == 0:
        logger.info("🎉 SYSTEM IS READY FOR DEPLOYMENT!")
        return True
    else:
        logger.error("⚠️ SYSTEM HAS ISSUES THAT NEED FIXING")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
