"""
Comprehensive Validation System
==============================

This module provides bulletproof validation for all system components.
"""

import asyncio
import logging
import os
import sys
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import json
import psycopg2
import redis

class ValidationSystem:
    """Comprehensive validation system for the trading system"""
    
    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self.validation_results = {}
        
    async def validate_entire_system(self) -> Dict[str, Any]:
        """Validate the entire trading system"""
        self.logger.info("🔍 Starting comprehensive system validation...")
        
        try:
            # 1. Validate environment
            env_validation = await self.validate_environment()
            
            # 2. Validate dependencies
            deps_validation = await self.validate_dependencies()
            
            # 3. Validate database connections
            db_validation = await self.validate_databases()
            
            # 4. Validate trading components
            trading_validation = await self.validate_trading_components()
            
            # 5. Validate AI systems
            ai_validation = await self.validate_ai_systems()
            
            # 6. Validate configuration
            config_validation = await self.validate_configuration()
            
            # Compile results
            self.validation_results = {
                "timestamp": datetime.now().isoformat(),
                "environment": env_validation,
                "dependencies": deps_validation,
                "databases": db_validation,
                "trading": trading_validation,
                "ai_systems": ai_validation,
                "configuration": config_validation,
                "overall_status": self._calculate_overall_status()
            }
            
            return self.validation_results
            
        except Exception as e:
            self.logger.error(f"Validation failed: {e}")
            return {"error": str(e), "overall_status": "FAILED"}
    
    async def validate_environment(self) -> Dict[str, Any]:
        """Validate environment variables and system environment"""
        results = {"passed": [], "failed": [], "warnings": []}
        
        # Check Python version
        if sys.version_info >= (3, 8):
            results["passed"].append("Python version >= 3.8")
        else:
            results["failed"].append(f"Python version {sys.version_info} < 3.8")
        
        # Check required environment variables
        required_vars = {
            "ALPACA_API_KEY": "Alpaca API key",
            "ALPACA_SECRET_KEY": "Alpaca secret key"
        }
        
        for var, description in required_vars.items():
            if os.getenv(var):
                results["passed"].append(f"{description} configured")
            else:
                results["warnings"].append(f"{description} not configured")
        
        # Check optional environment variables
        optional_vars = {
            "ALPACA_BASE_URL": "Alpaca base URL",
            "LOG_LEVEL": "Log level",
            "DATABASE_URL": "Database URL"
        }
        
        for var, description in optional_vars.items():
            if os.getenv(var):
                results["passed"].append(f"{description} configured")
            else:
                results["warnings"].append(f"{description} using default")
        
        return results
    
    async def validate_dependencies(self) -> Dict[str, Any]:
        """Validate Python dependencies"""
        results = {"passed": [], "failed": [], "warnings": []}
        
        required_packages = [
            "asyncio", "logging", "datetime", "json", "numpy", "pandas",
            "psycopg2", "redis", "docker", "requests", "aiohttp"
        ]
        
        for package in required_packages:
            try:
                __import__(package)
                results["passed"].append(f"Package {package} available")
            except ImportError:
                results["failed"].append(f"Package {package} not available")
        
        # Check specific trading packages
        trading_packages = [
            "alpaca_trade_api", "yfinance", "ta", "scikit-learn"
        ]
        
        for package in trading_packages:
            try:
                __import__(package)
                results["passed"].append(f"Trading package {package} available")
            except ImportError:
                results["warnings"].append(f"Trading package {package} not available")
        
        return results
    
    async def validate_databases(self) -> Dict[str, Any]:
        """Validate database connections"""
        results = {"passed": [], "failed": [], "warnings": []}
        
        # Validate PostgreSQL
        try:
            conn = psycopg2.connect(
                host="localhost",
                port=5432,
                database="trading_agents",
                user="trading_user",
                password="trading_password"
            )
            
            # Test basic query
            with conn.cursor() as cur:
                cur.execute("SELECT 1")
                cur.fetchone()
            
            conn.close()
            results["passed"].append("PostgreSQL connection successful")
            
        except Exception as e:
            results["failed"].append(f"PostgreSQL connection failed: {e}")
        
        # Validate Redis
        try:
            r = redis.Redis(host='localhost', port=6379, decode_responses=True)
            r.ping()
            results["passed"].append("Redis connection successful")
        except Exception as e:
            results["failed"].append(f"Redis connection failed: {e}")
        
        return results
    
    async def validate_trading_components(self) -> Dict[str, Any]:
        """Validate trading system components"""
        results = {"passed": [], "failed": [], "warnings": []}
        
        # Check if core modules can be imported
        core_modules = [
            "src.system_orchestrator",
            "src.base_agent",
            "src.real_alpaca_integration",
            "src.advanced_risk_manager",
            "src.data_sources"
        ]
        
        for module in core_modules:
            try:
                __import__(module)
                results["passed"].append(f"Module {module} importable")
            except Exception as e:
                results["failed"].append(f"Module {module} import failed: {e}")
        
        # Check if main entry point exists
        if os.path.exists("main.py"):
            results["passed"].append("Main entry point exists")
        else:
            results["failed"].append("Main entry point missing")
        
        # Check if logs directory exists
        if os.path.exists("logs"):
            results["passed"].append("Logs directory exists")
        else:
            results["warnings"].append("Logs directory missing")
        
        return results
    
    async def validate_ai_systems(self) -> Dict[str, Any]:
        """Validate AI systems"""
        results = {"passed": [], "failed": [], "warnings": []}
        
        # Check AI modules
        ai_modules = [
            "src.ai_enhanced_agent",
            "src.free_intelligence_system",
            "src.hybrid_ai_system",
            "src.perplexity_intelligence"
        ]
        
        for module in ai_modules:
            try:
                __import__(module)
                results["passed"].append(f"AI module {module} importable")
            except Exception as e:
                results["warnings"].append(f"AI module {module} import failed: {e}")
        
        # Check if free intelligence system works
        try:
            from src.free_intelligence_system import FreeIntelligenceSystem
            results["passed"].append("Free intelligence system available")
        except Exception as e:
            results["warnings"].append(f"Free intelligence system failed: {e}")
        
        return results
    
    async def validate_configuration(self) -> Dict[str, Any]:
        """Validate system configuration"""
        results = {"passed": [], "failed": [], "warnings": []}
        
        # Check if config files exist
        config_files = [
            "config/settings.py",
            "requirements.txt",
            "docker-compose.yml"
        ]
        
        for file in config_files:
            if os.path.exists(file):
                results["passed"].append(f"Config file {file} exists")
            else:
                results["warnings"].append(f"Config file {file} missing")
        
        # Check if optimized config exists
        if os.path.exists("optimized_config.json"):
            results["passed"].append("Optimized configuration available")
        else:
            results["warnings"].append("Optimized configuration missing")
        
        return results
    
    def _calculate_overall_status(self) -> str:
        """Calculate overall validation status"""
        if not self.validation_results:
            return "UNKNOWN"
        
        total_failed = 0
        total_warnings = 0
        
        for category, results in self.validation_results.items():
            if isinstance(results, dict) and "failed" in results:
                total_failed += len(results["failed"])
                total_warnings += len(results["warnings"])
        
        if total_failed == 0:
            if total_warnings == 0:
                return "EXCELLENT"
            elif total_warnings <= 3:
                return "GOOD"
            else:
                return "WARNING"
        elif total_failed <= 2:
            return "NEEDS_ATTENTION"
        else:
            return "CRITICAL"
    
    def generate_validation_report(self) -> str:
        """Generate human-readable validation report"""
        if not self.validation_results:
            return "No validation results available"
        
        report = []
        report.append("=" * 60)
        report.append("SYSTEM VALIDATION REPORT")
        report.append("=" * 60)
        report.append(f"Timestamp: {self.validation_results.get('timestamp', 'Unknown')}")
        report.append(f"Overall Status: {self.validation_results.get('overall_status', 'Unknown')}")
        report.append("")
        
        for category, results in self.validation_results.items():
            if isinstance(results, dict) and "passed" in results:
                report.append(f"{category.upper()}:")
                report.append(f"  ✅ Passed: {len(results['passed'])}")
                report.append(f"  ❌ Failed: {len(results['failed'])}")
                report.append(f"  ⚠️  Warnings: {len(results['warnings'])}")
                
                if results['failed']:
                    report.append("  Failed items:")
                    for item in results['failed']:
                        report.append(f"    - {item}")
                
                if results['warnings']:
                    report.append("  Warning items:")
                    for item in results['warnings']:
                        report.append(f"    - {item}")
                
                report.append("")
        
        return "\n".join(report)

async def main():
    """Main validation function"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    logger = logging.getLogger("ValidationSystem")
    validator = ValidationSystem(logger)
    
    results = await validator.validate_entire_system()
    
    # Save results
    with open('validation_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    # Print report
    print(validator.generate_validation_report())
    
    return results

if __name__ == "__main__":
    asyncio.run(main())

