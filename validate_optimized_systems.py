#!/usr/bin/env python3
"""
Comprehensive Validation Script for Optimized Trading Systems
============================================================
Validates both optimized trading systems before deployment.
"""

import asyncio
import logging
import sys
import os
import subprocess
import time
from datetime import datetime
from typing import Dict, List, Optional, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/validation.log', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger("SystemValidation")

class OptimizedSystemValidator:
    """Comprehensive validator for optimized trading systems"""
    
    def __init__(self):
        self.logger = logger
        self.validation_results = {
            'smart_trading': {'passed': 0, 'failed': 0, 'issues': []},
            'ultra_aggressive': {'passed': 0, 'failed': 0, 'issues': []},
            'deployment_scripts': {'passed': 0, 'failed': 0, 'issues': []},
            'dependencies': {'passed': 0, 'failed': 0, 'issues': []}
        }
    
    def validate_python_syntax(self, file_path: str) -> bool:
        """Validate Python syntax for a file"""
        try:
            result = subprocess.run([sys.executable, '-m', 'py_compile', file_path], 
                                  capture_output=True, text=True)
            if result.returncode == 0:
                self.logger.info(f"✅ Syntax validation passed for {file_path}")
                return True
            else:
                self.logger.error(f"❌ Syntax validation failed for {file_path}: {result.stderr}")
                return False
        except Exception as e:
            self.logger.error(f"❌ Syntax validation error for {file_path}: {e}")
            return False
    
    def validate_imports(self, file_path: str) -> bool:
        """Validate imports for a file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Check for required imports
            required_imports = [
                'import asyncio',
                'import alpaca_trade_api',
                'import logging',
                'import random',
                'import sys',
                'import time',
                'from datetime import datetime',
                'from typing import Dict, List, Optional, Any'
            ]
            
            missing_imports = []
            for imp in required_imports:
                if imp not in content:
                    missing_imports.append(imp)
            
            if missing_imports:
                self.logger.error(f"❌ Missing imports in {file_path}: {missing_imports}")
                return False
            else:
                self.logger.info(f"✅ Import validation passed for {file_path}")
                return True
                
        except Exception as e:
            self.logger.error(f"❌ Import validation error for {file_path}: {e}")
            return False
    
    def validate_trading_logic(self, file_path: str) -> bool:
        """Validate trading logic structure"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Check for required methods
            required_methods = [
                'get_account_info',
                'get_positions', 
                'get_market_data',
                'execute_trade',
                'run_cycle',
                'run_system'
            ]
            
            missing_methods = []
            for method in required_methods:
                if f'def {method}' not in content:
                    missing_methods.append(method)
            
            if missing_methods:
                self.logger.error(f"❌ Missing methods in {file_path}: {missing_methods}")
                return False
            
            # Check for error handling
            error_handling_patterns = [
                'try:',
                'except Exception as e:',
                'self.logger.error',
                'self.logger.warning'
            ]
            
            missing_error_handling = []
            for pattern in error_handling_patterns:
                if pattern not in content:
                    missing_error_handling.append(pattern)
            
            if missing_error_handling:
                self.logger.warning(f"⚠️ Missing error handling patterns in {file_path}: {missing_error_handling}")
            
            self.logger.info(f"✅ Trading logic validation passed for {file_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Trading logic validation error for {file_path}: {e}")
            return False
    
    def validate_deployment_scripts(self) -> bool:
        """Validate deployment batch scripts"""
        scripts = [
            'deploy_optimized_smart.bat',
            'deploy_optimized_ultra.bat'
        ]
        
        all_valid = True
        for script in scripts:
            if not os.path.exists(script):
                self.logger.error(f"❌ Deployment script missing: {script}")
                all_valid = False
                continue
            
            try:
                with open(script, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Check for required commands
                required_commands = [
                    'docker stop',
                    'docker rm',
                    'docker build',
                    'docker run',
                    'timeout',
                    'docker logs'
                ]
                
                missing_commands = []
                for cmd in required_commands:
                    if cmd not in content:
                        missing_commands.append(cmd)
                
                if missing_commands:
                    self.logger.error(f"❌ Missing commands in {script}: {missing_commands}")
                    all_valid = False
                else:
                    self.logger.info(f"✅ Deployment script validation passed for {script}")
                    
            except Exception as e:
                self.logger.error(f"❌ Deployment script validation error for {script}: {e}")
                all_valid = False
        
        return all_valid
    
    def validate_dependencies(self) -> bool:
        """Validate Python dependencies"""
        try:
            # Test critical imports
            critical_imports = [
                'alpaca_trade_api',
                'yfinance',
                'asyncio',
                'logging',
                'random',
                'sys',
                'time',
                'datetime',
                'typing'
            ]
            
            missing_imports = []
            for imp in critical_imports:
                try:
                    __import__(imp)
                except ImportError:
                    missing_imports.append(imp)
            
            if missing_imports:
                self.logger.error(f"❌ Missing dependencies: {missing_imports}")
                return False
            else:
                self.logger.info("✅ All critical dependencies available")
                return True
                
        except Exception as e:
            self.logger.error(f"❌ Dependency validation error: {e}")
            return False
    
    def validate_docker_setup(self) -> bool:
        """Validate Docker setup"""
        try:
            # Check if Docker is running
            result = subprocess.run(['docker', '--version'], capture_output=True, text=True)
            if result.returncode != 0:
                self.logger.error("❌ Docker not available")
                return False
            
            # Check if Docker Compose is available
            result = subprocess.run(['docker-compose', '--version'], capture_output=True, text=True)
            if result.returncode != 0:
                self.logger.warning("⚠️ Docker Compose not available, but Docker is working")
            
            self.logger.info("✅ Docker setup validation passed")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Docker setup validation error: {e}")
            return False
    
    def validate_network_connectivity(self) -> bool:
        """Validate network connectivity to required services"""
        try:
            # Test Alpaca API connectivity
            import alpaca_trade_api as tradeapi
            api = tradeapi.REST(
                key_id="PKK43GTIACJNUPGZPCPF",
                secret_key="CuQWde4QtPHAtwuMfxaQhB8njmrcJDq0YK4Oz9Rw",
                base_url="https://paper-api.alpaca.markets",
                api_version='v2'
            )
            
            # Test connection
            account = api.get_account()
            self.logger.info(f"✅ Alpaca API connectivity validated - Account: {account.id}")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Alpaca API connectivity failed: {e}")
            return False
    
    async def run_comprehensive_validation(self):
        """Run comprehensive validation of all systems"""
        self.logger.info("=" * 60)
        self.logger.info("COMPREHENSIVE SYSTEM VALIDATION STARTING")
        self.logger.info("=" * 60)
        
        # Create logs directory
        os.makedirs('logs', exist_ok=True)
        
        # Validate Python files
        python_files = [
            'run_optimized_smart_trading.py',
            'run_optimized_ultra_aggressive.py'
        ]
        
        for file_path in python_files:
            self.logger.info(f"Validating {file_path}...")
            
            # Syntax validation
            if self.validate_python_syntax(file_path):
                self.validation_results['smart_trading' if 'smart' in file_path else 'ultra_aggressive']['passed'] += 1
            else:
                self.validation_results['smart_trading' if 'smart' in file_path else 'ultra_aggressive']['failed'] += 1
                self.validation_results['smart_trading' if 'smart' in file_path else 'ultra_aggressive']['issues'].append(f"Syntax error in {file_path}")
            
            # Import validation
            if self.validate_imports(file_path):
                self.validation_results['smart_trading' if 'smart' in file_path else 'ultra_aggressive']['passed'] += 1
            else:
                self.validation_results['smart_trading' if 'smart' in file_path else 'ultra_aggressive']['failed'] += 1
                self.validation_results['smart_trading' if 'smart' in file_path else 'ultra_aggressive']['issues'].append(f"Import error in {file_path}")
            
            # Trading logic validation
            if self.validate_trading_logic(file_path):
                self.validation_results['smart_trading' if 'smart' in file_path else 'ultra_aggressive']['passed'] += 1
            else:
                self.validation_results['smart_trading' if 'smart' in file_path else 'ultra_aggressive']['failed'] += 1
                self.validation_results['smart_trading' if 'smart' in file_path else 'ultra_aggressive']['issues'].append(f"Trading logic error in {file_path}")
        
        # Validate deployment scripts
        self.logger.info("Validating deployment scripts...")
        if self.validate_deployment_scripts():
            self.validation_results['deployment_scripts']['passed'] += 1
        else:
            self.validation_results['deployment_scripts']['failed'] += 1
            self.validation_results['deployment_scripts']['issues'].append("Deployment script validation failed")
        
        # Validate dependencies
        self.logger.info("Validating dependencies...")
        if self.validate_dependencies():
            self.validation_results['dependencies']['passed'] += 1
        else:
            self.validation_results['dependencies']['failed'] += 1
            self.validation_results['dependencies']['issues'].append("Dependency validation failed")
        
        # Validate Docker setup
        self.logger.info("Validating Docker setup...")
        if self.validate_docker_setup():
            self.validation_results['dependencies']['passed'] += 1
        else:
            self.validation_results['dependencies']['failed'] += 1
            self.validation_results['dependencies']['issues'].append("Docker setup validation failed")
        
        # Validate network connectivity
        self.logger.info("Validating network connectivity...")
        if self.validate_network_connectivity():
            self.validation_results['dependencies']['passed'] += 1
        else:
            self.validation_results['dependencies']['failed'] += 1
            self.validation_results['dependencies']['issues'].append("Network connectivity validation failed")
        
        # Generate validation report
        self.generate_validation_report()
    
    def generate_validation_report(self):
        """Generate comprehensive validation report"""
        self.logger.info("=" * 60)
        self.logger.info("VALIDATION REPORT")
        self.logger.info("=" * 60)
        
        total_passed = 0
        total_failed = 0
        
        for category, results in self.validation_results.items():
            passed = results['passed']
            failed = results['failed']
            total_passed += passed
            total_failed += failed
            
            self.logger.info(f"{category.upper()}:")
            self.logger.info(f"  ✅ Passed: {passed}")
            self.logger.info(f"  ❌ Failed: {failed}")
            
            if results['issues']:
                self.logger.info(f"  Issues:")
                for issue in results['issues']:
                    self.logger.info(f"    - {issue}")
            self.logger.info("")
        
        self.logger.info(f"TOTAL: ✅ {total_passed} passed, ❌ {total_failed} failed")
        
        if total_failed == 0:
            self.logger.info("🎉 ALL VALIDATIONS PASSED! Systems are ready for deployment!")
        else:
            self.logger.warning(f"⚠️ {total_failed} validations failed. Please fix issues before deployment.")
        
        # Save report to file
        with open('logs/validation_report.json', 'w', encoding='utf-8') as f:
            import json
            json.dump(self.validation_results, f, indent=2, default=str)
        
        self.logger.info("Validation report saved to logs/validation_report.json")

async def main():
    """Main validation function"""
    validator = OptimizedSystemValidator()
    await validator.run_comprehensive_validation()

if __name__ == "__main__":
    asyncio.run(main())

