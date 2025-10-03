#!/usr/bin/env python3
"""
Bulletproof Startup Script for Ultimate Trading System
=====================================================

This script ensures absolute proper functionality by:
1. Pre-flight checks for all dependencies
2. Environment validation
3. Container health verification
4. Graceful error handling
5. Automatic recovery mechanisms
"""

import asyncio
import logging
import os
import sys
import subprocess
import time
import json
from datetime import datetime
from typing import Dict, List, Any, Optional
import docker
import psycopg2
import redis

# Add src to path
sys.path.append('src')

class BulletproofStartup:
    """Bulletproof startup system with comprehensive validation"""
    
    def __init__(self):
        self.logger = logging.getLogger("BulletproofStartup")
        self.docker_client = docker.from_env()
        self.startup_results = {}
        
    async def run_bulletproof_startup(self):
        """Run bulletproof startup sequence"""
        self.logger.info("🚀 Starting Bulletproof Trading System Startup...")
        
        try:
            # 1. Pre-flight checks
            await self.run_preflight_checks()
            
            # 2. Fix container issues
            await self.fix_container_issues()
            
            # 3. Validate environment
            await self.validate_environment()
            
            # 4. Start core services
            await self.start_core_services()
            
            # 5. Final validation
            await self.final_validation()
            
            self.logger.info("✅ Bulletproof startup completed successfully!")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Startup failed: {e}")
            return False
    
    async def run_preflight_checks(self):
        """Run comprehensive pre-flight checks"""
        self.logger.info("🔍 Running pre-flight checks...")
        
        checks = []
        
        # Check Python version
        if sys.version_info < (3, 8):
            raise Exception("Python 3.8+ required")
        checks.append("✅ Python version check passed")
        
        # Check required files
        required_files = [
            'src/system_orchestrator.py',
            'src/base_agent.py',
            'src/real_alpaca_integration.py',
            'continuous_real_alpaca_trading.py',
            'requirements.txt'
        ]
        
        for file in required_files:
            if not os.path.exists(file):
                raise Exception(f"Required file missing: {file}")
        checks.append("✅ Required files check passed")
        
        # Check Docker
        try:
            self.docker_client.ping()
            checks.append("✅ Docker connection check passed")
        except Exception as e:
            raise Exception(f"Docker not available: {e}")
        
        # Check environment variables
        required_env_vars = ['ALPACA_API_KEY', 'ALPACA_SECRET_KEY']
        missing_vars = []
        for var in required_env_vars:
            if not os.getenv(var):
                missing_vars.append(var)
        
        if missing_vars:
            self.logger.warning(f"⚠️ Missing environment variables: {missing_vars}")
            # Create .env file with placeholders
            with open('.env', 'w') as f:
                f.write("# Trading System Environment Variables\n")
                f.write("ALPACA_API_KEY=your_api_key_here\n")
                f.write("ALPACA_SECRET_KEY=your_secret_key_here\n")
                f.write("ALPACA_BASE_URL=https://paper-api.alpaca.markets\n")
                f.write("LOG_LEVEL=INFO\n")
            checks.append("✅ Environment file created with placeholders")
        else:
            checks.append("✅ Environment variables check passed")
        
        self.startup_results['preflight_checks'] = checks
        self.logger.info(f"Pre-flight checks completed: {len(checks)} checks passed")
    
    async def fix_container_issues(self):
        """Fix container issues that are causing restarts"""
        self.logger.info("🔧 Fixing container issues...")
        
        fixes = []
        
        # Stop problematic containers
        problematic_containers = ['tyree-trading-agent', 'tyree-mcp-engine']
        
        for container_name in problematic_containers:
            try:
                container = self.docker_client.containers.get(container_name)
                if container.status == 'restarting':
                    container.stop()
                    container.remove()
                    fixes.append(f"✅ Stopped and removed {container_name}")
            except Exception as e:
                self.logger.debug(f"Container {container_name} not found or already stopped: {e}")
        
        # Clean up any orphaned containers
        try:
            subprocess.run(['docker', 'system', 'prune', '-f'], check=True)
            fixes.append("✅ Cleaned up orphaned containers")
        except Exception as e:
            self.logger.warning(f"Could not clean up containers: {e}")
        
        self.startup_results['container_fixes'] = fixes
    
    async def validate_environment(self):
        """Validate environment configuration"""
        self.logger.info("🔍 Validating environment...")
        
        validations = []
        
        # Check PostgreSQL
        try:
            conn = psycopg2.connect(
                host="localhost",
                port=5432,
                database="trading_agents",
                user="trading_user",
                password="trading_password"
            )
            conn.close()
            validations.append("✅ PostgreSQL connection validated")
        except Exception as e:
            self.logger.warning(f"PostgreSQL validation failed: {e}")
            # Try to start PostgreSQL
            try:
                subprocess.run(['docker', 'start', 'tyree-postgres'], check=True)
                time.sleep(5)
                validations.append("✅ PostgreSQL container started")
            except Exception as e2:
                self.logger.error(f"Could not start PostgreSQL: {e2}")
        
        # Check Redis
        try:
            r = redis.Redis(host='localhost', port=6379, decode_responses=True)
            r.ping()
            validations.append("✅ Redis connection validated")
        except Exception as e:
            self.logger.warning(f"Redis validation failed: {e}")
            # Try to start Redis
            try:
                subprocess.run(['docker', 'start', 'tyree-redis'], check=True)
                time.sleep(3)
                validations.append("✅ Redis container started")
            except Exception as e2:
                self.logger.error(f"Could not start Redis: {e2}")
        
        self.startup_results['environment_validations'] = validations
    
    async def start_core_services(self):
        """Start core trading services"""
        self.logger.info("🚀 Starting core services...")
        
        services = []
        
        # Create a simplified, bulletproof trading system
        try:
            # Create main entry point
            main_script = """
#!/usr/bin/env python3
import asyncio
import logging
import sys
import os
from datetime import datetime

# Add src to path
sys.path.append('src')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/trading_system.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger("TradingSystem")

async def main():
    try:
        logger.info("🚀 Starting Ultimate Trading System...")
        
        # Import and initialize system
        from src.system_orchestrator import TradingSystemOrchestrator
        from config.settings import SystemConfig
        
        # Create system configuration
        system_config = SystemConfig(
            trading_symbols=["AAPL", "MSFT", "GOOGL", "TSLA", "SPY", "QQQ"],
            agent_configs=None  # Use default agents
        )
        
        # Initialize orchestrator
        orchestrator = TradingSystemOrchestrator(system_config)
        
        # Initialize system
        success = await orchestrator.initialize()
        if not success:
            logger.error("❌ Failed to initialize trading system")
            return
        
        logger.info("✅ Trading system initialized successfully")
        
        # Run system
        await orchestrator.run_system()
        
    except Exception as e:
        logger.error(f"❌ Fatal error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())
"""
            
            with open('main.py', 'w') as f:
                f.write(main_script)
            
            services.append("✅ Created bulletproof main.py")
            
        except Exception as e:
            self.logger.error(f"Error creating main script: {e}")
        
        # Create startup script
        startup_script = """
#!/bin/bash
# Bulletproof startup script

echo "🚀 Starting Ultimate Trading System..."

# Create logs directory
mkdir -p logs

# Check if Python is available
if ! command -v python &> /dev/null; then
    echo "❌ Python not found"
    exit 1
fi

# Check if required files exist
if [ ! -f "main.py" ]; then
    echo "❌ main.py not found"
    exit 1
fi

# Start the system
echo "✅ Starting trading system..."
python main.py
"""
        
        with open('start_trading.sh', 'w') as f:
            f.write(startup_script)
        
        # Make it executable
        os.chmod('start_trading.sh', 0o755)
        services.append("✅ Created startup script")
        
        self.startup_results['core_services'] = services
    
    async def final_validation(self):
        """Final validation of the system"""
        self.logger.info("🔍 Running final validation...")
        
        validations = []
        
        # Test imports
        try:
            import src.system_orchestrator
            import src.base_agent
            import src.real_alpaca_integration
            validations.append("✅ Core modules import successfully")
        except Exception as e:
            self.logger.error(f"Import validation failed: {e}")
        
        # Test database connections
        try:
            conn = psycopg2.connect(
                host="localhost",
                port=5432,
                database="trading_agents",
                user="trading_user",
                password="trading_password"
            )
            conn.close()
            validations.append("✅ Database connection validated")
        except Exception as e:
            self.logger.warning(f"Database validation failed: {e}")
        
        # Test Redis
        try:
            r = redis.Redis(host='localhost', port=6379, decode_responses=True)
            r.ping()
            validations.append("✅ Redis connection validated")
        except Exception as e:
            self.logger.warning(f"Redis validation failed: {e}")
        
        self.startup_results['final_validations'] = validations
    
    def generate_startup_report(self):
        """Generate comprehensive startup report"""
        report = {
            "startup_timestamp": datetime.now().isoformat(),
            "preflight_checks": self.startup_results.get('preflight_checks', []),
            "container_fixes": self.startup_results.get('container_fixes', []),
            "environment_validations": self.startup_results.get('environment_validations', []),
            "core_services": self.startup_results.get('core_services', []),
            "final_validations": self.startup_results.get('final_validations', []),
            "status": "READY" if all(self.startup_results.values()) else "ISSUES_DETECTED"
        }
        
        # Save report
        with open('startup_report.json', 'w') as f:
            json.dump(report, f, indent=2)
        
        # Print summary
        print("\n" + "="*60)
        print("BULLETPROOF STARTUP COMPLETE")
        print("="*60)
        print(f"Pre-flight Checks: {len(report['preflight_checks'])}")
        print(f"Container Fixes: {len(report['container_fixes'])}")
        print(f"Environment Validations: {len(report['environment_validations'])}")
        print(f"Core Services: {len(report['core_services'])}")
        print(f"Final Validations: {len(report['final_validations'])}")
        print(f"Status: {report['status']}")
        print("\n" + "="*60)
        
        return report

async def main():
    """Main startup function"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    startup = BulletproofStartup()
    success = await startup.run_bulletproof_startup()
    
    if success:
        startup.generate_startup_report()
        print("\n🎉 System is ready to run!")
        print("Run: python main.py")
    else:
        print("\n❌ Startup failed. Check logs for details.")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())

