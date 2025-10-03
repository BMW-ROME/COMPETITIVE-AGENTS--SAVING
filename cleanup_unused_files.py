#!/usr/bin/env python3
"""
Cleanup Unused Files - Remove debugging and temporary files
"""
import os
import shutil
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def cleanup_unused_files():
    """Remove unused debugging and temporary files"""
    
    # Files to remove (debugging, temporary, duplicate)
    files_to_remove = [
        # Debugging scripts
        "diagnose_and_fix_trading_issues.py",
        "diagnose_data_issue.py", 
        "fix_agent_decisions.py",
        "fix_agent_decision_logic.py",
        "fix_agent_opportunities.py",
        "fix_data_flow.py",
        "fix_orchestrator_data_flow.py",
        "force_agent_decisions.py",
        "test_agent_decisions.py",
        "final_agent_fix.py",
        "agent_patch.py",
        "quick_data_fix.py",
        "simple_diagnostic.py",
        "test_imports.py",
        "test_paper_config.py",
        "test_real_alpaca.py",
        "test_real_trading.py",
        "test_trading_setup.py",
        "test_mcp_simple.py",
        "test_mcp_system.py",
        "check_mcp_status.py",
        "fix_mcp_deployment.py",
        "fix_simple_mcp.py",
        "restart_mcp_system.py",
        "setup_mcp_system.py",
        "mcp_dashboard.py",
        "mcp_dashboard_simple.py",
        "run_ultimate_system_with_mcp.py",
        "run_ultimate_system_simple_mcp.py",
        "run_ultimate_system.py",
        "run_advanced_system.py",
        "run_paper_trading.py",
        "run_trading_system_no_mcp.py",
        "start_paper_trading.py",
        "quick_paper_test.py",
        "simple_trading_test.py",
        "multi_asset_demo.py",
        "multi_exchange_demo.py",
        "forex_crypto_agents.py",
        "advanced_agents_demo.py",
        "one_click_cloud_deploy.py",
        "setup_env_guide.py",
        "setup_real_alpaca_trading.py",
        "test_alpaca_connection.py",
        "fix_dependencies.py",
        "fix_docker_build.py",
        "fix_market_hours_logic.py",
        "fix_dashboard.py",
        "market_hours_checker.py",
        "initialize_system_data.py",
        "activate_trading.py",
        "ultimate_trading_optimizer.py",
        "simple_optimizer.py",
        "quick_restart.py",
        "quick_restart_windows.py",
        "python continuous_paper_trading.py.txt",
        "=0.23.2",  # Weird file
        "example_run.py",
        "python_commands_guide.py",
        "live_trading_setup.py",
        "start-docker-trading.bat",
        "start-docker-trading.sh",
        "deploy_to_cloud.sh",
        "cloud_setup.sh",
        "quick_data_fix.py",
        "fix_all_float_get_errors.py",
        "install_advanced_deps.py",
        "advanced_monitoring_dashboard.py",
        "multi_asset_demo.py",
        "multi_exchange_demo.py",
        "forex_crypto_agents.py",
        "advanced_agents_demo.py",
        "one_click_cloud_deploy.py",
        "setup_env_guide.py",
        "setup_real_alpaca_trading.py",
        "test_alpaca_connection.py",
        "fix_dependencies.py",
        "fix_docker_build.py",
        "fix_market_hours_logic.py",
        "fix_dashboard.py",
        "market_hours_checker.py",
        "initialize_system_data.py",
        "activate_trading.py",
        "ultimate_trading_optimizer.py",
        "simple_optimizer.py",
        "quick_restart.py",
        "quick_restart_windows.py",
        "python continuous_paper_trading.py.txt",
        "=0.23.2",  # Weird file
        "example_run.py",
        "python_commands_guide.py",
        "live_trading_setup.py",
        "start-docker-trading.bat",
        "start-docker-trading.sh",
        "deploy_to_cloud.sh",
        "cloud_setup.sh"
    ]
    
    # Directories to remove
    dirs_to_remove = [
        "mcp_dashboard",
        "mcp_models", 
        "ml_models",
        "tools",
        "tests",
        "venv"
    ]
    
    # Files to keep (essential)
    essential_files = [
        "docker-compose.yml",
        "Dockerfile", 
        "requirements.txt",
        "continuous_real_alpaca_trading.py",
        "continuous_paper_trading.py",
        "continuous_live_trading.py",
        "continuous_live_trading_windows.py",
        "continuous_paper_trading_windows.py",
        "monitoring_dashboard.py",
        "src/",
        "config/",
        "data/",
        "logs/",
        "templates/",
        "README.md",
        "ULTIMATE_SYSTEM_README.md",
        "DOCKER_SETUP_COMPLETE.md",
        "ALPACA_SETUP_GUIDE.md",
        "CLOUD_DEPLOYMENT_GUIDE.md",
        "MCP_INTEGRATION_GUIDE.md",
        "WINDOWS_FIXES.md",
        "QUICK_COMMANDS.md",
        "docker-commands.md",
        "REAL_ALPACA_SOLUTION.md",
        ".env",
        "init.sql"
    ]
    
    removed_count = 0
    kept_count = 0
    
    logger.info("🧹 Starting cleanup of unused files...")
    
    # Remove files
    for file_path in files_to_remove:
        if os.path.exists(file_path):
            try:
                if os.path.isfile(file_path):
                    os.remove(file_path)
                    logger.info(f"🗑️ Removed file: {file_path}")
                    removed_count += 1
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
                    logger.info(f"🗑️ Removed directory: {file_path}")
                    removed_count += 1
            except Exception as e:
                logger.warning(f"⚠️ Could not remove {file_path}: {e}")
        else:
            logger.info(f"✅ File not found (already removed): {file_path}")
    
    # Remove directories
    for dir_path in dirs_to_remove:
        if os.path.exists(dir_path):
            try:
                shutil.rmtree(dir_path)
                logger.info(f"🗑️ Removed directory: {dir_path}")
                removed_count += 1
            except Exception as e:
                logger.warning(f"⚠️ Could not remove {dir_path}: {e}")
        else:
            logger.info(f"✅ Directory not found (already removed): {dir_path}")
    
    # Clean up __pycache__ directories
    for root, dirs, files in os.walk("."):
        for dir_name in dirs:
            if dir_name == "__pycache__":
                pycache_path = os.path.join(root, dir_name)
                try:
                    shutil.rmtree(pycache_path)
                    logger.info(f"🗑️ Removed __pycache__: {pycache_path}")
                    removed_count += 1
                except Exception as e:
                    logger.warning(f"⚠️ Could not remove {pycache_path}: {e}")
    
    logger.info(f"📊 CLEANUP SUMMARY:")
    logger.info(f"  Files/Directories removed: {removed_count}")
    logger.info(f"  Essential files kept: {len(essential_files)}")
    logger.info(f"  Repository cleaned successfully! 🎉")
    
    return removed_count

if __name__ == "__main__":
    cleanup_unused_files()
