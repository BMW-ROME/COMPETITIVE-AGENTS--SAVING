#!/usr/bin/env python3
"""
Advanced Trading Agents Demo - Showcasing the new sophisticated trading agents.
This demonstrates the Fractal Analysis, Candle Range Theory, and Quantitative Pattern agents.
"""

import asyncio
import logging
import sys
import os
from datetime import datetime

# Add the current directory to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.system_orchestrator import TradingSystemOrchestrator
from config.settings import SystemConfig, TradingMode, AgentConfig, AgentType

async def run_advanced_agents_demo():
    """Demo showcasing the new advanced trading agents."""
    print("🚀 Advanced Trading Agents System - Sophisticated Strategies Demo")
    print("=" * 80)
    
    # Configure system for advanced agents
    config = SystemConfig()
    config.trading_mode = TradingMode.PAPER
    config.trading_symbols = [
        # Focus on high-volatility symbols for pattern recognition
        "AAPL", "GOOGL", "MSFT", "TSLA", "NVDA", "META", "AMZN", "NFLX"
    ]
    config.data_update_interval = 45  # 45 second updates for more analysis time
    
    # Create advanced agent configurations
    config.agent_configs = [
        # Traditional agents for comparison
        AgentConfig("conservative_baseline", AgentType.CONSERVATIVE, initial_capital=20000.0),
        AgentConfig("aggressive_baseline", AgentType.AGGRESSIVE, initial_capital=20000.0),
        
        # New advanced agents
        AgentConfig("fractal_master", AgentType.FRACTAL_ANALYSIS, initial_capital=20000.0),
        AgentConfig("candle_expert", AgentType.CANDLE_RANGE_THEORY, initial_capital=20000.0),
        AgentConfig("quant_pattern", AgentType.QUANTITATIVE_PATTERN, initial_capital=20000.0),
    ]
    
    # Create and initialize orchestrator
    orchestrator = TradingSystemOrchestrator(config)
    
    try:
        # Initialize system
        print("📋 Initializing advanced trading agents...")
        success = await orchestrator.initialize()
        
        if not success:
            print("❌ Failed to initialize system")
            return
        
        print("✅ Advanced agents initialized successfully")
        print(f"📊 Trading symbols: {len(config.trading_symbols)} symbols")
        print(f"🤖 Advanced agents: {len(config.agent_configs)}")
        print(f"⏱️  Analysis interval: {config.data_update_interval} seconds")
        print()
        
        # Display agent types
        print("🎯 Agent Strategies:")
        for agent_config in config.agent_configs:
            strategy_desc = {
                AgentType.CONSERVATIVE: "Risk-averse, steady returns",
                AgentType.AGGRESSIVE: "High-risk, momentum trading",
                AgentType.FRACTAL_ANALYSIS: "Fractal patterns on candlestick charts",
                AgentType.CANDLE_RANGE_THEORY: "Multi-timeframe candle analysis",
                AgentType.QUANTITATIVE_PATTERN: "ML-powered pattern recognition"
            }
            print(f"   {agent_config.agent_id}: {strategy_desc.get(agent_config.agent_type, 'Unknown')}")
        print()
        
        # Run system for demonstration
        print("🔄 Starting advanced trading competition...")
        print("Press Ctrl+C to stop the system")
        print("-" * 60)
        
        # Run for 8 cycles to demonstrate different strategies
        start_time = datetime.now()
        max_cycles = 8
        
        for cycle in range(max_cycles):
            try:
                print(f"\n🔄 Cycle {cycle + 1}/{max_cycles}")
                
                # Run one cycle
                await orchestrator._run_cycle()
                
                # Get and display detailed status
                status = await orchestrator.get_system_status()
                
                print(f"⏰ System Status:")
                print(f"   Total Return: {status['system_performance']['total_return']:.4f}")
                print(f"   Successful Trades: {status['system_performance']['successful_trades']}")
                print(f"   Failed Trades: {status['system_performance']['failed_trades']}")
                
                # Display agent performance with strategy details
                print(f"\n🤖 Agent Performance:")
                for agent_id, agent_data in status['agents'].items():
                    metrics = agent_data['performance_metrics']
                    agent_type = next(
                        (ac.agent_type for ac in config.agent_configs if ac.agent_id == agent_id),
                        "Unknown"
                    )
                    
                    print(f"   {agent_id} ({agent_type.value}):")
                    print(f"     Return: {metrics['total_return']:.4f}")
                    print(f"     Sharpe: {metrics['sharpe_ratio']:.4f}")
                    print(f"     Win Rate: {metrics['win_rate']:.4f}")
                    print(f"     Trades: {metrics['total_trades']}")
                
                # Get hierarchy rankings
                if orchestrator.hierarchy_manager:
                    hierarchy_result = await orchestrator.hierarchy_manager.run_oversight_cycle()
                    if "evaluation" in hierarchy_result:
                        rankings = hierarchy_result["evaluation"].get("rankings", {})
                        if rankings:
                            print(f"\n📊 Current Rankings:")
                            for agent_id, rank in sorted(rankings.items(), key=lambda x: x[1]):
                                agent_type = next(
                                    (ac.agent_type for ac in config.agent_configs if ac.agent_id == agent_id),
                                    "Unknown"
                                )
                                print(f"   {rank}. {agent_id} ({agent_type.value})")
                
                # Wait for next cycle
                if cycle < max_cycles - 1:
                    print(f"\n⏳ Waiting {config.data_update_interval} seconds for next cycle...")
                    await asyncio.sleep(config.data_update_interval)
                
            except KeyboardInterrupt:
                print("\n🛑 Received interrupt signal")
                break
            except Exception as e:
                print(f"❌ Error in cycle {cycle + 1}: {e}")
                await asyncio.sleep(5)  # Wait before retrying
        
        # Final analysis
        print("\n" + "="*60)
        print("📊 FINAL ANALYSIS - Advanced Trading Agents Competition")
        print("="*60)
        
        final_status = await orchestrator.get_system_status()
        
        print(f"\n🏆 Overall System Performance:")
        print(f"   Total Cycles: {final_status['cycle_count']}")
        print(f"   Total Return: {final_status['system_performance']['total_return']:.4f}")
        print(f"   Successful Trades: {final_status['system_performance']['successful_trades']}")
        print(f"   Failed Trades: {final_status['system_performance']['failed_trades']}")
        print(f"   Success Rate: {final_status['system_performance']['successful_trades'] / max(1, final_status['system_performance']['successful_trades'] + final_status['system_performance']['failed_trades']):.2%}")
        
        # Agent performance comparison
        print(f"\n🎯 Agent Strategy Performance:")
        agent_performance = []
        
        for agent_id, agent_data in final_status['agents'].items():
            metrics = agent_data['performance_metrics']
            agent_type = next(
                (ac.agent_type for ac in config.agent_configs if ac.agent_id == agent_id),
                "Unknown"
            )
            
            agent_performance.append({
                'id': agent_id,
                'type': agent_type.value,
                'return': metrics['total_return'],
                'sharpe': metrics['sharpe_ratio'],
                'win_rate': metrics['win_rate'],
                'trades': metrics['total_trades']
            })
        
        # Sort by total return
        agent_performance.sort(key=lambda x: x['return'], reverse=True)
        
        for i, agent in enumerate(agent_performance, 1):
            print(f"   {i}. {agent['id']} ({agent['type']}):")
            print(f"      Return: {agent['return']:.4f}")
            print(f"      Sharpe: {agent['sharpe']:.4f}")
            print(f"      Win Rate: {agent['win_rate']:.2%}")
            print(f"      Trades: {agent['trades']}")
        
        # Strategy insights
        print(f"\n💡 Strategy Insights:")
        advanced_agents = [a for a in agent_performance if a['type'] in ['fractal_analysis', 'candle_range_theory', 'quantitative_pattern']]
        traditional_agents = [a for a in agent_performance if a['type'] in ['conservative', 'aggressive']]
        
        if advanced_agents and traditional_agents:
            avg_advanced_return = sum(a['return'] for a in advanced_agents) / len(advanced_agents)
            avg_traditional_return = sum(a['return'] for a in traditional_agents) / len(traditional_agents)
            
            print(f"   Advanced Agents Avg Return: {avg_advanced_return:.4f}")
            print(f"   Traditional Agents Avg Return: {avg_traditional_return:.4f}")
            
            if avg_advanced_return > avg_traditional_return:
                print(f"   🎉 Advanced strategies outperformed traditional by {((avg_advanced_return - avg_traditional_return) / abs(avg_traditional_return) * 100):.1f}%")
            else:
                print(f"   📊 Traditional strategies outperformed advanced by {((avg_traditional_return - avg_advanced_return) / abs(avg_advanced_return) * 100):.1f}%")
        
        # Best performing strategy
        best_agent = agent_performance[0]
        print(f"\n🏆 Best Performing Strategy: {best_agent['type']}")
        print(f"   Agent: {best_agent['id']}")
        print(f"   Return: {best_agent['return']:.4f}")
        print(f"   Sharpe Ratio: {best_agent['sharpe']:.4f}")
        
    except Exception as e:
        print(f"❌ Fatal error: {e}")
    finally:
        print("\n🔄 Shutting down advanced trading system...")
        await orchestrator.shutdown()
        print("✅ Advanced trading system shutdown complete")

async def run_single_agent_demo():
    """Demo focusing on a single advanced agent type."""
    print("🎯 Single Agent Deep Dive Demo")
    print("=" * 50)
    
    agent_types = {
        "1": (AgentType.FRACTAL_ANALYSIS, "Fractal Analysis Agent"),
        "2": (AgentType.CANDLE_RANGE_THEORY, "Candle Range Theory Agent"),
        "3": (AgentType.QUANTITATIVE_PATTERN, "Quantitative Pattern Agent")
    }
    
    print("Available advanced agents:")
    for key, (agent_type, description) in agent_types.items():
        print(f"  {key}. {description}")
    
    try:
        choice = input("\nSelect an agent to demo (1-3): ").strip()
        
        if choice not in agent_types:
            print("❌ Invalid choice")
            return
        
        selected_type, description = agent_types[choice]
        
        print(f"\n🚀 Running {description} Demo")
        print("=" * 50)
        
        # Configure for single agent
        config = SystemConfig()
        config.trading_mode = TradingMode.PAPER
        config.trading_symbols = ["AAPL", "GOOGL", "MSFT", "TSLA", "NVDA"]
        config.data_update_interval = 30
        
        # Single agent configuration
        config.agent_configs = [
            AgentConfig(f"{selected_type.value}_demo", selected_type, initial_capital=50000.0)
        ]
        
        orchestrator = TradingSystemOrchestrator(config)
        
        try:
            await orchestrator.initialize()
            print(f"✅ {description} initialized")
            
            # Run 5 cycles
            for cycle in range(5):
                print(f"\n🔄 Cycle {cycle + 1}/5")
                await orchestrator._run_cycle()
                
                status = await orchestrator.get_system_status()
                agent_data = list(status['agents'].values())[0]
                metrics = agent_data['performance_metrics']
                
                print(f"   Return: {metrics['total_return']:.4f}")
                print(f"   Trades: {metrics['total_trades']}")
                print(f"   Win Rate: {metrics['win_rate']:.2%}")
                
                if cycle < 4:
                    await asyncio.sleep(30)
            
            print(f"\n✅ {description} demo completed")
            
        finally:
            await orchestrator.shutdown()
            
    except KeyboardInterrupt:
        print("\n👋 Demo cancelled")
    except Exception as e:
        print(f"❌ Error: {e}")

def main():
    """Main function for advanced agents demo."""
    print("🎯 Advanced Trading Agents System - Demo Menu")
    print("=" * 60)
    print()
    print("Available demos:")
    print("1. Full Advanced Agents Competition (6 agents)")
    print("2. Single Agent Deep Dive")
    print("3. Exit")
    print()
    
    try:
        choice = input("Select a demo (1-3): ").strip()
        
        if choice == "1":
            asyncio.run(run_advanced_agents_demo())
        elif choice == "2":
            asyncio.run(run_single_agent_demo())
        elif choice == "3":
            print("👋 Goodbye!")
        else:
            print("❌ Invalid choice")
    
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()
