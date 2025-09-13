#!/usr/bin/env python3
"""
Example script to run the competitive trading agents system.
This demonstrates how to set up and run the system with different configurations.
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

async def run_paper_trading_example():
    """Example: Run the system in paper trading mode."""
    print("🚀 Starting Competitive Trading Agents System - Paper Trading Mode")
    print("=" * 70)
    
    # Configure system for paper trading
    config = SystemConfig()
    config.trading_mode = TradingMode.PAPER
    config.trading_symbols = ["AAPL", "GOOGL", "MSFT"]  # Reduced symbols for demo
    config.data_update_interval = 60  # 1 minute updates for demo
    
    # Create custom agent configurations
    config.agent_configs = [
        AgentConfig("conservative_demo", AgentType.CONSERVATIVE, initial_capital=10000.0),
        AgentConfig("aggressive_demo", AgentType.AGGRESSIVE, initial_capital=10000.0)
    ]
    
    # Create and initialize orchestrator
    orchestrator = TradingSystemOrchestrator(config)
    
    try:
        # Initialize system
        print("📋 Initializing system components...")
        success = await orchestrator.initialize()
        
        if not success:
            print("❌ Failed to initialize system")
            return
        
        print("✅ System initialized successfully")
        print(f"📊 Trading symbols: {config.trading_symbols}")
        print(f"🤖 Agents: {len(config.agent_configs)}")
        print(f"⏱️  Update interval: {config.data_update_interval} seconds")
        print()
        
        # Run system for a limited time (demo purposes)
        print("🔄 Starting trading cycles...")
        print("Press Ctrl+C to stop the system")
        print("-" * 50)
        
        # Run for 5 minutes in demo mode
        start_time = datetime.now()
        max_runtime = 300  # 5 minutes
        
        while (datetime.now() - start_time).total_seconds() < max_runtime:
            try:
                # Run one cycle
                await orchestrator._run_cycle()
                
                # Get and display status
                status = await orchestrator.get_system_status()
                print(f"⏰ Cycle {status['cycle_count']}: "
                      f"Return: {status['system_performance']['total_return']:.4f}, "
                      f"Uptime: {status['system_performance']['system_uptime']:.0f}s")
                
                # Wait for next cycle
                await asyncio.sleep(config.data_update_interval)
                
            except KeyboardInterrupt:
                print("\n🛑 Received interrupt signal")
                break
            except Exception as e:
                print(f"❌ Error in cycle: {e}")
                await asyncio.sleep(5)  # Wait before retrying
        
        print("\n📊 Final System Status:")
        final_status = await orchestrator.get_system_status()
        print(f"   Total cycles: {final_status['cycle_count']}")
        print(f"   Total return: {final_status['system_performance']['total_return']:.4f}")
        print(f"   Successful trades: {final_status['system_performance']['successful_trades']}")
        print(f"   Failed trades: {final_status['system_performance']['failed_trades']}")
        
        # Display agent performance
        print("\n🤖 Agent Performance:")
        for agent_id, agent_data in final_status['agents'].items():
            metrics = agent_data['performance_metrics']
            print(f"   {agent_id}:")
            print(f"     Return: {metrics['total_return']:.4f}")
            print(f"     Sharpe: {metrics['sharpe_ratio']:.4f}")
            print(f"     Win Rate: {metrics['win_rate']:.4f}")
            print(f"     Trades: {metrics['total_trades']}")
        
    except Exception as e:
        print(f"❌ Fatal error: {e}")
    finally:
        print("\n🔄 Shutting down system...")
        await orchestrator.shutdown()
        print("✅ System shutdown complete")

async def run_custom_agents_example():
    """Example: Run with custom agent configurations."""
    print("🚀 Starting Custom Agents Example")
    print("=" * 50)
    
    # Configure system
    config = SystemConfig()
    config.trading_mode = TradingMode.PAPER
    config.trading_symbols = ["AAPL", "TSLA"]  # High volatility stocks
    
    # Create three different agent types
    config.agent_configs = [
        AgentConfig("conservative_1", AgentType.CONSERVATIVE, initial_capital=15000.0),
        AgentConfig("aggressive_1", AgentType.AGGRESSIVE, initial_capital=15000.0),
        AgentConfig("balanced_1", AgentType.BALANCED, initial_capital=15000.0)
    ]
    
    # Create orchestrator
    orchestrator = TradingSystemOrchestrator(config)
    
    try:
        # Initialize
        print("📋 Initializing custom agents...")
        success = await orchestrator.initialize()
        
        if not success:
            print("❌ Failed to initialize")
            return
        
        print("✅ Custom agents initialized")
        print(f"🤖 Agent types: Conservative, Aggressive, Balanced")
        print()
        
        # Run for 3 cycles to demonstrate competition
        for cycle in range(3):
            print(f"🔄 Running cycle {cycle + 1}/3...")
            await orchestrator._run_cycle()
            
            # Get hierarchy results
            if orchestrator.hierarchy_manager:
                hierarchy_result = await orchestrator.hierarchy_manager.run_oversight_cycle()
                if "evaluation" in hierarchy_result:
                    rankings = hierarchy_result["evaluation"].get("rankings", {})
                    if rankings:
                        print("📊 Current Rankings:")
                        for agent_id, rank in sorted(rankings.items(), key=lambda x: x[1]):
                            print(f"   {rank}. {agent_id}")
            
            await asyncio.sleep(30)  # Wait between cycles
        
        print("\n✅ Custom agents example completed")
        
    except Exception as e:
        print(f"❌ Error: {e}")
    finally:
        await orchestrator.shutdown()

async def run_data_analysis_example():
    """Example: Demonstrate data analysis capabilities."""
    print("🚀 Starting Data Analysis Example")
    print("=" * 50)
    
    # Import data components
    from src.data_sources import DataAggregator, MarketDataProvider, NewsProvider, SocialMediaProvider
    from config.settings import DataSourceConfig
    
    try:
        # Initialize data providers
        data_config = DataSourceConfig()
        market_provider = MarketDataProvider(data_config)
        news_provider = NewsProvider(data_config)
        social_provider = SocialMediaProvider(data_config)
        
        data_aggregator = DataAggregator(market_provider, news_provider, social_provider)
        
        print("📊 Collecting market data...")
        
        # Get comprehensive data
        symbols = ["AAPL", "GOOGL"]
        data = await data_aggregator.get_comprehensive_data(symbols)
        
        if data:
            print("✅ Data collected successfully")
            print(f"📈 Symbols analyzed: {data.get('symbols', [])}")
            print(f"⏰ Timestamp: {data.get('timestamp', 'N/A')}")
            
            # Display market overview
            market_overview = data.get('market_overview', {})
            print(f"📊 Market Trend: {market_overview.get('market_trend', 'N/A')}")
            print(f"📊 Volatility: {market_overview.get('volatility_level', 'N/A')}")
            print(f"📊 Sentiment: {market_overview.get('sentiment_score', 0):.3f}")
            print(f"📊 Risk Level: {market_overview.get('risk_level', 'N/A')}")
            
            # Display technical indicators
            technical_indicators = data.get('technical_indicators', {})
            for symbol, indicators in technical_indicators.items():
                print(f"\n📈 {symbol} Technical Indicators:")
                print(f"   RSI: {indicators.get('rsi', 0):.2f}")
                print(f"   SMA 20: {indicators.get('sma_20', 0):.2f}")
                print(f"   MACD: {indicators.get('macd', 0):.4f}")
                print(f"   Volume Ratio: {indicators.get('volume_ratio', 0):.2f}")
            
            # Display news sentiment
            news_sentiment = data.get('news_sentiment', {})
            for symbol, sentiment in news_sentiment.items():
                print(f"\n📰 {symbol} News Sentiment: {sentiment:.3f}")
            
            # Display social sentiment
            social_sentiment = data.get('social_sentiment', {})
            for symbol, sentiment_data in social_sentiment.items():
                print(f"\n🐦 {symbol} Social Sentiment:")
                print(f"   Twitter: {sentiment_data.get('twitter', 0):.3f}")
                print(f"   Reddit: {sentiment_data.get('reddit', 0):.3f}")
                print(f"   Combined: {sentiment_data.get('combined', 0):.3f}")
        
        else:
            print("❌ Failed to collect data")
    
    except Exception as e:
        print(f"❌ Error in data analysis: {e}")

def main():
    """Main function to run examples."""
    print("🎯 Competitive Trading Agents System - Examples")
    print("=" * 60)
    print()
    print("Available examples:")
    print("1. Paper Trading Demo (5 minutes)")
    print("2. Custom Agents Competition")
    print("3. Data Analysis Demo")
    print("4. Run all examples")
    print()
    
    try:
        choice = input("Select an example (1-4): ").strip()
        
        if choice == "1":
            asyncio.run(run_paper_trading_example())
        elif choice == "2":
            asyncio.run(run_custom_agents_example())
        elif choice == "3":
            asyncio.run(run_data_analysis_example())
        elif choice == "4":
            print("🔄 Running all examples...")
            asyncio.run(run_data_analysis_example())
            print("\n" + "="*50 + "\n")
            asyncio.run(run_custom_agents_example())
            print("\n" + "="*50 + "\n")
            asyncio.run(run_paper_trading_example())
        else:
            print("❌ Invalid choice")
    
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()
