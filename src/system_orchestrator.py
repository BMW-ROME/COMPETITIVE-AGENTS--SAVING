"""
Main system orchestrator for the competitive trading agents system.
"""
import asyncio
import logging
import json
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import signal
import sys

from src.base_agent import BaseTradingAgent
from src.trading_agents import (
    ConservativeTradingAgent, AggressiveTradingAgent, BalancedTradingAgent,
    FractalAnalysisAgent, CandleRangeTheoryAgent, QuantitativePatternAgent
)
from src.hierarchy_manager import HierarchyManager
from src.data_sources import DataAggregator, MarketDataProvider, NewsProvider, SocialMediaProvider
from src.alpaca_integration import AlpacaTradingInterface
from config.settings import SystemConfig, AlpacaConfig, DataSourceConfig, HierarchyConfig, AgentConfig, AgentType

class TradingSystemOrchestrator:
    """
    Main orchestrator for the competitive trading agents system.
    """
    
    def __init__(self, system_config: SystemConfig):
        self.system_config = system_config
        self.logger = logging.getLogger("TradingSystemOrchestrator")
        
        # Initialize components
        self.agents: Dict[str, BaseTradingAgent] = {}
        self.hierarchy_manager: Optional[HierarchyManager] = None
        self.data_aggregator: Optional[DataAggregator] = None
        self.alpaca_interface: Optional[AlpacaTradingInterface] = None
        
        # System state
        self.is_running = False
        self.cycle_count = 0
        self.start_time = None
        self.last_data_update = None
        
        # Performance tracking
        self.system_performance = {
            "total_cycles": 0,
            "successful_trades": 0,
            "failed_trades": 0,
            "total_return": 0.0,
            "system_uptime": 0.0
        }
        
        # Setup logging
        self._setup_logging()
        
        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _setup_logging(self):
        """Setup logging configuration."""
        logging.basicConfig(
            level=getattr(logging, self.system_config.log_level),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('logs/trading_system.log'),
                logging.StreamHandler(sys.stdout)
            ]
        )
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals."""
        self.logger.info(f"Received signal {signum}, initiating graceful shutdown...")
        asyncio.create_task(self.shutdown())
    
    async def initialize(self) -> bool:
        """Initialize all system components."""
        try:
            self.logger.info("Initializing trading system...")
            
            # Initialize data sources
            await self._initialize_data_sources()
            
            # Initialize Alpaca interface
            await self._initialize_alpaca_interface()
            
            # Initialize agents
            await self._initialize_agents()
            
            # Initialize hierarchy manager
            await self._initialize_hierarchy_manager()
            
            self.logger.info("Trading system initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Error initializing trading system: {e}")
            return False
    
    async def _initialize_data_sources(self):
        """Initialize data source providers."""
        try:
            data_config = DataSourceConfig()
            
            market_provider = MarketDataProvider(data_config)
            news_provider = NewsProvider(data_config)
            social_provider = SocialMediaProvider(data_config)
            
            self.data_aggregator = DataAggregator(market_provider, news_provider, social_provider)
            
            self.logger.info("Data sources initialized")
            
        except Exception as e:
            self.logger.error(f"Error initializing data sources: {e}")
            raise
    
    async def _initialize_alpaca_interface(self):
        """Initialize Alpaca trading interface."""
        try:
            alpaca_config = AlpacaConfig()
            
            self.alpaca_interface = AlpacaTradingInterface(alpaca_config, self.system_config)
            
            # Initialize connection
            success = await self.alpaca_interface.initialize()
            if not success:
                raise Exception("Failed to initialize Alpaca interface")
            
            self.logger.info("Alpaca interface initialized")
            
        except Exception as e:
            self.logger.error(f"Error initializing Alpaca interface: {e}")
            raise
    
    async def _initialize_agents(self):
        """Initialize trading agents."""
        try:
            for agent_config in self.system_config.agent_configs:
                if agent_config.agent_type == AgentType.CONSERVATIVE:
                    agent = ConservativeTradingAgent(agent_config)
                elif agent_config.agent_type == AgentType.AGGRESSIVE:
                    agent = AggressiveTradingAgent(agent_config)
                elif agent_config.agent_type == AgentType.BALANCED:
                    agent = BalancedTradingAgent(agent_config)
                elif agent_config.agent_type == AgentType.FRACTAL_ANALYSIS:
                    agent = FractalAnalysisAgent(agent_config)
                elif agent_config.agent_type == AgentType.CANDLE_RANGE_THEORY:
                    agent = CandleRangeTheoryAgent(agent_config)
                elif agent_config.agent_type == AgentType.QUANTITATIVE_PATTERN:
                    agent = QuantitativePatternAgent(agent_config)
                else:
                    self.logger.warning(f"Unknown agent type: {agent_config.agent_type}")
                    continue
                
                self.agents[agent.agent_id] = agent
                self.logger.info(f"Initialized agent: {agent.agent_id} ({agent_config.agent_type.value})")
            
            self.logger.info(f"Initialized {len(self.agents)} agents")
            
        except Exception as e:
            self.logger.error(f"Error initializing agents: {e}")
            raise
    
    async def _initialize_hierarchy_manager(self):
        """Initialize hierarchy manager."""
        try:
            hierarchy_config = HierarchyConfig()
            
            self.hierarchy_manager = HierarchyManager(hierarchy_config, self.system_config)
            
            # Register all agents with hierarchy manager
            for agent in self.agents.values():
                self.hierarchy_manager.register_agent(agent)
            
            self.logger.info("Hierarchy manager initialized")
            
        except Exception as e:
            self.logger.error(f"Error initializing hierarchy manager: {e}")
            raise
    
    async def run_system(self):
        """Run the main system loop."""
        try:
            self.is_running = True
            self.start_time = datetime.now()
            
            self.logger.info("Starting trading system...")
            
            # Main system loop
            while self.is_running:
                try:
                    await self._run_cycle()
                    await asyncio.sleep(self.system_config.data_update_interval)
                    
                except Exception as e:
                    self.logger.error(f"Error in system cycle: {e}")
                    await asyncio.sleep(5)  # Wait before retrying
            
        except Exception as e:
            self.logger.error(f"Fatal error in system: {e}")
        finally:
            await self.shutdown()
    
    async def _run_cycle(self):
        """Run one complete system cycle."""
        cycle_start = datetime.now()
        self.cycle_count += 1
        
        self.logger.info(f"Starting cycle {self.cycle_count}")
        
        try:
            # Step 1: Collect market data
            market_data = await self._collect_market_data()
            if not market_data:
                self.logger.warning("No market data collected, skipping cycle")
                return
            
            # Step 2: Run agent cycles
            agent_results = await self._run_agent_cycles(market_data)
            
            # Step 3: Execute trades
            trade_results = await self._execute_trades(agent_results)
            
            # Step 4: Run hierarchy oversight
            hierarchy_results = await self._run_hierarchy_oversight()
            
            # Step 5: Update system performance
            await self._update_system_performance(agent_results, trade_results)
            
            # Step 6: Log cycle results
            await self._log_cycle_results(agent_results, trade_results, hierarchy_results)
            
            cycle_duration = (datetime.now() - cycle_start).total_seconds()
            self.logger.info(f"Cycle {self.cycle_count} completed in {cycle_duration:.2f} seconds")
            
        except Exception as e:
            self.logger.error(f"Error in cycle {self.cycle_count}: {e}")
    
    async def _collect_market_data(self) -> Optional[Dict[str, Any]]:
        """Collect comprehensive market data."""
        try:
            if not self.data_aggregator:
                return None
            
            market_data = await self.data_aggregator.get_comprehensive_data(
                self.system_config.trading_symbols
            )
            
            self.last_data_update = datetime.now()
            return market_data
            
        except Exception as e:
            self.logger.error(f"Error collecting market data: {e}")
            return None
    
    async def _run_agent_cycles(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Run cycles for all agents."""
        agent_results = {}
        
        # Run agent cycles in parallel
        tasks = []
        for agent_id, agent in self.agents.items():
            task = asyncio.create_task(agent.run_cycle(market_data))
            tasks.append((agent_id, task))
        
        # Wait for all agents to complete
        for agent_id, task in tasks:
            try:
                result = await task
                agent_results[agent_id] = result
            except Exception as e:
                self.logger.error(f"Error in agent {agent_id} cycle: {e}")
                agent_results[agent_id] = {"error": str(e)}
        
        return agent_results
    
    async def _execute_trades(self, agent_results: Dict[str, Any]) -> Dict[str, Any]:
        """Execute trades from agent decisions."""
        trade_results = {}
        
        if not self.alpaca_interface:
            self.logger.warning("Alpaca interface not available, skipping trade execution")
            return trade_results
        
        for agent_id, result in agent_results.items():
            if "error" in result:
                continue
            
            decisions = result.get("decisions", [])
            agent_trade_results = []
            
            for decision in decisions:
                try:
                    # Execute trade through Alpaca
                    trade_result = await self.alpaca_interface.execute_trade_decision(decision)
                    
                    if trade_result:
                        agent_trade_results.append(trade_result)
                        
                        # Update system performance
                        if trade_result.get("status") == "filled":
                            self.system_performance["successful_trades"] += 1
                        else:
                            self.system_performance["failed_trades"] += 1
                    
                except Exception as e:
                    self.logger.error(f"Error executing trade for agent {agent_id}: {e}")
                    self.system_performance["failed_trades"] += 1
            
            trade_results[agent_id] = agent_trade_results
        
        return trade_results
    
    async def _run_hierarchy_oversight(self) -> Dict[str, Any]:
        """Run hierarchy manager oversight cycle."""
        try:
            if not self.hierarchy_manager:
                return {}
            
            hierarchy_results = await self.hierarchy_manager.run_oversight_cycle()
            return hierarchy_results
            
        except Exception as e:
            self.logger.error(f"Error in hierarchy oversight: {e}")
            return {"error": str(e)}
    
    async def _update_system_performance(self, agent_results: Dict[str, Any], trade_results: Dict[str, Any]):
        """Update system performance metrics."""
        try:
            self.system_performance["total_cycles"] = self.cycle_count
            
            # Calculate total return from all agents
            total_return = 0.0
            for agent in self.agents.values():
                metrics = agent.calculate_performance_metrics()
                total_return += metrics.total_return
            
            self.system_performance["total_return"] = total_return
            
            # Calculate system uptime
            if self.start_time:
                uptime = (datetime.now() - self.start_time).total_seconds()
                self.system_performance["system_uptime"] = uptime
            
        except Exception as e:
            self.logger.error(f"Error updating system performance: {e}")
    
    async def _log_cycle_results(self, agent_results: Dict[str, Any], trade_results: Dict[str, Any], hierarchy_results: Dict[str, Any]):
        """Log cycle results for monitoring."""
        try:
            # Log summary
            self.logger.info(f"Cycle {self.cycle_count} Summary:")
            self.logger.info(f"  Agents: {len(agent_results)}")
            self.logger.info(f"  Trades executed: {sum(len(trades) for trades in trade_results.values())}")
            self.logger.info(f"  System return: {self.system_performance['total_return']:.4f}")
            
            # Log agent performance
            for agent_id, result in agent_results.items():
                if "error" not in result:
                    decisions = len(result.get("decisions", []))
                    reflections = len(result.get("reflections", []))
                    self.logger.info(f"  Agent {agent_id}: {decisions} decisions, {reflections} reflections")
            
            # Log hierarchy results
            if hierarchy_results and "evaluation" in hierarchy_results:
                evaluation = hierarchy_results["evaluation"]
                rankings = evaluation.get("rankings", {})
                if rankings:
                    best_agent = min(rankings.items(), key=lambda x: x[1])[0]
                    self.logger.info(f"  Best performing agent: {best_agent}")
            
        except Exception as e:
            self.logger.error(f"Error logging cycle results: {e}")
    
    async def get_system_status(self) -> Dict[str, Any]:
        """Get current system status."""
        try:
            status = {
                "timestamp": datetime.now().isoformat(),
                "is_running": self.is_running,
                "cycle_count": self.cycle_count,
                "start_time": self.start_time.isoformat() if self.start_time else None,
                "last_data_update": self.last_data_update.isoformat() if self.last_data_update else None,
                "system_performance": self.system_performance,
                "agents": {}
            }
            
            # Get agent statuses
            for agent_id, agent in self.agents.items():
                status["agents"][agent_id] = {
                    "agent_type": agent.agent_type.value,
                    "status": "active",
                    "performance_metrics": agent.calculate_performance_metrics().__dict__,
                    "trade_count": len(agent.trade_history)
                }
            
            # Get hierarchy status
            if self.hierarchy_manager:
                status["hierarchy"] = await self.hierarchy_manager.get_system_status()
            
            # Get portfolio status
            if self.alpaca_interface:
                status["portfolio"] = await self.alpaca_interface.get_portfolio_performance()
            
            return status
            
        except Exception as e:
            self.logger.error(f"Error getting system status: {e}")
            return {"error": str(e)}
    
    async def shutdown(self):
        """Gracefully shutdown the system."""
        try:
            self.logger.info("Shutting down trading system...")
            
            self.is_running = False
            
            # Close all positions if in live trading mode
            if (self.system_config.trading_mode.value == "live" and 
                self.alpaca_interface):
                self.logger.info("Closing all positions...")
                await self.alpaca_interface.close_all_positions()
            
            # Log final performance
            self.logger.info(f"Final system performance: {self.system_performance}")
            
            self.logger.info("Trading system shutdown complete")
            
        except Exception as e:
            self.logger.error(f"Error during shutdown: {e}")


async def main():
    """Main entry point for the trading system."""
    # Load configuration
    system_config = SystemConfig()
    
    # Create and initialize orchestrator
    orchestrator = TradingSystemOrchestrator(system_config)
    
    # Initialize system
    success = await orchestrator.initialize()
    if not success:
        print("Failed to initialize trading system")
        return
    
    # Run system
    try:
        await orchestrator.run_system()
    except KeyboardInterrupt:
        print("Received keyboard interrupt, shutting down...")
    except Exception as e:
        print(f"Fatal error: {e}")
    finally:
        await orchestrator.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
