#!/usr/bin/env python3
"""
Enhanced System Orchestrator
============================
Implements competitive trading with:
- All agents making decisions every cycle
- Continuous learning and reflections
- Quick trades (scalping)
- Hierarchy-based agent selection
- Performance-based agent scaling
"""

import asyncio
import logging
import random
import numpy as np
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass

from enhanced_agent_competition import EnhancedAgentCompetition, AgentPerformance
from base_agent import BaseTradingAgent
from data_sources import DataAggregator
from real_alpaca_integration import RealAlpacaTradingInterface
from advanced_risk_manager import AdvancedRiskManager
from backtesting_engine import RealTimeBacktestingEngine
from mcp_integration_simple import SimpleMCPManager

@dataclass
class SystemMetrics:
    """System performance metrics"""
    total_cycles: int = 0
    total_trades: int = 0
    total_decisions: int = 0
    total_reflections: int = 0
    system_return: float = 0.0
    best_agent: str = ""
    worst_agent: str = ""

class EnhancedSystemOrchestrator:
    """Enhanced system orchestrator with competitive trading"""
    
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger("EnhancedSystemOrchestrator")
        
        # Initialize components
        self.data_aggregator = DataAggregator(config)
        self.trading_interface = RealAlpacaTradingInterface(config)
        self.risk_manager = AdvancedRiskManager(config)
        self.backtesting_engine = RealTimeBacktestingEngine(config)
        self.mcp_manager = SimpleMCPManager(config)
        
        # Enhanced competitive system
        self.competition_system = EnhancedAgentCompetition(self.logger)
        self.system_metrics = SystemMetrics()
        
        # Agent performance tracking
        self.agent_performance = {}
        self.cycle_results = []
        
        # Quick trade settings
        self.quick_trade_symbols = ['SPY', 'QQQ', 'AAPL', 'MSFT', 'TSLA']
        self.scalping_agents = ['scalping_1', 'momentum_1', 'arbitrage_1']
        
        # Learning parameters
        self.learning_enabled = True
        self.reflection_interval = 300  # 5 minutes
        self.performance_update_interval = 600  # 10 minutes
        
    async def initialize(self) -> bool:
        """Initialize the enhanced system"""
        try:
            self.logger.info("Initializing Enhanced Trading System...")
            
            # Initialize data sources
            await self.data_aggregator.initialize()
            
            # Initialize trading interface
            await self.trading_interface.initialize()
            
            # Initialize risk manager
            await self.risk_manager.initialize()
            
            # Initialize backtesting engine
            await self.backtesting_engine.initialize()
            
            # Initialize MCP manager
            await self.mcp_manager.initialize()
            
            # Initialize competitive system
            self.competition_system.initialize_agents()
            
            # Initialize agent performance tracking
            self._initialize_agent_performance()
            
            self.logger.info("Enhanced Trading System initialized successfully!")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize system: {e}")
            return False
    
    def _initialize_agent_performance(self):
        """Initialize agent performance tracking"""
        agent_ids = [
            'conservative_1', 'balanced_1', 'aggressive_1',
            'fractal_1', 'candle_range_1', 'quant_pattern_1',
            'forex_major_1', 'forex_minor_1', 'crypto_major_1',
            'crypto_defi_1', 'multi_asset_arb_1', 'ai_enhanced_1'
        ]
        
        for agent_id in agent_ids:
            self.agent_performance[agent_id] = AgentPerformance(agent_id)
    
    async def run_system(self):
        """Run the enhanced competitive trading system"""
        self.logger.info("Starting Enhanced Competitive Trading System...")
        
        while True:
            try:
                # Run competitive cycle
                cycle_result = await self._run_competitive_cycle()
                
                # Update system metrics
                self._update_system_metrics(cycle_result)
                
                # Log cycle summary
                self._log_cycle_summary(cycle_result)
                
                # Wait for next cycle
                await asyncio.sleep(30)  # 30-second cycles
                
            except KeyboardInterrupt:
                self.logger.info("System stopped by user")
                break
            except Exception as e:
                self.logger.error(f"Error in system cycle: {e}")
                await asyncio.sleep(60)  # Wait before retry
    
    async def _run_competitive_cycle(self) -> Dict[str, Any]:
        """Run a competitive trading cycle"""
        cycle_start = datetime.now()
        
        # Get market data
        market_data = await self.data_aggregator.get_market_data()
        
        # Run competitive cycle
        cycle_result = self.competition_system.run_competitive_cycle(market_data)
        
        # Execute selected trades
        executed_trades = await self._execute_selected_trades(cycle_result['hierarchy_selection'])
        cycle_result['executed_trades'] = executed_trades
        
        # Update agent performance
        self._update_agent_performance(cycle_result)
        
        # Run reflections
        reflections = await self._run_agent_reflections(cycle_result)
        cycle_result['reflections'] = reflections
        
        # Update learning
        if self.learning_enabled:
            await self._update_agent_learning(cycle_result)
        
        cycle_result['cycle_duration'] = (datetime.now() - cycle_start).total_seconds()
        
        return cycle_result
    
    async def _execute_selected_trades(self, selected_trades: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Execute selected trades"""
        executed_trades = []
        
        for trade in selected_trades:
            try:
                # Execute trade
                result = await self.trading_interface.execute_trade(trade)
                
                if result:
                    executed_trades.append(result)
                    self.logger.info(f"Executed trade: {trade['symbol']} {trade['action']} "
                                   f"{trade['quantity']} @ {trade['price']}")
                
            except Exception as e:
                self.logger.error(f"Failed to execute trade: {e}")
        
        return executed_trades
    
    async def _run_agent_reflections(self, cycle_result: Dict[str, Any]) -> Dict[str, Any]:
        """Run agent reflections"""
        reflections = {}
        
        for agent_id, performance in self.agent_performance.items():
            try:
                # Check if reflection is needed
                if self._should_reflect(agent_id):
                    reflection = await self._generate_agent_reflection(agent_id, cycle_result)
                    if reflection:
                        reflections[agent_id] = reflection
                        self.logger.info(f"Agent {agent_id} completed reflection")
                
            except Exception as e:
                self.logger.error(f"Error in reflection for {agent_id}: {e}")
        
        return reflections
    
    def _should_reflect(self, agent_id: str) -> bool:
        """Check if agent should reflect"""
        performance = self.agent_performance[agent_id]
        
        # Always reflect if no recent reflection
        if performance.last_decision_time is None:
            return True
        
        # Reflect every 5 minutes
        time_since_reflection = (datetime.now() - performance.last_decision_time).total_seconds()
        return time_since_reflection >= self.reflection_interval
    
    async def _generate_agent_reflection(self, agent_id: str, cycle_result: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Generate agent reflection"""
        performance = self.agent_performance[agent_id]
        
        reflection = {
            'agent_id': agent_id,
            'timestamp': datetime.now(),
            'performance_analysis': {
                'total_trades': performance.total_trades,
                'win_rate': performance.win_rate,
                'total_pnl': performance.total_pnl,
                'confidence_score': performance.confidence_score
            },
            'learning_insights': self._generate_learning_insights(agent_id),
            'strategy_adjustments': self._suggest_strategy_adjustments(agent_id),
            'confidence_update': self._update_agent_confidence(agent_id)
        }
        
        # Update performance
        performance.last_decision_time = datetime.now()
        performance.confidence_score = reflection['confidence_update']
        
        return reflection
    
    def _generate_learning_insights(self, agent_id: str) -> List[str]:
        """Generate learning insights"""
        performance = self.agent_performance[agent_id]
        insights = []
        
        if performance.win_rate > 0.6:
            insights.append("High win rate - continue current strategy")
        elif performance.win_rate < 0.4:
            insights.append("Low win rate - consider strategy adjustment")
        
        if performance.total_pnl > 0:
            insights.append("Positive PnL - strategy is profitable")
        else:
            insights.append("Negative PnL - need to review approach")
        
        return insights
    
    def _suggest_strategy_adjustments(self, agent_id: str) -> List[str]:
        """Suggest strategy adjustments"""
        performance = self.agent_performance[agent_id]
        adjustments = []
        
        if performance.win_rate < 0.4:
            adjustments.append("Reduce position size")
            adjustments.append("Increase confidence threshold")
        
        if performance.total_pnl < 0:
            adjustments.append("Review stop-loss levels")
            adjustments.append("Consider different timeframes")
        
        return adjustments
    
    def _update_agent_confidence(self, agent_id: str) -> float:
        """Update agent confidence"""
        performance = self.agent_performance[agent_id]
        
        # Learning rate adjustment
        if performance.win_rate > 0.6:
            performance.learning_rate = min(0.2, performance.learning_rate + 0.01)
        elif performance.win_rate < 0.4:
            performance.learning_rate = max(0.05, performance.learning_rate - 0.01)
        
        # Update confidence
        new_confidence = performance.confidence_score + (performance.learning_rate * 0.1)
        return max(0.1, min(0.9, new_confidence))
    
    def _update_agent_performance(self, cycle_result: Dict[str, Any]):
        """Update agent performance"""
        for agent_id, decision in cycle_result['agent_decisions'].items():
            if decision:
                performance = self.agent_performance[agent_id]
                performance.total_trades += 1
                performance.last_decision_time = datetime.now()
    
    async def _update_agent_learning(self, cycle_result: Dict[str, Any]):
        """Update agent learning"""
        for agent_id, performance in self.agent_performance.items():
            # Update learning rate based on performance
            if performance.win_rate > 0.6:
                performance.learning_rate = min(0.2, performance.learning_rate + 0.01)
            elif performance.win_rate < 0.4:
                performance.learning_rate = max(0.05, performance.learning_rate - 0.01)
    
    def _update_system_metrics(self, cycle_result: Dict[str, Any]):
        """Update system metrics"""
        self.system_metrics.total_cycles += 1
        self.system_metrics.total_decisions += cycle_result['total_decisions']
        self.system_metrics.total_reflections += cycle_result['total_reflections']
        self.system_metrics.total_trades += len(cycle_result.get('executed_trades', []))
        
        # Update best/worst agents
        if self.agent_performance:
            best_agent = max(self.agent_performance.items(), key=lambda x: x[1].total_pnl)
            worst_agent = min(self.agent_performance.items(), key=lambda x: x[1].total_pnl)
            self.system_metrics.best_agent = best_agent[0]
            self.system_metrics.worst_agent = worst_agent[0]
    
    def _log_cycle_summary(self, cycle_result: Dict[str, Any]):
        """Log cycle summary"""
        self.logger.info(f"Cycle {cycle_result['cycle']} Summary:")
        self.logger.info(f"  Total Decisions: {cycle_result['total_decisions']}")
        self.logger.info(f"  Total Reflections: {cycle_result['total_reflections']}")
        self.logger.info(f"  Executed Trades: {len(cycle_result.get('executed_trades', []))}")
        self.logger.info(f"  Cycle Duration: {cycle_result['cycle_duration']:.2f}s")
        
        # Log agent performance
        for agent_id, performance in self.agent_performance.items():
            self.logger.info(f"  Agent {agent_id}: {performance.total_trades} trades, "
                           f"{performance.win_rate:.2f} win rate, "
                           f"${performance.total_pnl:.2f} PnL")
        
        # Log system metrics
        self.logger.info(f"System Metrics:")
        self.logger.info(f"  Total Cycles: {self.system_metrics.total_cycles}")
        self.logger.info(f"  Total Decisions: {self.system_metrics.total_decisions}")
        self.logger.info(f"  Total Reflections: {self.system_metrics.total_reflections}")
        self.logger.info(f"  Total Trades: {self.system_metrics.total_trades}")
        self.logger.info(f"  Best Agent: {self.system_metrics.best_agent}")
        self.logger.info(f"  Worst Agent: {self.system_metrics.worst_agent}")
