"""
Base trading agent with self-reflection and learning capabilities.
"""
import asyncio
import logging
import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
import json

from config.settings import AgentConfig, AgentType

@dataclass
class TradeDecision:
    """Represents a trading decision made by an agent."""
    symbol: str
    action: str  # 'BUY', 'SELL', 'HOLD'
    quantity: float
    price: float
    confidence: float
    reasoning: str
    timestamp: datetime
    agent_id: str

@dataclass
class PerformanceMetrics:
    """Performance metrics for an agent."""
    total_return: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    profit_factor: float
    total_trades: int
    winning_trades: int
    losing_trades: int
    avg_win: float
    avg_loss: float

class BaseTradingAgent(ABC):
    """
    Base class for competitive trading agents with self-reflection capabilities.
    """
    
    def __init__(self, config: AgentConfig):
        self.config = config
        self.agent_id = config.agent_id
        self.agent_type = config.agent_type
        self.initial_capital = config.initial_capital
        self.current_capital = config.initial_capital
        self.positions = {}
        self.trade_history = []
        self.performance_history = []
        self.learning_memory = []
        self.competitor_performance = {}
        
        # Self-reflection components
        self.reflection_interval = 300  # 5 minutes
        self.last_reflection = datetime.now()
        self.performance_metrics = None
        
        # Learning components
        self.learning_rate = config.learning_rate
        self.memory_size = config.memory_size
        
        # Logging
        self.logger = logging.getLogger(f"Agent-{self.agent_id}")
        
    @abstractmethod
    async def analyze_market_data(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze market data and return insights."""
        pass
    
    @abstractmethod
    async def make_trading_decision(self, market_data: Dict[str, Any]) -> TradeDecision:
        """Make a trading decision based on market data."""
        pass
    
    @abstractmethod
    async def update_strategy(self, performance_feedback: Dict[str, Any]) -> None:
        """Update trading strategy based on performance feedback."""
        pass
    
    async def self_reflect(self) -> Dict[str, Any]:
        """
        Perform self-reflection to analyze performance and identify improvements.
        """
        self.logger.info(f"Agent {self.agent_id} performing self-reflection")
        
        # Calculate current performance metrics
        current_metrics = self.calculate_performance_metrics()
        
        # Compare with previous performance
        reflection_insights = self.analyze_performance_trends(current_metrics)
        
        # Identify areas for improvement
        improvement_areas = self.identify_improvement_areas(reflection_insights)
        
        # Generate learning insights
        learning_insights = self.generate_learning_insights()
        
        reflection_result = {
            "timestamp": datetime.now(),
            "agent_id": self.agent_id,
            "current_metrics": current_metrics,
            "reflection_insights": reflection_insights,
            "improvement_areas": improvement_areas,
            "learning_insights": learning_insights,
            "action_plan": self.create_action_plan(improvement_areas)
        }
        
        # Store reflection in memory
        self.learning_memory.append(reflection_result)
        if len(self.learning_memory) > self.memory_size:
            self.learning_memory.pop(0)
        
        self.last_reflection = datetime.now()
        return reflection_result
    
    def calculate_performance_metrics(self) -> PerformanceMetrics:
        """Calculate comprehensive performance metrics."""
        if not self.trade_history:
            return PerformanceMetrics(0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
        
        # Calculate returns
        returns = [trade.get('return', 0) for trade in self.trade_history if 'return' in trade]
        total_return = sum(returns)
        
        # Calculate Sharpe ratio
        if returns and len(returns) > 1:
            sharpe_ratio = np.mean(returns) / np.std(returns) if np.std(returns) > 0 else 0
        else:
            sharpe_ratio = 0
        
        # Calculate max drawdown
        cumulative_returns = np.cumsum(returns)
        running_max = np.maximum.accumulate(cumulative_returns)
        drawdowns = cumulative_returns - running_max
        max_drawdown = np.min(drawdowns) if len(drawdowns) > 0 else 0
        
        # Calculate win rate and profit factor
        winning_trades = [r for r in returns if r > 0]
        losing_trades = [r for r in returns if r < 0]
        
        win_rate = len(winning_trades) / len(returns) if returns else 0
        profit_factor = abs(sum(winning_trades) / sum(losing_trades)) if losing_trades else float('inf')
        
        return PerformanceMetrics(
            total_return=total_return,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_drawdown,
            win_rate=win_rate,
            profit_factor=profit_factor,
            total_trades=len(self.trade_history),
            winning_trades=len(winning_trades),
            losing_trades=len(losing_trades),
            avg_win=np.mean(winning_trades) if winning_trades else 0,
            avg_loss=np.mean(losing_trades) if losing_trades else 0
        )
    
    def analyze_performance_trends(self, current_metrics: PerformanceMetrics) -> Dict[str, Any]:
        """Analyze performance trends over time."""
        if not self.performance_history:
            return {"trend": "baseline", "change": 0}
        
        previous_metrics = self.performance_history[-1]
        
        # Calculate changes
        return_change = current_metrics.total_return - previous_metrics.total_return
        sharpe_change = current_metrics.sharpe_ratio - previous_metrics.sharpe_ratio
        win_rate_change = current_metrics.win_rate - previous_metrics.win_rate
        
        # Determine trend
        if return_change > 0 and sharpe_change > 0:
            trend = "improving"
        elif return_change < 0 and sharpe_change < 0:
            trend = "declining"
        else:
            trend = "mixed"
        
        return {
            "trend": trend,
            "return_change": return_change,
            "sharpe_change": sharpe_change,
            "win_rate_change": win_rate_change,
            "overall_change": (return_change + sharpe_change + win_rate_change) / 3
        }
    
    def identify_improvement_areas(self, reflection_insights: Dict[str, Any]) -> List[str]:
        """Identify specific areas for improvement."""
        improvement_areas = []
        
        if reflection_insights.get("win_rate_change", 0) < 0:
            improvement_areas.append("trade_accuracy")
        
        if reflection_insights.get("sharpe_change", 0) < 0:
            improvement_areas.append("risk_management")
        
        if reflection_insights.get("return_change", 0) < 0:
            improvement_areas.append("profit_optimization")
        
        # Add agent-specific improvement areas
        if self.agent_type == AgentType.CONSERVATIVE:
            if reflection_insights.get("return_change", 0) < 0:
                improvement_areas.append("opportunity_selection")
        elif self.agent_type == AgentType.AGGRESSIVE:
            if reflection_insights.get("win_rate_change", 0) < 0:
                improvement_areas.append("risk_control")
        
        return improvement_areas
    
    def generate_learning_insights(self) -> Dict[str, Any]:
        """Generate insights for learning and adaptation."""
        recent_trades = self.trade_history[-10:] if len(self.trade_history) >= 10 else self.trade_history
        
        if not recent_trades:
            return {"insights": [], "patterns": []}
        
        # Analyze trade patterns
        successful_patterns = []
        failed_patterns = []
        
        for trade in recent_trades:
            if trade.get('return', 0) > 0:
                successful_patterns.append(trade.get('reasoning', ''))
            else:
                failed_patterns.append(trade.get('reasoning', ''))
        
        return {
            "insights": [
                f"Recent win rate: {len(successful_patterns)}/{len(recent_trades)}",
                f"Average confidence in successful trades: {np.mean([t.get('confidence', 0) for t in successful_patterns]) if successful_patterns else 0:.2f}",
                f"Average confidence in failed trades: {np.mean([t.get('confidence', 0) for t in failed_patterns]) if failed_patterns else 0:.2f}"
            ],
            "patterns": {
                "successful": successful_patterns,
                "failed": failed_patterns
            }
        }
    
    def create_action_plan(self, improvement_areas: List[str]) -> Dict[str, str]:
        """Create an action plan based on improvement areas."""
        action_plan = {}
        
        for area in improvement_areas:
            if area == "trade_accuracy":
                action_plan[area] = "Increase confidence threshold and improve signal quality"
            elif area == "risk_management":
                action_plan[area] = "Implement stricter position sizing and stop-loss rules"
            elif area == "profit_optimization":
                action_plan[area] = "Optimize entry and exit timing strategies"
            elif area == "opportunity_selection":
                action_plan[area] = "Be more selective with high-probability setups"
            elif area == "risk_control":
                action_plan[area] = "Reduce position sizes and increase diversification"
        
        return action_plan
    
    async def learn_from_competitor(self, competitor_performance: Dict[str, Any]) -> None:
        """Learn from competitor's performance and strategies."""
        self.competitor_performance = competitor_performance
        
        # Analyze competitor's strengths
        competitor_metrics = competitor_performance.get('metrics', {})
        my_metrics = self.calculate_performance_metrics()
        
        # Identify what competitor is doing better
        learning_opportunities = []
        
        if competitor_metrics.get('sharpe_ratio', 0) > my_metrics.sharpe_ratio:
            learning_opportunities.append("risk_adjusted_returns")
        
        if competitor_metrics.get('win_rate', 0) > my_metrics.win_rate:
            learning_opportunities.append("trade_accuracy")
        
        if competitor_metrics.get('total_return', 0) > my_metrics.total_return:
            learning_opportunities.append("profit_generation")
        
        # Store learning insights
        learning_insight = {
            "timestamp": datetime.now(),
            "competitor_id": competitor_performance.get('agent_id'),
            "learning_opportunities": learning_opportunities,
            "my_metrics": my_metrics.__dict__,
            "competitor_metrics": competitor_metrics
        }
        
        self.learning_memory.append(learning_insight)
        self.logger.info(f"Agent {self.agent_id} learned from competitor: {learning_opportunities}")
    
    async def execute_trade(self, decision: TradeDecision) -> Dict[str, Any]:
        """Execute a trading decision."""
        self.logger.info(f"Agent {self.agent_id} executing trade: {decision.action} {decision.quantity} {decision.symbol}")
        
        # Store trade decision
        trade_record = {
            "timestamp": decision.timestamp,
            "symbol": decision.symbol,
            "action": decision.action,
            "quantity": decision.quantity,
            "price": decision.price,
            "confidence": decision.confidence,
            "reasoning": decision.reasoning,
            "agent_id": decision.agent_id
        }
        
        self.trade_history.append(trade_record)
        
        # Update positions
        if decision.action == "BUY":
            if decision.symbol in self.positions:
                self.positions[decision.symbol] += decision.quantity
            else:
                self.positions[decision.symbol] = decision.quantity
        elif decision.action == "SELL":
            if decision.symbol in self.positions:
                self.positions[decision.symbol] -= decision.quantity
        
        return trade_record
    
    def should_reflect(self) -> bool:
        """Check if it's time for self-reflection."""
        return (datetime.now() - self.last_reflection).total_seconds() >= self.reflection_interval
    
    async def run_cycle(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Run one complete cycle of the agent."""
        cycle_result = {
            "agent_id": self.agent_id,
            "timestamp": datetime.now(),
            "decisions": [],
            "reflections": [],
            "learning": []
        }
        
        # Make trading decisions
        try:
            decision = await self.make_trading_decision(market_data)
            if decision:
                trade_result = await self.execute_trade(decision)
                cycle_result["decisions"].append(trade_result)
        except Exception as e:
            self.logger.error(f"Error making trading decision: {e}")
        
        # Perform self-reflection if needed
        if self.should_reflect():
            try:
                reflection = await self.self_reflect()
                cycle_result["reflections"].append(reflection)
            except Exception as e:
                self.logger.error(f"Error in self-reflection: {e}")
        
        # Update performance history
        self.performance_history.append(self.calculate_performance_metrics())
        if len(self.performance_history) > 100:  # Keep last 100 performance records
            self.performance_history.pop(0)
        
        return cycle_result
