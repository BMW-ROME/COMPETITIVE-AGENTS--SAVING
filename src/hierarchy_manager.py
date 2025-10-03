"""
Simplified Hierarchy Manager for overseeing competitive trading agents.
"""
import asyncio
import logging
import json
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
import pandas as pd

from src.base_agent import BaseTradingAgent, PerformanceMetrics, TradeDecision
from src.persistence import SQLitePersistence
from config.settings import HierarchyConfig, SystemConfig

@dataclass
class AgentReport:
    """Report from an agent to the hierarchy manager."""
    agent_id: str
    timestamp: datetime
    performance_metrics: PerformanceMetrics
    recent_decisions: List[Dict[str, Any]]
    reflections: List[Dict[str, Any]]
    learning_insights: List[Dict[str, Any]]
    status: str  # "active", "learning", "reflecting", "error"

class HierarchyManager:
    """
    Simplified manager for competitive trading agents.
    """
    
    def __init__(self, config: HierarchyConfig, system_config: SystemConfig):
        self.config = config
        self.system_config = system_config
        self.agents: Dict[str, BaseTradingAgent] = {}
        self.agent_reports: Dict[str, List[AgentReport]] = {}
        self.performance_history: Dict[str, List[PerformanceMetrics]] = {}
        # Suspension state
        self.fail_streaks: Dict[str, int] = {}
        self.suspension_levels: Dict[str, int] = {}
        self.suspended: Dict[str, Dict[str, Any]] = {}
        
        # Logging
        self.logger = logging.getLogger("HierarchyManager")

        # Persistence
        self.db: Optional[SQLitePersistence] = None
        
        # Performance tracking
        self.overall_performance = {
            "total_return": 0.0,
            "best_agent": None,
            "worst_agent": None,
            "competition_intensity": 0.0
        }
    
    def register_agent(self, agent: BaseTradingAgent) -> None:
        """Register an agent with the hierarchy manager."""
        self.agents[agent.agent_id] = agent
        self.agent_reports[agent.agent_id] = []
        self.performance_history[agent.agent_id] = []
        self.fail_streaks[agent.agent_id] = 0
        self.suspension_levels[agent.agent_id] = 0
        self.logger.info(f"Registered agent: {agent.agent_id}")

    def attach_persistence(self, persistence: SQLitePersistence) -> None:
        self.db = persistence
    
    async def collect_agent_reports(self) -> Dict[str, AgentReport]:
        """Collect reports from all registered agents."""
        reports = {}
        
        for agent_id, agent in self.agents.items():
            try:
                # Get current performance metrics
                performance_metrics = agent.calculate_performance_metrics()
                
                # Get recent decisions (last 10)
                recent_decisions = agent.trade_history[-10:] if len(agent.trade_history) >= 10 else agent.trade_history
                
                # Get recent reflections (last 5)
                recent_reflections = agent.learning_memory[-5:] if len(agent.learning_memory) >= 5 else agent.learning_memory
                
                # Get learning insights
                learning_insights = [reflection for reflection in recent_reflections if "learning_insights" in reflection]
                
                # Determine agent status
                status = "active"
                if agent.should_reflect():
                    status = "reflecting"
                elif len(agent.learning_memory) > 0:
                    latest_learning = agent.learning_memory[-1]
                    if "learning_opportunities" in latest_learning:
                        status = "learning"
                
                report = AgentReport(
                    agent_id=agent_id,
                    timestamp=datetime.now(),
                    performance_metrics=performance_metrics,
                    recent_decisions=recent_decisions,
                    reflections=recent_reflections,
                    learning_insights=learning_insights,
                    status=status
                )
                
                reports[agent_id] = report
                self.agent_reports[agent_id].append(report)
                
                # Keep only last 100 reports per agent
                if len(self.agent_reports[agent_id]) > 100:
                    self.agent_reports[agent_id].pop(0)
                
            except Exception as e:
                self.logger.error(f"Error collecting report from agent {agent_id}: {e}")
                reports[agent_id] = None
        
        return reports
    
    async def evaluate_agent_performance(self, reports: Dict[str, AgentReport]) -> Dict[str, Any]:
        """Evaluate and compare agent performance."""
        evaluation = {
            "rankings": {},
            "performance_comparison": {},
            "competition_analysis": {},
            "recommendations": {}
        }
        
        # Calculate rankings based on multiple metrics
        rankings = self._calculate_rankings(reports)
        evaluation["rankings"] = rankings
        
        # Performance comparison
        performance_comparison = self._compare_performance(reports)
        evaluation["performance_comparison"] = performance_comparison
        
        # Competition analysis
        competition_analysis = self._analyze_competition(reports)
        evaluation["competition_analysis"] = competition_analysis
        
        # Generate recommendations
        recommendations = self._generate_recommendations(reports, rankings)
        evaluation["recommendations"] = recommendations
        
        return evaluation
    
    def _calculate_rankings(self, reports: Dict[str, AgentReport]) -> Dict[str, int]:
        """Calculate agent rankings based on multiple performance metrics."""
        if len(reports) < 2:
            return {agent_id: 1 for agent_id in reports.keys()}
        
        # Define ranking criteria with weights
        criteria_weights = {
            "total_return": 0.3,
            "sharpe_ratio": 0.25,
            "win_rate": 0.2,
            "max_drawdown": 0.15,  # Lower is better
            "profit_factor": 0.1
        }
        
        agent_scores = {}
        
        for agent_id, report in reports.items():
            if report is None:
                continue
            
            metrics = report.performance_metrics
            score = 0
            
            # Calculate weighted score
            score += metrics.total_return * criteria_weights["total_return"]
            score += metrics.sharpe_ratio * criteria_weights["sharpe_ratio"]
            score += metrics.win_rate * criteria_weights["win_rate"]
            score += (1 - abs(metrics.max_drawdown)) * criteria_weights["max_drawdown"]  # Invert drawdown
            score += min(metrics.profit_factor, 5) * criteria_weights["profit_factor"]  # Cap profit factor
            
            agent_scores[agent_id] = score
        
        # Sort by score and assign rankings
        sorted_agents = sorted(agent_scores.items(), key=lambda x: x[1], reverse=True)
        rankings = {agent_id: rank + 1 for rank, (agent_id, _) in enumerate(sorted_agents)}
        
        return rankings
    
    def _compare_performance(self, reports: Dict[str, AgentReport]) -> Dict[str, Any]:
        """Compare performance between agents."""
        comparison = {
            "best_performer": None,
            "worst_performer": None,
            "performance_gaps": {},
            "strengths_weaknesses": {}
        }
        
        valid_reports = {k: v for k, v in reports.items() if v is not None}
        if len(valid_reports) < 2:
            return comparison
        
        # Find best and worst performers
        best_score = float('-inf')
        worst_score = float('inf')
        
        for agent_id, report in valid_reports.items():
            metrics = report.performance_metrics
            score = metrics.total_return + metrics.sharpe_ratio + metrics.win_rate
            
            if score > best_score:
                best_score = score
                comparison["best_performer"] = agent_id
            
            if score < worst_score:
                worst_score = score
                comparison["worst_performer"] = agent_id
        
        # Calculate performance gaps
        if comparison["best_performer"] and comparison["worst_performer"]:
            best_metrics = valid_reports[comparison["best_performer"]].performance_metrics
            worst_metrics = valid_reports[comparison["worst_performer"]].performance_metrics
            
            comparison["performance_gaps"] = {
                "return_gap": best_metrics.total_return - worst_metrics.total_return,
                "sharpe_gap": best_metrics.sharpe_ratio - worst_metrics.sharpe_ratio,
                "win_rate_gap": best_metrics.win_rate - worst_metrics.win_rate
            }
        
        return comparison
    
    def _analyze_competition(self, reports: Dict[str, AgentReport]) -> Dict[str, Any]:
        """Analyze the competitive dynamics between agents."""
        competition_analysis = {
            "intensity": 0.0,
            "leader_changes": 0,
            "performance_convergence": False,
            "competitive_advantages": {}
        }
        
        valid_reports = {k: v for k, v in reports.items() if v is not None}
        if len(valid_reports) < 2:
            return competition_analysis
        
        # Calculate competition intensity
        scores = []
        for report in valid_reports.values():
            metrics = report.performance_metrics
            score = metrics.total_return + metrics.sharpe_ratio + metrics.win_rate
            scores.append(score)
        
        if len(scores) > 1:
            score_std = np.std(scores)
            competition_analysis["intensity"] = min(1.0, score_std * 2)  # Normalize to 0-1
        
        # Check for performance convergence
        if len(scores) > 1:
            score_range = max(scores) - min(scores)
            competition_analysis["performance_convergence"] = score_range < 0.1
        
        return competition_analysis
    
    def _generate_recommendations(self, reports: Dict[str, AgentReport], rankings: Dict[str, int]) -> Dict[str, List[str]]:
        """Generate recommendations for each agent."""
        recommendations = {}
        
        for agent_id, report in reports.items():
            if report is None:
                continue
            
            agent_recommendations = []
            metrics = report.performance_metrics
            ranking = rankings.get(agent_id, len(rankings))
            
            # Performance-based recommendations
            if metrics.sharpe_ratio < 1.0:
                agent_recommendations.append("Improve risk-adjusted returns by reducing position sizes or improving entry timing")
            
            if metrics.win_rate < 0.6:
                agent_recommendations.append("Focus on trade accuracy and signal quality")
            
            if abs(metrics.max_drawdown) > 0.05:
                agent_recommendations.append("Implement stricter risk management and stop-loss rules")
            
            # Ranking-based recommendations
            if ranking > 1:
                agent_recommendations.append(f"Analyze the strategies of higher-ranked agents for learning opportunities")
            
            recommendations[agent_id] = agent_recommendations
        
        return recommendations
    
    async def run_oversight_cycle(self) -> Dict[str, Any]:
        """Run one complete oversight cycle."""
        cycle_result = {
            "timestamp": datetime.now(),
            "reports_collected": 0,
            "evaluation": {},
            "decisions_made": [],
            "communications_sent": 0
        }
        
        try:
            # Collect agent reports
            reports = await self.collect_agent_reports()
            cycle_result["reports_collected"] = len([r for r in reports.values() if r is not None])
            
            # Evaluate performance
            evaluation = await self.evaluate_agent_performance(reports)
            cycle_result["evaluation"] = evaluation
            
            # Update overall performance tracking
            self._update_overall_performance(evaluation)

            # Apply suspensions or lifts based on streaks and system profitability milestones
            self._apply_suspension_policy()
            
        except Exception as e:
            self.logger.error(f"Error in oversight cycle: {e}")
            cycle_result["error"] = str(e)
        
        return cycle_result

    def record_trade_outcome(self, agent_id: str, trade_return: float, system_profitable_trades: int) -> None:
        """Update fail streaks and manage suspension lift counters."""
        try:
            if trade_return is None:
                return
            # Update fail streaks
            if trade_return <= 0:
                self.fail_streaks[agent_id] = self.fail_streaks.get(agent_id, 0) + 1
            else:
                self.fail_streaks[agent_id] = 0

            # Decrement suspension lift counters for ALL suspended agents on each profitable system trade
            if trade_return > 0 and self.suspended:
                for suspended_agent_id, info in list(self.suspended.items()):
                    remaining = max(0, info.get("profit_trades_remaining", 0) - 1)
                    info["profit_trades_remaining"] = remaining
                    self.suspended[suspended_agent_id] = info
                    if self.db:
                        self.db.log_suspension_event(
                            suspended_agent_id,
                            "lift_progress",
                            info.get("level", 0),
                            reason="system_profitable_trade",
                            profit_trades_required=info.get("profit_trades_required", 0),
                            profit_trades_remaining=remaining,
                        )
        except Exception as e:
            self.logger.warning(f"record_trade_outcome error: {e}")

    def _apply_suspension_policy(self) -> None:
        """Suspend agents with 2 consecutive losses; lift after 4,8,12... profitable system trades."""
        try:
            for agent_id in list(self.agents.keys()):
                # Trigger suspension
                if self.fail_streaks.get(agent_id, 0) >= 2 and agent_id not in self.suspended:
                    level = self.suspension_levels.get(agent_id, 0) + 1
                    required = 4 * level
                    self.suspended[agent_id] = {
                        "level": level,
                        "profit_trades_required": required,
                        "profit_trades_remaining": required,
                        "since": datetime.now(),
                    }
                    self.suspension_levels[agent_id] = level
                    self.logger.warning(f"Agent {agent_id} suspended at level {level} after {self.fail_streaks.get(agent_id)} consecutive losses")
                    if self.db:
                        self.db.log_suspension_event(agent_id, "suspend", level, reason="two_consecutive_losses", profit_trades_required=required, profit_trades_remaining=required)

                # Lift suspension when remaining hits zero
                if agent_id in self.suspended:
                    info = self.suspended[agent_id]
                    if info.get("profit_trades_remaining", 1) <= 0:
                        self.logger.info(f"Lifting suspension for {agent_id} at level {info.get('level')}")
                        if self.db:
                            self.db.log_suspension_event(agent_id, "lift", info.get("level", 0), reason="milestone_reached")
                        self.suspended.pop(agent_id, None)
                        # Reset fail streak to avoid immediate re-suspension
                        self.fail_streaks[agent_id] = 0
        except Exception as e:
            self.logger.warning(f"_apply_suspension_policy error: {e}")
    
    def _update_overall_performance(self, evaluation: Dict[str, Any]) -> None:
        """Update overall system performance metrics."""
        rankings = evaluation.get("rankings", {})
        
        if rankings:
            # Find best performing agent
            best_agent = min(rankings.items(), key=lambda x: x[1])[0]
            self.overall_performance["best_agent"] = best_agent
            
            # Find worst performing agent
            worst_agent = max(rankings.items(), key=lambda x: x[1])[0]
            self.overall_performance["worst_agent"] = worst_agent
        
        # Update competition intensity
        competition_analysis = evaluation.get("competition_analysis", {})
        self.overall_performance["competition_intensity"] = competition_analysis.get("intensity", 0)
        
        # Calculate total system return
        total_return = 0
        for agent_id, agent in self.agents.items():
            metrics = agent.calculate_performance_metrics()
            total_return += metrics.total_return
        
        self.overall_performance["total_return"] = total_return
    
    async def get_system_status(self) -> Dict[str, Any]:
        """Get current system status and performance summary."""
        return {
            "timestamp": datetime.now(),
            "total_agents": len(self.agents),
            "active_agents": len([a for a in self.agents.values()]),
            "overall_performance": self.overall_performance,
            "competition_metrics": {}
        }
