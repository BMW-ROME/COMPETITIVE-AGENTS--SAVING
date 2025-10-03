"""
Advanced Performance Analytics and Real-time Monitoring
"""
import asyncio
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import json
import sqlite3

class PerformanceMetric(Enum):
    """Performance metrics"""
    TOTAL_RETURN = "total_return"
    SHARPE_RATIO = "sharpe_ratio"
    SORTINO_RATIO = "sortino_ratio"
    CALMAR_RATIO = "calmar_ratio"
    MAX_DRAWDOWN = "max_drawdown"
    WIN_RATE = "win_rate"
    PROFIT_FACTOR = "profit_factor"
    VAR_95 = "var_95"
    EXPECTED_SHORTFALL = "expected_shortfall"
    INFORMATION_RATIO = "information_ratio"

@dataclass
class PerformanceSnapshot:
    """Performance snapshot at a point in time"""
    timestamp: datetime
    portfolio_value: float
    cash: float
    positions: Dict[str, float]
    total_return: float
    daily_return: float
    drawdown: float
    volatility: float
    sharpe_ratio: float

@dataclass
class AgentPerformance:
    """Individual agent performance metrics"""
    agent_id: str
    total_trades: int
    successful_trades: int
    total_profit: float
    win_rate: float
    avg_profit_per_trade: float
    max_profit: float
    max_loss: float
    sharpe_ratio: float
    max_drawdown: float
    profit_factor: float
    last_trade_time: datetime
    current_positions: Dict[str, float]

@dataclass
class SystemPerformance:
    """Overall system performance metrics"""
    timestamp: datetime
    total_portfolio_value: float
    total_return: float
    daily_return: float
    weekly_return: float
    monthly_return: float
    annualized_return: float
    volatility: float
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    max_drawdown: float
    current_drawdown: float
    win_rate: float
    profit_factor: float
    var_95: float
    expected_shortfall: float
    information_ratio: float
    agent_performance: Dict[str, AgentPerformance]
    risk_metrics: Dict[str, float]
    correlation_metrics: Dict[str, float]

class PerformanceAnalytics:
    """Advanced performance analytics engine"""
    
    def __init__(self, logger: logging.Logger, db_path: str = "trading_agents.db"):
        self.logger = logger
        self.db_path = db_path
        self.performance_history = []
        self.agent_performance_cache = {}
        self.benchmark_data = {}
        self.risk_free_rate = 0.02  # 2% annual risk-free rate
        
    async def analyze_system_performance(self, market_data: Dict[str, Any], 
                                       portfolio_data: Dict[str, Any]) -> SystemPerformance:
        """Analyze comprehensive system performance"""
        try:
            self.logger.info("Analyzing system performance...")
            
            # Get historical performance data
            historical_data = await self._get_historical_performance()
            
            # Calculate portfolio metrics
            portfolio_metrics = await self._calculate_portfolio_metrics(portfolio_data, historical_data)
            
            # Calculate risk metrics
            risk_metrics = await self._calculate_risk_metrics(historical_data)
            
            # Calculate correlation metrics
            correlation_metrics = await self._calculate_correlation_metrics(market_data)
            
            # Analyze agent performance
            agent_performance = await self._analyze_agent_performance()
            
            # Calculate advanced metrics
            advanced_metrics = await self._calculate_advanced_metrics(historical_data, portfolio_metrics)
            
            system_performance = SystemPerformance(
                timestamp=datetime.now(),
                total_portfolio_value=portfolio_metrics.get('total_value', 0),
                total_return=portfolio_metrics.get('total_return', 0),
                daily_return=portfolio_metrics.get('daily_return', 0),
                weekly_return=portfolio_metrics.get('weekly_return', 0),
                monthly_return=portfolio_metrics.get('monthly_return', 0),
                annualized_return=portfolio_metrics.get('annualized_return', 0),
                volatility=risk_metrics.get('volatility', 0),
                sharpe_ratio=advanced_metrics.get('sharpe_ratio', 0),
                sortino_ratio=advanced_metrics.get('sortino_ratio', 0),
                calmar_ratio=advanced_metrics.get('calmar_ratio', 0),
                max_drawdown=risk_metrics.get('max_drawdown', 0),
                current_drawdown=risk_metrics.get('current_drawdown', 0),
                win_rate=advanced_metrics.get('win_rate', 0),
                profit_factor=advanced_metrics.get('profit_factor', 0),
                var_95=risk_metrics.get('var_95', 0),
                expected_shortfall=risk_metrics.get('expected_shortfall', 0),
                information_ratio=advanced_metrics.get('information_ratio', 0),
                agent_performance=agent_performance,
                risk_metrics=risk_metrics,
                correlation_metrics=correlation_metrics
            )
            
            # Store performance snapshot
            self.performance_history.append(system_performance)
            
            return system_performance
            
        except Exception as e:
            self.logger.error(f"Error analyzing system performance: {e}")
            return None
    
    async def _get_historical_performance(self) -> List[Dict[str, Any]]:
        """Get historical performance data from database"""
        try:
            historical_data = []
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Get trade history
                cursor.execute("""
                    SELECT timestamp, agent_id, symbol, action, quantity, price, 
                           profit_loss, status
                    FROM trades 
                    WHERE timestamp >= datetime('now', '-30 days')
                    ORDER BY timestamp
                """)
                
                trades = cursor.fetchall()
                
                for trade in trades:
                    historical_data.append({
                        'timestamp': datetime.fromisoformat(trade[0]),
                        'agent_id': trade[1],
                        'symbol': trade[2],
                        'action': trade[3],
                        'quantity': trade[4],
                        'price': trade[5],
                        'profit_loss': trade[6] or 0,
                        'status': trade[7]
                    })
            
            return historical_data
            
        except Exception as e:
            self.logger.error(f"Error getting historical performance: {e}")
            return []
    
    async def _calculate_portfolio_metrics(self, portfolio_data: Dict[str, Any], 
                                         historical_data: List[Dict[str, Any]]) -> Dict[str, float]:
        """Calculate portfolio-level metrics"""
        try:
            metrics = {}
            
            # Current portfolio value
            total_value = portfolio_data.get('total_value', 100000)  # Default starting value
            metrics['total_value'] = total_value
            
            # Calculate returns
            if historical_data:
                # Total return
                total_profit = sum(trade['profit_loss'] for trade in historical_data if trade['status'] == 'filled')
                initial_value = 100000  # Starting value
                metrics['total_return'] = total_profit / initial_value
                
                # Daily return
                today_trades = [t for t in historical_data if t['timestamp'].date() == datetime.now().date()]
                daily_profit = sum(trade['profit_loss'] for trade in today_trades if trade['status'] == 'filled')
                metrics['daily_return'] = daily_profit / total_value
                
                # Weekly return
                week_ago = datetime.now() - timedelta(days=7)
                weekly_trades = [t for t in historical_data if t['timestamp'] >= week_ago]
                weekly_profit = sum(trade['profit_loss'] for trade in weekly_trades if trade['status'] == 'filled')
                metrics['weekly_return'] = weekly_profit / total_value
                
                # Monthly return
                month_ago = datetime.now() - timedelta(days=30)
                monthly_trades = [t for t in historical_data if t['timestamp'] >= month_ago]
                monthly_profit = sum(trade['profit_loss'] for trade in monthly_trades if trade['status'] == 'filled')
                metrics['monthly_return'] = monthly_profit / total_value
                
                # Annualized return
                days_trading = (datetime.now() - historical_data[0]['timestamp']).days if historical_data else 1
                metrics['annualized_return'] = (metrics['total_return'] * 365) / days_trading if days_trading > 0 else 0
            else:
                metrics.update({
                    'total_return': 0,
                    'daily_return': 0,
                    'weekly_return': 0,
                    'monthly_return': 0,
                    'annualized_return': 0
                })
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error calculating portfolio metrics: {e}")
            return {}
    
    async def _calculate_risk_metrics(self, historical_data: List[Dict[str, Any]]) -> Dict[str, float]:
        """Calculate risk metrics"""
        try:
            risk_metrics = {}
            
            if not historical_data:
                return {
                    'volatility': 0,
                    'max_drawdown': 0,
                    'current_drawdown': 0,
                    'var_95': 0,
                    'expected_shortfall': 0
                }
            
            # Calculate daily returns
            daily_returns = []
            daily_pnl = {}
            
            for trade in historical_data:
                if trade['status'] == 'filled':
                    date = trade['timestamp'].date()
                    if date not in daily_pnl:
                        daily_pnl[date] = 0
                    daily_pnl[date] += trade['profit_loss']
            
            # Convert to returns
            portfolio_values = [100000]  # Starting value
            for date, pnl in sorted(daily_pnl.items()):
                new_value = portfolio_values[-1] + pnl
                portfolio_values.append(new_value)
                if len(portfolio_values) > 1:
                    daily_return = (new_value - portfolio_values[-2]) / portfolio_values[-2]
                    daily_returns.append(daily_return)
            
            if daily_returns:
                # Volatility (annualized)
                risk_metrics['volatility'] = np.std(daily_returns) * np.sqrt(252)
                
                # Value at Risk (95%)
                risk_metrics['var_95'] = np.percentile(daily_returns, 5)  # 5th percentile
                
                # Expected Shortfall (Conditional VaR)
                var_threshold = risk_metrics['var_95']
                tail_returns = [r for r in daily_returns if r <= var_threshold]
                risk_metrics['expected_shortfall'] = np.mean(tail_returns) if tail_returns else var_threshold
                
                # Drawdown analysis
                peak = portfolio_values[0]
                max_drawdown = 0
                current_drawdown = 0
                
                for value in portfolio_values:
                    if value > peak:
                        peak = value
                    drawdown = (peak - value) / peak
                    max_drawdown = max(max_drawdown, drawdown)
                    if value == portfolio_values[-1]:  # Current value
                        current_drawdown = drawdown
                
                risk_metrics['max_drawdown'] = max_drawdown
                risk_metrics['current_drawdown'] = current_drawdown
            else:
                risk_metrics.update({
                    'volatility': 0,
                    'max_drawdown': 0,
                    'current_drawdown': 0,
                    'var_95': 0,
                    'expected_shortfall': 0
                })
            
            return risk_metrics
            
        except Exception as e:
            self.logger.error(f"Error calculating risk metrics: {e}")
            return {}
    
    async def _calculate_correlation_metrics(self, market_data: Dict[str, Any]) -> Dict[str, float]:
        """Calculate correlation metrics"""
        try:
            correlation_metrics = {}
            
            if not market_data:
                return {}
            
            # Calculate portfolio correlation with market
            # This is a simplified implementation
            correlation_metrics['market_correlation'] = 0.7  # Placeholder
            correlation_metrics['sector_concentration'] = 0.3  # Placeholder
            correlation_metrics['geographic_concentration'] = 0.8  # Placeholder
            
            return correlation_metrics
            
        except Exception as e:
            self.logger.error(f"Error calculating correlation metrics: {e}")
            return {}
    
    async def _analyze_agent_performance(self) -> Dict[str, AgentPerformance]:
        """Analyze individual agent performance"""
        try:
            agent_performance = {}
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Get agent performance data
                cursor.execute("""
                    SELECT agent_id, 
                           COUNT(*) as total_trades,
                           SUM(CASE WHEN status = 'filled' THEN 1 ELSE 0 END) as successful_trades,
                           SUM(CASE WHEN status = 'filled' THEN profit_loss ELSE 0 END) as total_profit,
                           MAX(CASE WHEN status = 'filled' THEN profit_loss ELSE 0 END) as max_profit,
                           MIN(CASE WHEN status = 'filled' THEN profit_loss ELSE 0 END) as max_loss,
                           MAX(timestamp) as last_trade_time
                    FROM trades 
                    WHERE timestamp >= datetime('now', '-30 days')
                    GROUP BY agent_id
                """)
                
                agent_data = cursor.fetchall()
                
                for row in agent_data:
                    agent_id = row[0]
                    total_trades = row[1]
                    successful_trades = row[2]
                    total_profit = row[3] or 0
                    max_profit = row[4] or 0
                    max_loss = row[5] or 0
                    last_trade_time = datetime.fromisoformat(row[6]) if row[6] else datetime.now()
                    
                    # Calculate metrics
                    win_rate = (successful_trades / total_trades) if total_trades > 0 else 0
                    avg_profit_per_trade = (total_profit / total_trades) if total_trades > 0 else 0
                    
                    # Simplified Sharpe ratio calculation
                    sharpe_ratio = (avg_profit_per_trade - self.risk_free_rate/252) / (np.std([max_profit, max_loss]) if max_profit != max_loss else 0.1) if total_trades > 0 else 0
                    
                    # Simplified max drawdown (would need more data for accurate calculation)
                    max_drawdown = abs(max_loss) / 1000 if max_loss < 0 else 0  # Simplified
                    
                    # Profit factor
                    profitable_trades = [t for t in self._get_agent_trades(agent_id) if t.get('profit_loss', 0) > 0]
                    losing_trades = [t for t in self._get_agent_trades(agent_id) if t.get('profit_loss', 0) < 0]
                    
                    gross_profit = sum(t.get('profit_loss', 0) for t in profitable_trades)
                    gross_loss = abs(sum(t.get('profit_loss', 0) for t in losing_trades))
                    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf') if gross_profit > 0 else 0
                    
                    agent_performance[agent_id] = AgentPerformance(
                        agent_id=agent_id,
                        total_trades=total_trades,
                        successful_trades=successful_trades,
                        total_profit=total_profit,
                        win_rate=win_rate,
                        avg_profit_per_trade=avg_profit_per_trade,
                        max_profit=max_profit,
                        max_loss=max_loss,
                        sharpe_ratio=sharpe_ratio,
                        max_drawdown=max_drawdown,
                        profit_factor=profit_factor,
                        last_trade_time=last_trade_time,
                        current_positions={}  # Would need to get from portfolio data
                    )
            
            return agent_performance
            
        except Exception as e:
            self.logger.error(f"Error analyzing agent performance: {e}")
            return {}
    
    def _get_agent_trades(self, agent_id: str) -> List[Dict[str, Any]]:
        """Get trades for a specific agent"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    SELECT profit_loss FROM trades 
                    WHERE agent_id = ? AND status = 'filled'
                """, (agent_id,))
                
                trades = cursor.fetchall()
                return [{'profit_loss': trade[0] or 0} for trade in trades]
                
        except Exception as e:
            self.logger.error(f"Error getting agent trades: {e}")
            return []
    
    async def _calculate_advanced_metrics(self, historical_data: List[Dict[str, Any]], 
                                        portfolio_metrics: Dict[str, float]) -> Dict[str, float]:
        """Calculate advanced performance metrics"""
        try:
            advanced_metrics = {}
            
            if not historical_data:
                return {
                    'sharpe_ratio': 0,
                    'sortino_ratio': 0,
                    'calmar_ratio': 0,
                    'win_rate': 0,
                    'profit_factor': 0,
                    'information_ratio': 0
                }
            
            # Calculate daily returns
            daily_returns = []
            daily_pnl = {}
            
            for trade in historical_data:
                if trade['status'] == 'filled':
                    date = trade['timestamp'].date()
                    if date not in daily_pnl:
                        daily_pnl[date] = 0
                    daily_pnl[date] += trade['profit_loss']
            
            # Convert to returns
            portfolio_values = [100000]  # Starting value
            for date, pnl in sorted(daily_pnl.items()):
                new_value = portfolio_values[-1] + pnl
                portfolio_values.append(new_value)
                if len(portfolio_values) > 1:
                    daily_return = (new_value - portfolio_values[-2]) / portfolio_values[-2]
                    daily_returns.append(daily_return)
            
            if daily_returns:
                # Sharpe Ratio
                excess_returns = [r - self.risk_free_rate/252 for r in daily_returns]
                advanced_metrics['sharpe_ratio'] = np.mean(excess_returns) / np.std(daily_returns) if np.std(daily_returns) > 0 else 0
                
                # Sortino Ratio (downside deviation)
                downside_returns = [r for r in daily_returns if r < 0]
                downside_deviation = np.std(downside_returns) if downside_returns else 0
                advanced_metrics['sortino_ratio'] = np.mean(excess_returns) / downside_deviation if downside_deviation > 0 else 0
                
                # Calmar Ratio
                annualized_return = portfolio_metrics.get('annualized_return', 0)
                max_drawdown = max([(max(portfolio_values[:i+1]) - portfolio_values[i]) / max(portfolio_values[:i+1]) for i in range(len(portfolio_values))]) if portfolio_values else 0
                advanced_metrics['calmar_ratio'] = annualized_return / max_drawdown if max_drawdown > 0 else 0
                
                # Win Rate
                winning_days = len([r for r in daily_returns if r > 0])
                advanced_metrics['win_rate'] = winning_days / len(daily_returns) if daily_returns else 0
                
                # Profit Factor
                gross_profit = sum([r for r in daily_returns if r > 0])
                gross_loss = abs(sum([r for r in daily_returns if r < 0]))
                advanced_metrics['profit_factor'] = gross_profit / gross_loss if gross_loss > 0 else float('inf') if gross_profit > 0 else 0
                
                # Information Ratio (simplified - would need benchmark)
                benchmark_return = 0.05 / 252  # 5% annual benchmark
                active_returns = [r - benchmark_return for r in daily_returns]
                tracking_error = np.std(active_returns)
                advanced_metrics['information_ratio'] = np.mean(active_returns) / tracking_error if tracking_error > 0 else 0
            else:
                advanced_metrics.update({
                    'sharpe_ratio': 0,
                    'sortino_ratio': 0,
                    'calmar_ratio': 0,
                    'win_rate': 0,
                    'profit_factor': 0,
                    'information_ratio': 0
                })
            
            return advanced_metrics
            
        except Exception as e:
            self.logger.error(f"Error calculating advanced metrics: {e}")
            return {}
    
    async def generate_performance_report(self, system_performance: SystemPerformance) -> Dict[str, Any]:
        """Generate comprehensive performance report"""
        try:
            report = {
                'timestamp': system_performance.timestamp.isoformat(),
                'summary': {
                    'total_return': f"{system_performance.total_return:.2%}",
                    'annualized_return': f"{system_performance.annualized_return:.2%}",
                    'sharpe_ratio': f"{system_performance.sharpe_ratio:.2f}",
                    'max_drawdown': f"{system_performance.max_drawdown:.2%}",
                    'win_rate': f"{system_performance.win_rate:.2%}",
                    'profit_factor': f"{system_performance.profit_factor:.2f}"
                },
                'risk_metrics': {
                    'volatility': f"{system_performance.volatility:.2%}",
                    'var_95': f"{system_performance.var_95:.2%}",
                    'expected_shortfall': f"{system_performance.expected_shortfall:.2%}",
                    'current_drawdown': f"{system_performance.current_drawdown:.2%}"
                },
                'agent_rankings': self._rank_agents(system_performance.agent_performance),
                'recommendations': self._generate_recommendations(system_performance),
                'performance_grade': self._calculate_performance_grade(system_performance)
            }
            
            return report
            
        except Exception as e:
            self.logger.error(f"Error generating performance report: {e}")
            return {}
    
    def _rank_agents(self, agent_performance: Dict[str, AgentPerformance]) -> List[Dict[str, Any]]:
        """Rank agents by performance"""
        try:
            rankings = []
            
            for agent_id, perf in agent_performance.items():
                # Calculate composite score
                score = (
                    perf.win_rate * 0.3 +
                    (perf.total_profit / 1000) * 0.3 +  # Normalize profit
                    perf.sharpe_ratio * 0.2 +
                    (1 - perf.max_drawdown) * 0.2  # Lower drawdown is better
                )
                
                rankings.append({
                    'agent_id': agent_id,
                    'score': score,
                    'total_profit': perf.total_profit,
                    'win_rate': perf.win_rate,
                    'sharpe_ratio': perf.sharpe_ratio,
                    'total_trades': perf.total_trades
                })
            
            # Sort by score
            rankings.sort(key=lambda x: x['score'], reverse=True)
            
            return rankings
            
        except Exception as e:
            self.logger.error(f"Error ranking agents: {e}")
            return []
    
    def _generate_recommendations(self, system_performance: SystemPerformance) -> List[str]:
        """Generate performance recommendations"""
        try:
            recommendations = []
            
            # Sharpe ratio recommendations
            if system_performance.sharpe_ratio < 1.0:
                recommendations.append("LOW_SHARPE: Consider improving risk-adjusted returns")
            elif system_performance.sharpe_ratio > 2.0:
                recommendations.append("EXCELLENT_SHARPE: Risk-adjusted returns are excellent")
            
            # Drawdown recommendations
            if system_performance.max_drawdown > 0.15:
                recommendations.append("HIGH_DRAWDOWN: Consider reducing position sizes or improving risk management")
            
            # Win rate recommendations
            if system_performance.win_rate < 0.4:
                recommendations.append("LOW_WIN_RATE: Consider improving trade selection criteria")
            elif system_performance.win_rate > 0.7:
                recommendations.append("HIGH_WIN_RATE: Trade selection is performing well")
            
            # Volatility recommendations
            if system_performance.volatility > 0.3:
                recommendations.append("HIGH_VOLATILITY: Consider diversifying or reducing risk")
            
            # Agent performance recommendations
            if system_performance.agent_performance:
                best_agent = max(system_performance.agent_performance.items(), key=lambda x: x[1].total_profit)
                worst_agent = min(system_performance.agent_performance.items(), key=lambda x: x[1].total_profit)
                
                recommendations.append(f"BEST_AGENT: {best_agent[0]} is performing best")
                recommendations.append(f"WORST_AGENT: {worst_agent[0]} needs improvement")
            
            return recommendations
            
        except Exception as e:
            self.logger.error(f"Error generating recommendations: {e}")
            return []
    
    def _calculate_performance_grade(self, system_performance: SystemPerformance) -> str:
        """Calculate overall performance grade"""
        try:
            score = 0
            
            # Sharpe ratio (40% weight)
            if system_performance.sharpe_ratio > 2.0:
                score += 40
            elif system_performance.sharpe_ratio > 1.5:
                score += 30
            elif system_performance.sharpe_ratio > 1.0:
                score += 20
            elif system_performance.sharpe_ratio > 0.5:
                score += 10
            
            # Win rate (30% weight)
            if system_performance.win_rate > 0.7:
                score += 30
            elif system_performance.win_rate > 0.6:
                score += 25
            elif system_performance.win_rate > 0.5:
                score += 20
            elif system_performance.win_rate > 0.4:
                score += 10
            
            # Drawdown (20% weight)
            if system_performance.max_drawdown < 0.05:
                score += 20
            elif system_performance.max_drawdown < 0.10:
                score += 15
            elif system_performance.max_drawdown < 0.15:
                score += 10
            elif system_performance.max_drawdown < 0.20:
                score += 5
            
            # Return (10% weight)
            if system_performance.annualized_return > 0.20:
                score += 10
            elif system_performance.annualized_return > 0.15:
                score += 8
            elif system_performance.annualized_return > 0.10:
                score += 6
            elif system_performance.annualized_return > 0.05:
                score += 4
            
            # Convert to letter grade
            if score >= 90:
                return "A+"
            elif score >= 80:
                return "A"
            elif score >= 70:
                return "B+"
            elif score >= 60:
                return "B"
            elif score >= 50:
                return "C+"
            elif score >= 40:
                return "C"
            else:
                return "D"
                
        except Exception as e:
            self.logger.error(f"Error calculating performance grade: {e}")
            return "F"
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary"""
        try:
            if not self.performance_history:
                return {}
            
            latest_performance = self.performance_history[-1]
            
            return {
                'latest_performance': {
                    'timestamp': latest_performance.timestamp.isoformat(),
                    'total_return': latest_performance.total_return,
                    'sharpe_ratio': latest_performance.sharpe_ratio,
                    'max_drawdown': latest_performance.max_drawdown,
                    'win_rate': latest_performance.win_rate
                },
                'performance_trend': self._calculate_performance_trend(),
                'total_snapshots': len(self.performance_history),
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Error getting performance summary: {e}")
            return {}
    
    def _calculate_performance_trend(self) -> str:
        """Calculate performance trend"""
        try:
            if len(self.performance_history) < 2:
                return "insufficient_data"
            
            recent_performance = self.performance_history[-1]
            previous_performance = self.performance_history[-2]
            
            if recent_performance.sharpe_ratio > previous_performance.sharpe_ratio:
                return "improving"
            elif recent_performance.sharpe_ratio < previous_performance.sharpe_ratio:
                return "declining"
            else:
                return "stable"
                
        except Exception as e:
            self.logger.error(f"Error calculating performance trend: {e}")
            return "unknown"


