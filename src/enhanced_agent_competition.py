#!/usr/bin/env python3
"""
Enhanced Agent Competition System
=================================
Implements competitive trading with:
- All agents making decisions every cycle
- Continuous learning and reflections
- Quick trades (scalping)
- Hierarchy-based agent selection
- Performance-based agent scaling
"""

import logging
import random
import numpy as np
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass

@dataclass
class AgentPerformance:
    """Track agent performance metrics"""
    agent_id: str
    total_trades: int = 0
    successful_trades: int = 0
    total_pnl: float = 0.0
    win_rate: float = 0.0
    avg_trade_duration: float = 0.0
    last_decision_time: Optional[datetime] = None
    confidence_score: float = 0.5
    learning_rate: float = 0.1

class EnhancedAgentCompetition:
    """Enhanced competitive trading system"""
    
    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self.agents = {}
        self.agent_performance = {}
        self.hierarchy_manager = HierarchyManager(logger)
        self.quick_trade_agents = ['scalping_1', 'momentum_1', 'arbitrage_1']
        self.learning_agents = ['ai_enhanced_1', 'ml_pattern_1', 'adaptive_1']
        self.cycle_count = 0
        
    def initialize_agents(self):
        """Initialize all 12 agents with distinct styles"""
        agent_configs = {
            # Conservative Agents
            'conservative_1': {
                'style': 'conservative',
                'max_position': 0.02,
                'confidence_threshold': 0.8,
                'trade_frequency': 'low',
                'hold_duration': 'long'
            },
            'balanced_1': {
                'style': 'balanced',
                'max_position': 0.05,
                'confidence_threshold': 0.6,
                'trade_frequency': 'medium',
                'hold_duration': 'medium'
            },
            'aggressive_1': {
                'style': 'aggressive',
                'max_position': 0.08,
                'confidence_threshold': 0.4,
                'trade_frequency': 'high',
                'hold_duration': 'short'
            },
            # Technical Analysis Agents
            'fractal_1': {
                'style': 'fractal_analysis',
                'max_position': 0.04,
                'confidence_threshold': 0.7,
                'trade_frequency': 'medium',
                'hold_duration': 'medium'
            },
            'candle_range_1': {
                'style': 'candle_range',
                'max_position': 0.03,
                'confidence_threshold': 0.6,
                'trade_frequency': 'high',
                'hold_duration': 'short'
            },
            'quant_pattern_1': {
                'style': 'quantitative',
                'max_position': 0.06,
                'confidence_threshold': 0.5,
                'trade_frequency': 'medium',
                'hold_duration': 'medium'
            },
            # Quick Trade Agents (Scalping)
            'scalping_1': {
                'style': 'scalping',
                'max_position': 0.02,
                'confidence_threshold': 0.6,
                'trade_frequency': 'very_high',
                'hold_duration': 'very_short'
            },
            'momentum_1': {
                'style': 'momentum',
                'max_position': 0.05,
                'confidence_threshold': 0.5,
                'trade_frequency': 'high',
                'hold_duration': 'short'
            },
            # AI Enhanced Agents
            'ai_enhanced_1': {
                'style': 'ai_enhanced',
                'max_position': 0.07,
                'confidence_threshold': 0.6,
                'trade_frequency': 'medium',
                'hold_duration': 'medium'
            },
            'ml_pattern_1': {
                'style': 'ml_pattern',
                'max_position': 0.04,
                'confidence_threshold': 0.7,
                'trade_frequency': 'medium',
                'hold_duration': 'medium'
            },
            # Specialized Agents
            'arbitrage_1': {
                'style': 'arbitrage',
                'max_position': 0.03,
                'confidence_threshold': 0.8,
                'trade_frequency': 'high',
                'hold_duration': 'very_short'
            },
            'adaptive_1': {
                'style': 'adaptive',
                'max_position': 0.05,
                'confidence_threshold': 0.5,
                'trade_frequency': 'variable',
                'hold_duration': 'variable'
            }
        }
        
        for agent_id, config in agent_configs.items():
            self.agents[agent_id] = config
            self.agent_performance[agent_id] = AgentPerformance(agent_id)
            
        self.logger.info(f"Initialized {len(self.agents)} competitive agents")
    
    def run_competitive_cycle(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Run a competitive trading cycle with all agents"""
        self.cycle_count += 1
        cycle_results = {
            'cycle': self.cycle_count,
            'timestamp': datetime.now(),
            'agent_decisions': {},
            'agent_reflections': {},
            'hierarchy_selection': {},
            'total_decisions': 0,
            'total_reflections': 0
        }
        
        # Force all agents to make decisions
        for agent_id in self.agents.keys():
            try:
                # Generate decision
                decision = self._force_agent_decision(agent_id, market_data)
                cycle_results['agent_decisions'][agent_id] = decision
                
                if decision:
                    cycle_results['total_decisions'] += 1
                
                # Force reflection
                reflection = self._force_agent_reflection(agent_id, decision, market_data)
                cycle_results['agent_reflections'][agent_id] = reflection
                
                if reflection:
                    cycle_results['total_reflections'] += 1
                    
            except Exception as e:
                self.logger.error(f"Error in agent {agent_id}: {e}")
        
        # Hierarchy selection
        selected_trades = self.hierarchy_manager.select_trades(
            cycle_results['agent_decisions'], 
            self.agent_performance
        )
        cycle_results['hierarchy_selection'] = selected_trades
        
        # Update performance metrics
        self._update_agent_performance(cycle_results)
        
        # Log cycle summary
        self._log_cycle_summary(cycle_results)
        
        return cycle_results
    
    def _force_agent_decision(self, agent_id: str, market_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Force agent to make a decision based on their style"""
        agent_config = self.agents[agent_id]
        performance = self.agent_performance[agent_id]
        
        # Calculate agent confidence based on performance
        confidence = self._calculate_agent_confidence(agent_id)
        
        # Force decision based on agent style
        if agent_config['style'] == 'scalping':
            return self._generate_scalping_decision(agent_id, market_data, confidence)
        elif agent_config['style'] == 'momentum':
            return self._generate_momentum_decision(agent_id, market_data, confidence)
        elif agent_config['style'] == 'arbitrage':
            return self._generate_arbitrage_decision(agent_id, market_data, confidence)
        elif agent_config['style'] == 'ai_enhanced':
            return self._generate_ai_decision(agent_id, market_data, confidence)
        else:
            return self._generate_standard_decision(agent_id, market_data, confidence)
    
    def _force_agent_reflection(self, agent_id: str, decision: Optional[Dict[str, Any]], 
                              market_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Force agent to reflect on their performance"""
        performance = self.agent_performance[agent_id]
        
        # Always reflect if enough time has passed
        if (performance.last_decision_time is None or 
            (datetime.now() - performance.last_decision_time).total_seconds() > 300):  # 5 minutes
            
            reflection = {
                'agent_id': agent_id,
                'timestamp': datetime.now(),
                'performance_analysis': self._analyze_agent_performance(agent_id),
                'learning_insights': self._generate_learning_insights(agent_id),
                'strategy_adjustments': self._suggest_strategy_adjustments(agent_id),
                'confidence_update': self._update_confidence(agent_id)
            }
            
            # Update agent performance
            performance.last_decision_time = datetime.now()
            performance.confidence_score = reflection['confidence_update']
            
            return reflection
        
        return None
    
    def _generate_scalping_decision(self, agent_id: str, market_data: Dict[str, Any], 
                                  confidence: float) -> Optional[Dict[str, Any]]:
        """Generate scalping decision for quick trades"""
        symbols = ['SPY', 'QQQ', 'AAPL', 'MSFT']  # High liquidity for scalping
        
        for symbol in symbols:
            if symbol in market_data:
                price = market_data[symbol].get('price', 0)
                volume = market_data[symbol].get('volume', 0)
                
                # Scalping criteria: high volume, small price movements
                if volume > 1000000 and random.random() < 0.3:  # 30% chance
                    return {
                        'agent_id': agent_id,
                        'symbol': symbol,
                        'action': 'BUY',
                        'quantity': random.uniform(1, 5),  # Small quantities
                        'price': price,
                        'confidence': confidence,
                        'reasoning': 'Scalping opportunity',
                        'hold_duration': 'very_short',  # Minutes
                        'stop_loss': price * 0.995,  # 0.5% stop loss
                        'take_profit': price * 1.005,  # 0.5% take profit
                        'style': 'scalping'
                    }
        
        return None
    
    def _generate_momentum_decision(self, agent_id: str, market_data: Dict[str, Any], 
                                  confidence: float) -> Optional[Dict[str, Any]]:
        """Generate momentum decision"""
        symbols = ['TSLA', 'GOOGL', 'AAPL', 'MSFT']
        
        for symbol in symbols:
            if symbol in market_data:
                price = market_data[symbol].get('price', 0)
                price_change = market_data[symbol].get('price_change', 0)
                
                # Momentum criteria: significant price movement
                if abs(price_change) > 0.02 and random.random() < 0.4:  # 40% chance
                    action = 'BUY' if price_change > 0 else 'SELL'
                    return {
                        'agent_id': agent_id,
                        'symbol': symbol,
                        'action': action,
                        'quantity': random.uniform(2, 8),
                        'price': price,
                        'confidence': confidence,
                        'reasoning': f'Momentum {action.lower()}',
                        'hold_duration': 'short',
                        'stop_loss': price * (0.98 if action == 'BUY' else 1.02),
                        'take_profit': price * (1.03 if action == 'BUY' else 0.97),
                        'style': 'momentum'
                    }
        
        return None
    
    def _generate_arbitrage_decision(self, agent_id: str, market_data: Dict[str, Any], 
                                   confidence: float) -> Optional[Dict[str, Any]]:
        """Generate arbitrage decision"""
        # Look for price discrepancies between related assets
        spy_price = market_data.get('SPY', {}).get('price', 0)
        qqq_price = market_data.get('QQQ', {}).get('price', 0)
        
        if spy_price > 0 and qqq_price > 0:
            ratio = spy_price / qqq_price
            # Arbitrage opportunity if ratio is outside normal range
            if ratio < 0.8 or ratio > 1.2:
                return {
                    'agent_id': agent_id,
                    'symbol': 'SPY' if ratio < 0.8 else 'QQQ',
                    'action': 'BUY',
                    'quantity': random.uniform(1, 3),
                    'price': spy_price if ratio < 0.8 else qqq_price,
                    'confidence': confidence,
                    'reasoning': 'Arbitrage opportunity',
                    'hold_duration': 'very_short',
                    'stop_loss': (spy_price if ratio < 0.8 else qqq_price) * 0.99,
                    'take_profit': (spy_price if ratio < 0.8 else qqq_price) * 1.01,
                    'style': 'arbitrage'
                }
        
        return None
    
    def _generate_ai_decision(self, agent_id: str, market_data: Dict[str, Any], 
                            confidence: float) -> Optional[Dict[str, Any]]:
        """Generate AI-enhanced decision"""
        symbols = ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'SPY', 'QQQ']
        
        # AI agents use more sophisticated analysis
        for symbol in symbols:
            if symbol in market_data:
                price = market_data[symbol].get('price', 0)
                volume = market_data[symbol].get('volume', 0)
                volatility = market_data[symbol].get('volatility', 0)
                
                # AI criteria: complex analysis
                ai_score = (volume / 1000000) * (volatility * 100) * confidence
                
                if ai_score > 0.5 and random.random() < 0.6:  # 60% chance
                    return {
                        'agent_id': agent_id,
                        'symbol': symbol,
                        'action': 'BUY',
                        'quantity': random.uniform(3, 10),
                        'price': price,
                        'confidence': confidence,
                        'reasoning': f'AI analysis: score {ai_score:.2f}',
                        'hold_duration': 'medium',
                        'stop_loss': price * 0.97,
                        'take_profit': price * 1.05,
                        'style': 'ai_enhanced'
                    }
        
        return None
    
    def _generate_standard_decision(self, agent_id: str, market_data: Dict[str, Any], 
                                  confidence: float) -> Optional[Dict[str, Any]]:
        """Generate standard trading decision"""
        symbols = ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'SPY', 'QQQ']
        
        for symbol in symbols:
            if symbol in market_data:
                price = market_data[symbol].get('price', 0)
                
                # Standard criteria: moderate confidence
                if random.random() < confidence:
                    return {
                        'agent_id': agent_id,
                        'symbol': symbol,
                        'action': 'BUY',
                        'quantity': random.uniform(2, 6),
                        'price': price,
                        'confidence': confidence,
                        'reasoning': 'Standard analysis',
                        'hold_duration': 'medium',
                        'stop_loss': price * 0.96,
                        'take_profit': price * 1.06,
                        'style': 'standard'
                    }
        
        return None
    
    def _calculate_agent_confidence(self, agent_id: str) -> float:
        """Calculate agent confidence based on performance"""
        performance = self.agent_performance[agent_id]
        
        # Base confidence
        base_confidence = 0.5
        
        # Adjust based on win rate
        if performance.win_rate > 0:
            win_rate_bonus = (performance.win_rate - 0.5) * 0.3
            base_confidence += win_rate_bonus
        
        # Adjust based on total PnL
        if performance.total_pnl > 0:
            pnl_bonus = min(0.2, performance.total_pnl / 1000)
            base_confidence += pnl_bonus
        
        # Ensure confidence is between 0.1 and 0.9
        return max(0.1, min(0.9, base_confidence))
    
    def _analyze_agent_performance(self, agent_id: str) -> Dict[str, Any]:
        """Analyze agent performance for reflection"""
        performance = self.agent_performance[agent_id]
        
        return {
            'total_trades': performance.total_trades,
            'win_rate': performance.win_rate,
            'total_pnl': performance.total_pnl,
            'confidence_score': performance.confidence_score,
            'learning_rate': performance.learning_rate
        }
    
    def _generate_learning_insights(self, agent_id: str) -> List[str]:
        """Generate learning insights for agent"""
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
        """Suggest strategy adjustments based on performance"""
        performance = self.agent_performance[agent_id]
        
        adjustments = []
        
        if performance.win_rate < 0.4:
            adjustments.append("Reduce position size")
            adjustments.append("Increase confidence threshold")
        
        if performance.total_pnl < 0:
            adjustments.append("Review stop-loss levels")
            adjustments.append("Consider different timeframes")
        
        return adjustments
    
    def _update_confidence(self, agent_id: str) -> float:
        """Update agent confidence based on performance"""
        performance = self.agent_performance[agent_id]
        
        # Learning rate adjustment
        if performance.win_rate > 0.6:
            performance.learning_rate = min(0.2, performance.learning_rate + 0.01)
        elif performance.win_rate < 0.4:
            performance.learning_rate = max(0.05, performance.learning_rate - 0.01)
        
        # Update confidence
        new_confidence = performance.confidence_score + (performance.learning_rate * 0.1)
        return max(0.1, min(0.9, new_confidence))
    
    def _update_agent_performance(self, cycle_results: Dict[str, Any]):
        """Update agent performance metrics"""
        for agent_id, decision in cycle_results['agent_decisions'].items():
            if decision:
                performance = self.agent_performance[agent_id]
                performance.total_trades += 1
                performance.last_decision_time = datetime.now()
    
    def _log_cycle_summary(self, cycle_results: Dict[str, Any]):
        """Log cycle summary"""
        self.logger.info(f"Cycle {cycle_results['cycle']} Summary:")
        self.logger.info(f"  Total Decisions: {cycle_results['total_decisions']}")
        self.logger.info(f"  Total Reflections: {cycle_results['total_reflections']}")
        self.logger.info(f"  Selected Trades: {len(cycle_results['hierarchy_selection'])}")
        
        # Log agent performance
        for agent_id, performance in self.agent_performance.items():
            self.logger.info(f"  Agent {agent_id}: {performance.total_trades} trades, "
                           f"{performance.win_rate:.2f} win rate, "
                           f"${performance.total_pnl:.2f} PnL")

class HierarchyManager:
    """Manages agent hierarchy and trade selection"""
    
    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self.performance_weights = {}
        
    def select_trades(self, agent_decisions: Dict[str, Any], 
                     agent_performance: Dict[str, AgentPerformance]) -> List[Dict[str, Any]]:
        """Select trades based on agent hierarchy and performance"""
        selected_trades = []
        
        # Calculate performance scores
        performance_scores = {}
        for agent_id, performance in agent_performance.items():
            score = (performance.win_rate * 0.4 + 
                    (performance.total_pnl / 1000) * 0.3 + 
                    performance.confidence_score * 0.3)
            performance_scores[agent_id] = score
        
        # Sort agents by performance
        sorted_agents = sorted(performance_scores.items(), key=lambda x: x[1], reverse=True)
        
        # Select top performing agents' trades
        max_trades = min(11, len(agent_decisions))  # Max 11 trades per cycle
        
        for i, (agent_id, score) in enumerate(sorted_agents[:max_trades]):
            decision = agent_decisions.get(agent_id)
            if decision and score > 0.3:  # Minimum performance threshold
                selected_trades.append(decision)
                self.logger.info(f"Selected trade from {agent_id} (score: {score:.2f})")
        
        return selected_trades

