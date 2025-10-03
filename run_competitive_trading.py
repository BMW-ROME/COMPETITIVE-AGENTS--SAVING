#!/usr/bin/env python3
"""
Competitive Trading System - Simplified
======================================
Implements competitive trading with:
- All agents making decisions every cycle
- Continuous learning and reflections
- Quick trades (scalping)
- Hierarchy-based agent selection
"""
from rl_100_percent_execution import get_100_percent_execution_optimizer


import asyncio
import logging
import sys
import os
import random
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from dataclasses import dataclass

# Add src to path
sys.path.insert(0, 'src')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/competitive_trading.log', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger("CompetitiveTrading")

@dataclass
class AgentPerformance:
    """Track agent performance metrics"""
    agent_id: str
    total_trades: int = 0
    successful_trades: int = 0
    total_pnl: float = 0.0
    win_rate: float = 0.0
    confidence_score: float = 0.5
    last_decision_time: Optional[datetime] = None
    learning_rate: float = 0.1

class CompetitiveTradingSystem:
    """Simplified competitive trading system"""
    
    def __init__(self):
        self.logger = logger
        self.agents = {}
        self.agent_performance = {}
        self.cycle_count = 0
        self.total_decisions = 0
        self.total_reflections = 0
        self.total_trades = 0
        
        # Initialize agents
        self._initialize_agents()
        
    def _initialize_agents(self):
        """Initialize all 12 agents with distinct styles"""
        agent_configs = {
            # Conservative Agents
            'conservative_1': {'style': 'conservative', 'confidence_threshold': 0.8, 'max_position': 0.02},
            'balanced_1': {'style': 'balanced', 'confidence_threshold': 0.6, 'max_position': 0.05},
            'aggressive_1': {'style': 'aggressive', 'confidence_threshold': 0.4, 'max_position': 0.08},
            
            # Technical Analysis Agents
            'fractal_1': {'style': 'fractal_analysis', 'confidence_threshold': 0.7, 'max_position': 0.04},
            'candle_range_1': {'style': 'candle_range', 'confidence_threshold': 0.6, 'max_position': 0.03},
            'quant_pattern_1': {'style': 'quantitative', 'confidence_threshold': 0.5, 'max_position': 0.06},
            
            # Quick Trade Agents (Scalping)
            'scalping_1': {'style': 'scalping', 'confidence_threshold': 0.6, 'max_position': 0.02},
            'momentum_1': {'style': 'momentum', 'confidence_threshold': 0.5, 'max_position': 0.05},
            'arbitrage_1': {'style': 'arbitrage', 'confidence_threshold': 0.8, 'max_position': 0.03},
            
            # AI Enhanced Agents
            'ai_enhanced_1': {'style': 'ai_enhanced', 'confidence_threshold': 0.6, 'max_position': 0.07},
            'ml_pattern_1': {'style': 'ml_pattern', 'confidence_threshold': 0.7, 'max_position': 0.04},
            'adaptive_1': {'style': 'adaptive', 'confidence_threshold': 0.5, 'max_position': 0.05}
        }
        
        for agent_id, config in agent_configs.items():
            self.agents[agent_id] = config
            self.agent_performance[agent_id] = AgentPerformance(agent_id)
            
        self.logger.info(f"Initialized {len(self.agents)} competitive agents")
    
    async def run_competitive_cycle(self) -> Dict[str, Any]:
        """Run a competitive trading cycle"""
        self.cycle_count += 1
        cycle_start = datetime.now()
        
        # Generate market data
        market_data = self._generate_market_data()
        
        # Force all agents to make decisions
        agent_decisions = {}
        agent_reflections = {}
        
        for agent_id in self.agents.keys():
            # Force decision
            decision = self._force_agent_decision(agent_id, market_data)
            agent_decisions[agent_id] = decision
            
            if decision:
                self.total_decisions += 1
            
            # Force reflection
            reflection = self._force_agent_reflection(agent_id, decision)
            agent_reflections[agent_id] = reflection
            
            if reflection:
                self.total_reflections += 1
        
        # Hierarchy selection
        selected_trades = self._select_trades(agent_decisions)
        self.total_trades += len(selected_trades)
        
        # Update performance
        self._update_agent_performance(agent_decisions)
        
        cycle_result = {
            'cycle': self.cycle_count,
            'timestamp': datetime.now(),
            'agent_decisions': agent_decisions,
            'agent_reflections': agent_reflections,
            'selected_trades': selected_trades,
            'total_decisions': sum(1 for d in agent_decisions.values() if d),
            'total_reflections': sum(1 for r in agent_reflections.values() if r),
            'cycle_duration': (datetime.now() - cycle_start).total_seconds()
        }
        
        # Log cycle summary
        self._log_cycle_summary(cycle_result)
        
        return cycle_result
    
    def _generate_market_data(self) -> Dict[str, Any]:
        """Generate simulated market data"""
        symbols = ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'SPY', 'QQQ']
        market_data = {}
        
        for symbol in symbols:
            base_price = random.uniform(100, 500)
            price_change = random.uniform(-0.05, 0.05)
            volume = random.randint(1000000, 10000000)
            
            market_data[symbol] = {
                'price': base_price * (1 + price_change),
                'price_change': price_change,
                'volume': volume,
                'volatility': random.uniform(0.01, 0.05)
            }
        
        return market_data
    
    def _force_agent_decision(self, agent_id: str, market_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Force agent to make a decision"""
        agent_config = self.agents[agent_id]
        performance = self.agent_performance[agent_id]
        
        # Calculate confidence
        confidence = self._calculate_agent_confidence(agent_id)
        
        # Force decision based on style
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
    
    def _force_agent_reflection(self, agent_id: str, decision: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Force agent to reflect"""
        performance = self.agent_performance[agent_id]
        
        # Always reflect if enough time has passed
        if (performance.last_decision_time is None or 
            (datetime.now() - performance.last_decision_time).total_seconds() > 300):  # 5 minutes
            
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
        
        return None
    
    def _generate_scalping_decision(self, agent_id: str, market_data: Dict[str, Any], confidence: float) -> Optional[Dict[str, Any]]:
        """Generate scalping decision"""
        symbols = ['SPY', 'QQQ', 'AAPL', 'MSFT']
        
        for symbol in symbols:
            if symbol in market_data:
                price = market_data[symbol]['price']
                volume = market_data[symbol]['volume']
                
                if volume > 1000000 and random.random() < 0.4:  # 40% chance
                    return {
                        'agent_id': agent_id,
                        'symbol': symbol,
                        'action': 'BUY',
                        'quantity': random.uniform(1, 5),
                        'price': price,
                        'confidence': confidence,
                        'reasoning': 'Scalping opportunity',
                        'hold_duration': 'very_short',
                        'style': 'scalping'
                    }
        
        return None
    
    def _generate_momentum_decision(self, agent_id: str, market_data: Dict[str, Any], confidence: float) -> Optional[Dict[str, Any]]:
        """Generate momentum decision"""
        symbols = ['TSLA', 'GOOGL', 'AAPL', 'MSFT']
        
        for symbol in symbols:
            if symbol in market_data:
                price = market_data[symbol]['price']
                price_change = market_data[symbol]['price_change']
                
                if abs(price_change) > 0.02 and random.random() < 0.5:  # 50% chance
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
                        'style': 'momentum'
                    }
        
        return None
    
    def _generate_arbitrage_decision(self, agent_id: str, market_data: Dict[str, Any], confidence: float) -> Optional[Dict[str, Any]]:
        """Generate arbitrage decision"""
        spy_price = market_data.get('SPY', {}).get('price', 0)
        qqq_price = market_data.get('QQQ', {}).get('price', 0)
        
        if spy_price > 0 and qqq_price > 0:
            ratio = spy_price / qqq_price
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
                    'style': 'arbitrage'
                }
        
        return None
    
    def _generate_ai_decision(self, agent_id: str, market_data: Dict[str, Any], confidence: float) -> Optional[Dict[str, Any]]:
        """Generate AI-enhanced decision"""
        symbols = ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'SPY', 'QQQ']
        
        for symbol in symbols:
            if symbol in market_data:
                price = market_data[symbol]['price']
                volume = market_data[symbol]['volume']
                volatility = market_data[symbol]['volatility']
                
                ai_score = (volume / 1000000) * (volatility * 100) * confidence
                
                if ai_score > 0.5 and random.random() < 0.7:  # 70% chance
                    return {
                        'agent_id': agent_id,
                        'symbol': symbol,
                        'action': 'BUY',
                        'quantity': random.uniform(3, 10),
                        'price': price,
                        'confidence': confidence,
                        'reasoning': f'AI analysis: score {ai_score:.2f}',
                        'hold_duration': 'medium',
                        'style': 'ai_enhanced'
                    }
        
        return None
    
    def _generate_standard_decision(self, agent_id: str, market_data: Dict[str, Any], confidence: float) -> Optional[Dict[str, Any]]:
        """Generate standard decision"""
        symbols = ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'SPY', 'QQQ']
        
        for symbol in symbols:
            if symbol in market_data:
                price = market_data[symbol]['price']
                
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
                        'style': 'standard'
                    }
        
        return None
    
    def _calculate_agent_confidence(self, agent_id: str) -> float:
        """Calculate agent confidence"""
        performance = self.agent_performance[agent_id]
        
        base_confidence = 0.5
        
        if performance.win_rate > 0:
            win_rate_bonus = (performance.win_rate - 0.5) * 0.3
            base_confidence += win_rate_bonus
        
        if performance.total_pnl > 0:
            pnl_bonus = min(0.2, performance.total_pnl / 1000)
            base_confidence += pnl_bonus
        
        return max(0.1, min(0.9, base_confidence))
    
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
        
        if performance.win_rate > 0.6:
            performance.learning_rate = min(0.2, performance.learning_rate + 0.01)
        elif performance.win_rate < 0.4:
            performance.learning_rate = max(0.05, performance.learning_rate - 0.01)
        
        new_confidence = performance.confidence_score + (performance.learning_rate * 0.1)
        return max(0.1, min(0.9, new_confidence))
    
    def _select_trades(self, agent_decisions: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Select trades based on hierarchy"""
        selected_trades = []
        
        # Calculate performance scores
        performance_scores = {}
        for agent_id, performance in self.agent_performance.items():
            score = (performance.win_rate * 0.4 + 
                    (performance.total_pnl / 1000) * 0.3 + 
                    performance.confidence_score * 0.3)
            performance_scores[agent_id] = score
        
        # Sort agents by performance
        sorted_agents = sorted(performance_scores.items(), key=lambda x: x[1], reverse=True)
        
        # Select top performing agents' trades
        max_trades = min(11, len(agent_decisions))
        
        for i, (agent_id, score) in enumerate(sorted_agents[:max_trades]):
            decision = agent_decisions.get(agent_id)
            if decision and score > 0.3:
                selected_trades.append(decision)
                self.logger.info(f"Selected trade from {agent_id} (score: {score:.2f})")
        
        return selected_trades
    
    def _update_agent_performance(self, agent_decisions: Dict[str, Any]):
        """Update agent performance"""
        for agent_id, decision in agent_decisions.items():
            if decision:
                performance = self.agent_performance[agent_id]
                performance.total_trades += 1
                performance.last_decision_time = datetime.now()
    
    def _log_cycle_summary(self, cycle_result: Dict[str, Any]):
        """Log cycle summary"""
        self.logger.info(f"Cycle {cycle_result['cycle']} Summary:")
        self.logger.info(f"  Total Decisions: {cycle_result['total_decisions']}")
        self.logger.info(f"  Total Reflections: {cycle_result['total_reflections']}")
        self.logger.info(f"  Selected Trades: {len(cycle_result['selected_trades'])}")
        self.logger.info(f"  Cycle Duration: {cycle_result['cycle_duration']:.2f}s")
        
        # Log agent performance
        for agent_id, performance in self.agent_performance.items():
            self.logger.info(f"  Agent {agent_id}: {performance.total_trades} trades, "
                           f"{performance.win_rate:.2f} win rate, "
                           f"${performance.total_pnl:.2f} PnL")
        
        # Log system metrics
        self.logger.info(f"System Metrics:")
        self.logger.info(f"  Total Cycles: {self.cycle_count}")
        self.logger.info(f"  Total Decisions: {self.total_decisions}")
        self.logger.info(f"  Total Reflections: {self.total_reflections}")
        self.logger.info(f"  Total Trades: {self.total_trades}")

async def main():
    try:
        logger.info("=" * 60)
        logger.info("COMPETITIVE TRADING SYSTEM STARTING")
        logger.info("=" * 60)
        
        # Create competitive trading system
        system = CompetitiveTradingSystem()
        
        logger.info("Competitive trading system initialized!")
        logger.info("All agents will now make decisions and reflections every cycle!")
        
        # Run competitive cycles
        while True:
            try:
                cycle_result = await system.run_competitive_cycle()
                await asyncio.sleep(30)  # 30-second cycles
                
            except KeyboardInterrupt:
                logger.info("System stopped by user")
                break
            except Exception as e:
                logger.error(f"Error in competitive cycle: {e}")
                await asyncio.sleep(60)
        
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)

if __name__ == "__main__":
    # Create logs directory
    os.makedirs('logs', exist_ok=True)
    
    # Run the competitive system
    asyncio.run(main())

