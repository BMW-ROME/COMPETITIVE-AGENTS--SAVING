#!/usr/bin/env python3
"""
🧠 REINFORCEMENT LEARNING OPTIMIZATION ENGINE
=============================================
Advanced RL-based trading optimization system that learns from execution
failures and adapts position sizing, timing, and strategy selection.
"""

import numpy as np
import pandas as pd
import json
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import random
from collections import deque
import pickle
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("RL_Optimizer")

@dataclass
class TradeState:
    """Comprehensive state representation for RL agent"""
    buying_power: float
    portfolio_value: float
    position_count: int
    recent_failures: int
    market_volatility: float
    time_since_last_trade: int
    wash_trade_risk: float
    symbol_momentum: Dict[str, float]
    
@dataclass
class TradeAction:
    """Action space for trading decisions"""
    symbol: str
    side: str  # 'buy' or 'sell'
    quantity: float
    confidence: float
    timing_delay: int  # seconds to wait before execution

@dataclass
class TradeOutcome:
    """Result of trade execution for learning"""
    success: bool
    error_type: Optional[str]  # 'insufficient_power', 'wash_trade', 'insufficient_qty'
    actual_quantity: float
    expected_profit: float
    actual_profit: float
    execution_time: float

class QLearningAgent:
    """Q-Learning agent for optimal position sizing and timing"""
    
    def __init__(self, learning_rate=0.1, discount_factor=0.95, epsilon=0.2):
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        
        # Q-table for state-action values
        self.q_table = {}
        
        # Experience replay buffer
        self.experience_buffer = deque(maxlen=10000)
        
        # Performance tracking
        self.success_rate = 0.0
        self.total_actions = 0
        self.successful_actions = 0
        
        # Load existing model if available
        self.load_model()
        
    def get_state_key(self, state: TradeState) -> str:
        """Convert state to hashable key for Q-table"""
        # Discretize continuous values for Q-table
        buying_power_bucket = min(int(state.buying_power / 100), 50)  # Buckets of $100
        portfolio_bucket = int(state.portfolio_value / 10000)  # Buckets of $10K
        volatility_bucket = min(int(state.market_volatility * 100), 20)  # 0-20% buckets
        
        return f"{buying_power_bucket}_{portfolio_bucket}_{state.position_count}_{state.recent_failures}_{volatility_bucket}"
    
    def get_action_key(self, action: TradeAction) -> str:
        """Convert action to hashable key"""
        qty_bucket = min(int(action.quantity * 10), 100)  # Quantity buckets
        confidence_bucket = min(int(action.confidence * 10), 10)
        
        return f"{action.symbol}_{action.side}_{qty_bucket}_{confidence_bucket}"
    
    def select_action(self, state: TradeState, possible_actions: List[TradeAction]) -> TradeAction:
        """Select action using epsilon-greedy policy with Q-learning"""
        
        # Epsilon-greedy exploration
        if random.random() < self.epsilon:
            logger.info(f"🎲 [RL] EXPLORATION: Random action selection (epsilon={self.epsilon})")
            return random.choice(possible_actions)
        
        # Exploitation: Choose best Q-value action
        state_key = self.get_state_key(state)
        best_action = None
        best_q_value = float('-inf')
        
        for action in possible_actions:
            action_key = self.get_action_key(action)
            q_key = f"{state_key}_{action_key}"
            
            q_value = self.q_table.get(q_key, 0.0)
            
            if q_value > best_q_value:
                best_q_value = q_value
                best_action = action
        
        if best_action is None:
            logger.info("🎯 [RL] NO Q-VALUES: Defaulting to exploration")
            best_action = random.choice(possible_actions)
        else:
            logger.info(f"🧠 [RL] EXPLOITATION: Q-value={best_q_value:.3f}")
        
        return best_action
    
    def update_q_value(self, state: TradeState, action: TradeAction, 
                      outcome: TradeOutcome, next_state: TradeState):
        """Update Q-table based on outcome"""
        
        # Calculate reward based on outcome
        reward = self.calculate_reward(outcome, state)
        
        # Get keys
        state_key = self.get_state_key(state)
        action_key = self.get_action_key(action)
        q_key = f"{state_key}_{action_key}"
        
        # Current Q-value
        current_q = self.q_table.get(q_key, 0.0)
        
        # Max Q-value for next state (approximation)
        next_state_key = self.get_state_key(next_state)
        next_max_q = 0.0  # Simplified - in full implementation, would check all actions
        
        # Q-learning update
        new_q = current_q + self.learning_rate * (
            reward + self.discount_factor * next_max_q - current_q
        )
        
        self.q_table[q_key] = new_q
        
        # Update performance metrics
        self.total_actions += 1
        if outcome.success:
            self.successful_actions += 1
        
        self.success_rate = self.successful_actions / self.total_actions
        
        logger.info(f"📊 [RL] Q-UPDATE: {q_key} | Old Q: {current_q:.3f} → New Q: {new_q:.3f} | Reward: {reward:.2f}")
        logger.info(f"📈 [RL] SUCCESS RATE: {self.success_rate:.1%} ({self.successful_actions}/{self.total_actions})")
        
        # Store experience for replay
        self.experience_buffer.append((state, action, outcome, next_state))
        
        # Decay epsilon (reduce exploration over time)
        if self.epsilon > 0.01:
            self.epsilon *= 0.999
    
    def calculate_reward(self, outcome: TradeOutcome, state: TradeState) -> float:
        """Calculate reward signal for reinforcement learning"""
        
        if outcome.success:
            # Base reward for successful execution
            reward = 100.0
            
            # Bonus for profit
            if outcome.actual_profit > 0:
                reward += outcome.actual_profit * 10  # Scale profit reward
            
            # Bonus for efficient capital usage
            capital_efficiency = outcome.actual_quantity / (state.buying_power / 100)
            reward += min(capital_efficiency * 20, 50)  # Cap at 50
            
        else:
            # Penalties for failures
            if outcome.error_type == 'insufficient_power':
                reward = -50.0  # Heavy penalty for poor capital management
            elif outcome.error_type == 'wash_trade':
                reward = -30.0  # Medium penalty for wash trades
            elif outcome.error_type == 'insufficient_qty':
                reward = -20.0  # Light penalty for quantity issues
            else:
                reward = -10.0  # Generic failure penalty
            
            # Additional penalty for repeated failures
            reward -= state.recent_failures * 5
        
        return reward
    
    def save_model(self):
        """Save Q-table and parameters"""
        model_data = {
            'q_table': self.q_table,
            'epsilon': self.epsilon,
            'success_rate': self.success_rate,
            'total_actions': self.total_actions,
            'successful_actions': self.successful_actions
        }
        
        os.makedirs('models', exist_ok=True)
        with open('models/rl_trading_model.json', 'w') as f:
            json.dump(model_data, f, indent=2)
        
        logger.info(f"💾 [RL] Model saved: {len(self.q_table)} Q-values, {self.success_rate:.1%} success rate")
    
    def load_model(self):
        """Load existing Q-table and parameters"""
        try:
            with open('models/rl_trading_model.json', 'r') as f:
                model_data = json.load(f)
            
            self.q_table = model_data.get('q_table', {})
            self.epsilon = model_data.get('epsilon', 0.2)
            self.success_rate = model_data.get('success_rate', 0.0)
            self.total_actions = model_data.get('total_actions', 0)
            self.successful_actions = model_data.get('successful_actions', 0)
            
            logger.info(f"🔄 [RL] Model loaded: {len(self.q_table)} Q-values, {self.success_rate:.1%} success rate")
            
        except FileNotFoundError:
            logger.info("🆕 [RL] No existing model found, starting fresh")

class RLTradingOptimizer:
    """Main RL optimization engine"""
    
    def __init__(self):
        self.rl_agent = QLearningAgent()
        self.wash_trade_tracker = deque(maxlen=100)  # Track recent trades
        self.failure_counter = 0
        
    def optimize_position_size(self, suggested_quantity: float, symbol: str, 
                             side: str, buying_power: float, 
                             current_price: float) -> Tuple[float, str]:
        """RL-optimized position sizing based on available capital"""
        
        max_affordable = buying_power / current_price * 0.95  # 95% safety margin
        
        if suggested_quantity > max_affordable:
            # Learn from capital constraint
            optimal_quantity = max_affordable * 0.8  # Conservative approach
            
            reduction_pct = (1 - optimal_quantity / suggested_quantity) * 100
            reason = f"RL Capital Optimization: {reduction_pct:.1f}% reduction (affordable: {max_affordable:.4f})"
            
            logger.info(f"🧠 [RL] CAPITAL OPTIMIZATION: {symbol} {side}")
            logger.info(f"    Original: {suggested_quantity:.4f} → Optimized: {optimal_quantity:.4f}")
            logger.info(f"    Buying Power: ${buying_power:.2f} | Price: ${current_price:.2f}")
            
            return optimal_quantity, reason
        
        return suggested_quantity, "No optimization needed"
    
    def check_wash_trade_risk(self, symbol: str, side: str) -> Tuple[bool, int]:
        """RL-based wash trade detection and timing optimization"""
        
        current_time = datetime.now()
        
        # Check recent trades for same symbol
        recent_symbol_trades = [
            trade for trade in self.wash_trade_tracker 
            if trade['symbol'] == symbol and 
            (current_time - trade['timestamp']).seconds < 300  # 5 minute window
        ]
        
        if len(recent_symbol_trades) >= 2:
            # High wash trade risk - suggest delay
            last_trade = recent_symbol_trades[-1]
            time_since_last = (current_time - last_trade['timestamp']).seconds
            
            if time_since_last < 60:  # Less than 1 minute
                suggested_delay = 90 - time_since_last
                logger.info(f"⚠️ [RL] WASH TRADE RISK: {symbol} - suggest {suggested_delay}s delay")
                return True, suggested_delay
        
        return False, 0
    
    def create_state_representation(self, market_data: Dict, 
                                  buying_power: float, 
                                  portfolio_value: float,
                                  position_count: int) -> TradeState:
        """Create comprehensive state for RL agent"""
        
        # Calculate market volatility
        volatilities = []
        momentums = {}
        
        for symbol, data in market_data.items():
            vol = abs(data.get('change_pct', 0)) / 100
            volatilities.append(vol)
            momentums[symbol] = data.get('change_pct', 0) / 100
        
        avg_volatility = np.mean(volatilities) if volatilities else 0.0
        
        # Calculate wash trade risk
        wash_risk = min(len(self.wash_trade_tracker) / 50, 1.0)  # Normalize to 0-1
        
        # Time since last trade
        time_since_last = 0
        if self.wash_trade_tracker:
            last_trade_time = self.wash_trade_tracker[-1]['timestamp']
            time_since_last = (datetime.now() - last_trade_time).seconds
        
        return TradeState(
            buying_power=buying_power,
            portfolio_value=portfolio_value,
            position_count=position_count,
            recent_failures=min(self.failure_counter, 10),
            market_volatility=avg_volatility,
            time_since_last_trade=min(time_since_last, 300),
            wash_trade_risk=wash_risk,
            symbol_momentum=momentums
        )
    
    def optimize_trades(self, suggested_trades: List[Dict], 
                       market_data: Dict, buying_power: float,
                       portfolio_value: float, position_count: int) -> List[Dict]:
        """Main RL optimization function for trade selection and sizing"""
        
        if not suggested_trades:
            return []
        
        # Create current state
        state = self.create_state_representation(
            market_data, buying_power, portfolio_value, position_count
        )
        
        # Convert suggested trades to actions
        possible_actions = []
        for trade in suggested_trades:
            action = TradeAction(
                symbol=trade['symbol'],
                side=trade['side'],
                quantity=trade['quantity'],
                confidence=trade.get('confidence', 0.5),
                timing_delay=0
            )
            possible_actions.append(action)
        
        # Select optimal action using RL
        if possible_actions:
            optimal_action = self.rl_agent.select_action(state, possible_actions)
            
            # Apply capital optimization
            current_price = market_data[optimal_action.symbol]['price']
            optimized_qty, optimization_reason = self.optimize_position_size(
                optimal_action.quantity, 
                optimal_action.symbol,
                optimal_action.side,
                buying_power,
                current_price
            )
            
            # Check wash trade risk
            wash_risk, delay = self.check_wash_trade_risk(
                optimal_action.symbol, 
                optimal_action.side
            )
            
            # Create optimized trade
            optimized_trade = {
                'symbol': optimal_action.symbol,
                'side': optimal_action.side,
                'quantity': optimized_qty,
                'confidence': optimal_action.confidence,
                'optimization_reason': optimization_reason,
                'wash_trade_delay': delay,
                'rl_selected': True
            }
            
            logger.info(f"🎯 [RL] OPTIMAL TRADE SELECTED:")
            logger.info(f"    {optimal_action.symbol} {optimal_action.side} {optimized_qty:.4f}")
            logger.info(f"    Confidence: {optimal_action.confidence:.3f}")
            logger.info(f"    Optimization: {optimization_reason}")
            if wash_risk:
                logger.info(f"    Wash Trade Delay: {delay}s")
            
            return [optimized_trade]
        
        return []
    
    def record_trade_outcome(self, trade: Dict, success: bool, 
                           error_type: Optional[str] = None,
                           actual_quantity: float = 0.0,
                           actual_profit: float = 0.0):
        """Record trade outcome for RL learning"""
        
        # Track trade in wash trade detector
        self.wash_trade_tracker.append({
            'symbol': trade['symbol'],
            'side': trade['side'],
            'timestamp': datetime.now(),
            'success': success
        })
        
        # Update failure counter
        if success:
            self.failure_counter = max(0, self.failure_counter - 1)
        else:
            self.failure_counter += 1
        
        # Create outcome for RL learning
        outcome = TradeOutcome(
            success=success,
            error_type=error_type,
            actual_quantity=actual_quantity,
            expected_profit=trade.get('expected_profit', 0.0),
            actual_profit=actual_profit,
            execution_time=time.time()
        )
        
        logger.info(f"📝 [RL] TRADE OUTCOME RECORDED:")
        logger.info(f"    Success: {success} | Error: {error_type}")
        logger.info(f"    Quantity: {actual_quantity:.4f} | Profit: ${actual_profit:.2f}")
        logger.info(f"    Failure Counter: {self.failure_counter}")
        
        # Save model periodically
        if self.rl_agent.total_actions % 10 == 0:
            self.rl_agent.save_model()
    
    def get_performance_stats(self) -> Dict:
        """Get current RL performance statistics"""
        return {
            'success_rate': self.rl_agent.success_rate,
            'total_actions': self.rl_agent.total_actions,
            'successful_actions': self.rl_agent.successful_actions,
            'epsilon': self.rl_agent.epsilon,
            'q_table_size': len(self.rl_agent.q_table),
            'failure_counter': self.failure_counter,
            'recent_trades': len(self.wash_trade_tracker)
        }

# Global RL optimizer instance
rl_optimizer = RLTradingOptimizer()

def get_rl_optimizer():
    """Get global RL optimizer instance"""
    return rl_optimizer

if __name__ == "__main__":
    # Test RL optimizer
    optimizer = RLTradingOptimizer()
    
    # Mock data for testing
    test_market_data = {
        'AAPL': {'current_price': 254.37, 'change_pct': 0.5},
        'MSFT': {'current_price': 514.64, 'change_pct': -0.2}
    }
    
    test_trades = [
        {'symbol': 'AAPL', 'side': 'buy', 'quantity': 10.0, 'confidence': 0.8},
        {'symbol': 'MSFT', 'side': 'sell', 'quantity': 5.0, 'confidence': 0.6}
    ]
    
    # Test optimization
    optimized = optimizer.optimize_trades(
        test_trades, test_market_data, 
        buying_power=1000.0, portfolio_value=50000.0, position_count=5
    )
    
    print(f"🧪 [TEST] Optimized trades: {len(optimized)}")
    for trade in optimized:
        print(f"    {trade}")
    
    # Test outcome recording
    if optimized:
        optimizer.record_trade_outcome(
            optimized[0], success=True, 
            actual_quantity=2.5, actual_profit=15.75
        )
    
    # Print stats
    stats = optimizer.get_performance_stats()
    print(f"📊 [STATS] {stats}")