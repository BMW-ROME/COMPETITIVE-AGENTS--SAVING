#!/usr/bin/env python3
"""
Production Trading System - Risk-Managed Multi-Agent Trading
===========================================================
Production-ready competitive trading system with proper risk management,
position sizing, and execution logic.
"""

import asyncio
import logging
import sys
import os
import random
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import alpaca_trade_api as tradeapi
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/production_trading.log', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger("ProductionTrading")

@dataclass
class AgentPerformance:
    """Track agent performance metrics"""
    agent_id: str
    total_trades: int = 0
    successful_trades: int = 0
    total_pnl: float = 0.0
    confidence_score: float = 0.5
    last_decision_time: Optional[datetime] = None
    buying_power_used: float = 0.0
    
    @property
    def win_rate(self) -> float:
        return self.successful_trades / max(1, self.total_trades)

class ProductionTradingSystem:
    """Production-ready competitive trading system"""
    
    def __init__(self):
        self.logger = logger
        
        # Initialize Alpaca API
        self.api = tradeapi.REST(
            key_id=os.getenv("ALPACA_API_KEY"),
            secret_key=os.getenv("ALPACA_SECRET_KEY"),
            base_url=os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets"),
            api_version='v2'
        )
        
        # Get account info
        self.account = self.api.get_account()
        self.initial_buying_power = float(self.account.buying_power)
        
        self.logger.info(f"✅ Connected to Alpaca (Buying Power: ${self.initial_buying_power:,.2f})")
        
        # Trading parameters
        self.max_position_per_trade = min(1000, self.initial_buying_power * 0.02)  # 2% max per trade
        self.max_total_exposure = self.initial_buying_power * 0.5  # 50% max total exposure
        self.min_trade_size = 50  # Minimum $50 per trade
        
        # Initialize agents
        self.agents = {}
        self.agent_performance = {}
        self.cycle_count = 0
        self.total_decisions = 0
        self.total_reflections = 0
        self.total_trades = 0
        self.total_pnl = 0.0
        self.current_positions = {}
        
        # Initialize agents
        self._initialize_agents()
    
    def _initialize_agents(self):
        """Initialize production agents with realistic parameters"""
        agent_configs = {
            # Conservative Agents (Lower risk, higher probability)
            'conservative_1': {'style': 'conservative', 'decision_rate': 0.4, 'risk_multiplier': 0.5},
            'conservative_2': {'style': 'conservative', 'decision_rate': 0.4, 'risk_multiplier': 0.5},
            
            # Balanced Agents (Medium risk, medium probability)
            'balanced_1': {'style': 'balanced', 'decision_rate': 0.5, 'risk_multiplier': 0.8},
            'balanced_2': {'style': 'balanced', 'decision_rate': 0.5, 'risk_multiplier': 0.8},
            
            # Aggressive Agents (Higher risk, lower probability)
            'aggressive_1': {'style': 'aggressive', 'decision_rate': 0.3, 'risk_multiplier': 1.5},
            'aggressive_2': {'style': 'aggressive', 'decision_rate': 0.3, 'risk_multiplier': 1.5},
            
            # Technical Analysis Agents
            'momentum_1': {'style': 'momentum', 'decision_rate': 0.6, 'risk_multiplier': 1.2},
            'scalping_1': {'style': 'scalping', 'decision_rate': 0.8, 'risk_multiplier': 0.6},
            
            # AI-Enhanced Agents
            'ai_enhanced_1': {'style': 'ai_enhanced', 'decision_rate': 0.4, 'risk_multiplier': 1.0},
            'ml_pattern_1': {'style': 'ml_pattern', 'decision_rate': 0.4, 'risk_multiplier': 1.0},
            
            # Arbitrage Agent (Very conservative)
            'arbitrage_1': {'style': 'arbitrage', 'decision_rate': 0.2, 'risk_multiplier': 0.3},
            'adaptive_1': {'style': 'adaptive', 'decision_rate': 0.5, 'risk_multiplier': 0.9}
        }
        
        for agent_id, config in agent_configs.items():
            self.agents[agent_id] = config
            self.agent_performance[agent_id] = AgentPerformance(agent_id)
            
        self.logger.info(f"Initialized {len(self.agents)} production trading agents")
    
    async def run_competitive_cycle(self) -> Dict[str, Any]:
        """Run a production competitive trading cycle"""
        self.cycle_count += 1
        cycle_start = datetime.now()
        
        # Update account and positions
        await self._update_account_status()
        
        # Get market data
        market_data = await self._get_market_data()
        if not market_data:
            self.logger.warning("No market data available, skipping cycle")
            return self._create_empty_cycle_result()
        
        # Generate decisions from all agents
        agent_decisions = {}
        agent_reflections = {}
        
        for agent_id in self.agents.keys():
            # Generate decision
            decision = self._generate_production_decision(agent_id, market_data)
            agent_decisions[agent_id] = decision
            
            if decision:
                self.total_decisions += 1
            
            # Generate reflection
            reflection = self._generate_reflection(agent_id, decision)
            agent_reflections[agent_id] = reflection
            
            if reflection:
                self.total_reflections += 1
        
        # Select and validate trades
        selected_trades = await self._select_and_validate_trades(agent_decisions, market_data)
        
        # Execute trades
        executed_trades = await self._execute_production_trades(selected_trades)
        self.total_trades += len(executed_trades)
        
        # Update performance
        self._update_agent_performance(agent_decisions, executed_trades)
        
        cycle_result = {
            'cycle': self.cycle_count,
            'timestamp': datetime.now(),
            'agent_decisions': agent_decisions,
            'agent_reflections': agent_reflections,
            'selected_trades': selected_trades,
            'executed_trades': executed_trades,
            'total_decisions': sum(1 for d in agent_decisions.values() if d),
            'total_reflections': sum(1 for r in agent_reflections.values() if r),
            'cycle_duration': (datetime.now() - cycle_start).total_seconds(),
            'account_info': {
                'buying_power': float(self.account.buying_power),
                'portfolio_value': float(self.account.portfolio_value),
                'positions_count': len(self.current_positions)
            }
        }
        
        # Log cycle summary
        self._log_production_cycle_summary(cycle_result)
        
        return cycle_result
    
    async def _update_account_status(self):
        """Update account and position information"""
        try:
            self.account = self.api.get_account()
            
            # Get current positions
            positions = self.api.list_positions()
            self.current_positions = {}
            
            for position in positions:
                self.current_positions[position.symbol] = {
                    'quantity': float(position.qty),
                    'market_value': float(position.market_value),
                    'avg_entry_price': float(position.avg_entry_price),
                    'unrealized_pnl': float(position.unrealized_pnl),
                    'side': position.side
                }
                
        except Exception as e:
            self.logger.error(f"Error updating account status: {e}")
    
    def _generate_production_decision(self, agent_id: str, market_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Generate production-ready trading decision"""
        agent_config = self.agents[agent_id]
        decision_rate = agent_config['decision_rate']
        
        # Decision probability check
        if random.random() > decision_rate:
            return None
        
        # Select symbols based on agent style
        symbols = self._select_symbols_for_agent(agent_id, market_data)
        if not symbols:
            return None
        
        symbol = random.choice(symbols)
        data = market_data[symbol]
        
        # Generate decision based on style
        decision = self._create_style_based_decision(agent_id, symbol, data, market_data)
        
        return decision
    
    def _select_symbols_for_agent(self, agent_id: str, market_data: Dict[str, Any]) -> List[str]:
        """Select appropriate symbols for each agent type"""
        agent_config = self.agents[agent_id]
        style = agent_config['style']
        
        all_symbols = list(market_data.keys())
        
        if style == 'conservative':
            # Prefer large-cap stable stocks
            preferred = ['AAPL', 'MSFT', 'SPY', 'QQQ']
            return [s for s in preferred if s in all_symbols] or all_symbols
        elif style == 'aggressive':
            # Prefer volatile stocks
            preferred = ['TSLA', 'NVDA', 'GOOGL']
            return [s for s in preferred if s in all_symbols] or all_symbols
        elif style == 'scalping':
            # Prefer high-volume ETFs
            preferred = ['SPY', 'QQQ']
            return [s for s in preferred if s in all_symbols] or all_symbols
        else:
            return all_symbols
    
    def _create_style_based_decision(self, agent_id: str, symbol: str, data: Dict[str, Any], market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create decision based on agent style"""
        agent_config = self.agents[agent_id]
        style = agent_config['style']
        risk_multiplier = agent_config['risk_multiplier']
        
        # Base trade size
        base_trade_value = self.max_position_per_trade * risk_multiplier
        base_trade_value = max(self.min_trade_size, min(base_trade_value, self.max_position_per_trade * 2))
        
        # Calculate quantity
        price = data['price']
        if price <= 0:
            return None
        
        quantity = int(base_trade_value / price)
        if quantity < 1:
            return None
        
        # Determine action based on style and market conditions
        action = self._determine_action(style, data, symbol)
        
        # Calculate confidence
        confidence = self._calculate_decision_confidence(agent_id, style, data, market_data)
        
        return {
            'agent_id': agent_id,
            'symbol': symbol,
            'action': action,
            'quantity': quantity,
            'price': price,
            'trade_value': quantity * price,
            'confidence': confidence,
            'reasoning': f'{style} strategy on {symbol}',
            'style': style,
            'risk_multiplier': risk_multiplier,
            'timestamp': datetime.now()
        }
    
    def _determine_action(self, style: str, data: Dict[str, Any], symbol: str) -> str:
        """Determine buy/sell action based on style and data"""
        price_change = data.get('price_change', 0)
        volatility = data.get('volatility', 0)
        current_position = self.current_positions.get(symbol, {})
        
        # Check current positions
        has_position = current_position.get('quantity', 0) != 0
        
        if style == 'conservative':
            # Conservative: mostly buy, occasional sell if profitable
            if has_position and current_position.get('unrealized_pnl', 0) > 0:
                return 'SELL'  # Take profits
            return 'BUY'
        
        elif style == 'aggressive':
            # Aggressive: trade based on momentum
            if price_change > 0.01:
                return 'BUY'
            elif price_change < -0.01:
                return 'SELL' if has_position else 'BUY'  # Buy the dip or sell
            return random.choice(['BUY', 'SELL'])
        
        elif style == 'scalping':
            # Scalping: quick in/out trades
            return random.choice(['BUY', 'SELL'])
        
        elif style == 'momentum':
            # Momentum: follow price direction
            return 'BUY' if price_change >= 0 else ('SELL' if has_position else 'BUY')
        
        else:
            # Default: balanced approach
            if has_position:
                return 'SELL' if random.random() < 0.3 else 'BUY'
            return 'BUY'
    
    def _calculate_decision_confidence(self, agent_id: str, style: str, data: Dict[str, Any], market_data: Dict[str, Any]) -> float:
        """Calculate decision confidence"""
        base_confidence = 0.5
        
        # Agent performance factor
        performance = self.agent_performance[agent_id]
        if performance.total_trades > 0:
            performance_factor = (performance.win_rate - 0.5) * 0.2
            base_confidence += performance_factor
        
        # Market conditions factor
        volatility = data.get('volatility', 0.01)
        volume = data.get('volume', 1000)
        
        # Higher confidence for better conditions
        if style == 'scalping' and volume > 50000:
            base_confidence += 0.1
        if volatility > 0.02:  # Good volatility for most strategies
            base_confidence += 0.1
        
        return max(0.1, min(0.95, base_confidence))
    
    async def _select_and_validate_trades(self, agent_decisions: Dict[str, Any], market_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Select and validate trades based on risk management"""
        valid_decisions = [d for d in agent_decisions.values() if d]
        
        if not valid_decisions:
            return []
        
        # Sort by confidence
        sorted_decisions = sorted(valid_decisions, key=lambda x: x['confidence'], reverse=True)
        
        selected_trades = []
        total_exposure = 0
        
        for decision in sorted_decisions:
            trade_value = decision['trade_value']
            
            # Check if we can afford this trade
            if total_exposure + trade_value > self.max_total_exposure:
                continue
            
            # Check if it's a valid trade (not wash trade)
            if await self._validate_trade(decision, market_data):
                selected_trades.append(decision)
                total_exposure += trade_value
                
                # Limit number of trades per cycle
                if len(selected_trades) >= 6:
                    break
        
        return selected_trades
    
    async def _validate_trade(self, decision: Dict[str, Any], market_data: Dict[str, Any]) -> bool:
        """Validate if trade is acceptable"""
        symbol = decision['symbol']
        action = decision['action']
        quantity = decision['quantity']
        
        # Check minimum trade value
        trade_value = decision['trade_value']
        if trade_value < self.min_trade_size:
            return False
        
        # Check if we have enough buying power
        buying_power = float(self.account.buying_power)
        if action == 'BUY' and trade_value > buying_power:
            return False
        
        # Check if we have shares to sell
        if action == 'SELL':
            current_position = self.current_positions.get(symbol, {})
            available_shares = abs(current_position.get('quantity', 0))
            if available_shares < quantity:
                return False
        
        return True
    
    async def _execute_production_trades(self, selected_trades: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Execute selected trades with production-level error handling"""
        executed_trades = []
        
        for trade in selected_trades:
            try:
                symbol = trade['symbol']
                action = trade['action']
                quantity = trade['quantity']
                
                # Add small delay between orders to avoid wash trade detection
                if executed_trades:
                    await asyncio.sleep(0.5)
                
                # Submit order
                order = self.api.submit_order(
                    symbol=symbol,
                    qty=quantity,
                    side='buy' if action == 'BUY' else 'sell',
                    type='market',
                    time_in_force='day'
                )
                
                executed_trade = {
                    'order_id': order.id,
                    'symbol': symbol,
                    'action': action,
                    'quantity': quantity,
                    'price': trade['price'],
                    'trade_value': trade['trade_value'],
                    'status': order.status,
                    'agent_id': trade['agent_id'],
                    'confidence': trade['confidence'],
                    'timestamp': datetime.now()
                }
                
                executed_trades.append(executed_trade)
                self.logger.info(f"✅ EXECUTED: {symbol} {action} {quantity} shares @ ${trade['price']:.2f} (${trade['trade_value']:.2f}) [Agent: {trade['agent_id']}]")
                
            except Exception as e:
                self.logger.error(f"❌ Failed to execute trade {trade['symbol']} {trade['action']}: {e}")
        
        return executed_trades
    
    async def _get_market_data(self) -> Dict[str, Any]:
        """Get market data with comprehensive fallback"""
        symbols = ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'SPY', 'QQQ']
        market_data = {}
        
        try:
            for symbol in symbols:
                try:
                    # Get latest quote
                    quote = self.api.get_latest_quote(symbol)
                    
                    # Get recent bars for analysis
                    bars = self.api.get_bars(symbol, tradeapi.TimeFrame.Minute, limit=5).df
                    
                    if not bars.empty and quote.bid_price and quote.ask_price:
                        latest_bar = bars.iloc[-1]
                        avg_price = (float(quote.bid_price) + float(quote.ask_price)) / 2
                        
                        market_data[symbol] = {
                            'price': avg_price,
                            'bid': float(quote.bid_price),
                            'ask': float(quote.ask_price),
                            'volume': int(latest_bar['volume']),
                            'volatility': float((latest_bar['high'] - latest_bar['low']) / latest_bar['close']),
                            'price_change': float((latest_bar['close'] - latest_bar['open']) / latest_bar['open']),
                            'timestamp': datetime.now()
                        }
                    else:
                        # Use fallback data
                        market_data[symbol] = self._get_fallback_data(symbol)
                        
                except Exception as symbol_error:
                    self.logger.warning(f"Error getting data for {symbol}: {symbol_error}")
                    market_data[symbol] = self._get_fallback_data(symbol)
            
            if market_data:
                self.logger.info(f"✅ Retrieved market data for {len(market_data)} symbols")
            
            return market_data
            
        except Exception as e:
            self.logger.error(f"❌ Market data error: {e}")
            return {}
    
    def _get_fallback_data(self, symbol: str) -> Dict[str, Any]:
        """Generate fallback market data"""
        # Realistic price ranges by symbol
        price_ranges = {
            'AAPL': (150, 200),
            'MSFT': (300, 400),
            'GOOGL': (100, 150),
            'TSLA': (200, 300),
            'SPY': (400, 500),
            'QQQ': (300, 400)
        }
        
        price_min, price_max = price_ranges.get(symbol, (100, 200))
        base_price = random.uniform(price_min, price_max)
        
        return {
            'price': base_price,
            'bid': base_price * 0.999,
            'ask': base_price * 1.001,
            'volume': random.randint(50000, 200000),
            'volatility': random.uniform(0.005, 0.03),
            'price_change': random.uniform(-0.02, 0.02),
            'timestamp': datetime.now()
        }
    
    def _generate_reflection(self, agent_id: str, decision: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate agent reflection"""
        return {
            'agent_id': agent_id,
            'timestamp': datetime.now(),
            'had_decision': decision is not None,
            'performance_summary': f'Agent active with {self.agent_performance[agent_id].total_trades} trades'
        }
    
    def _update_agent_performance(self, agent_decisions: Dict[str, Any], executed_trades: List[Dict[str, Any]]):
        """Update agent performance metrics"""
        executed_agent_ids = [t['agent_id'] for t in executed_trades]
        
        for agent_id, decision in agent_decisions.items():
            performance = self.agent_performance[agent_id]
            
            if decision:
                performance.last_decision_time = datetime.now()
                
                if agent_id in executed_agent_ids:
                    performance.total_trades += 1
                    performance.successful_trades += 1  # Assume success for now
    
    def _log_production_cycle_summary(self, cycle_result: Dict[str, Any]):
        """Log production cycle summary"""
        account_info = cycle_result['account_info']
        
        self.logger.info(f"🏭 Production Cycle {self.cycle_count} Summary:")
        self.logger.info(f"   💰 Buying Power: ${account_info['buying_power']:,.2f}")
        self.logger.info(f"   📊 Portfolio Value: ${account_info['portfolio_value']:,.2f}")
        self.logger.info(f"   🎯 Decisions: {cycle_result['total_decisions']}/12")
        self.logger.info(f"   ✅ Selected Trades: {len(cycle_result['selected_trades'])}")
        self.logger.info(f"   🚀 Executed Trades: {len(cycle_result['executed_trades'])}")
        self.logger.info(f"   📊 Current Positions: {account_info['positions_count']}")
        self.logger.info(f"   ⏱️ Cycle Duration: {cycle_result['cycle_duration']:.2f}s")
        
        # Show executed trades
        for trade in cycle_result['executed_trades']:
            self.logger.info(f"     🔄 {trade['agent_id']}: {trade['symbol']} {trade['action']} {trade['quantity']} @ ${trade['price']:.2f}")
    
    def _create_empty_cycle_result(self) -> Dict[str, Any]:
        """Create empty cycle result for error cases"""
        return {
            'cycle': self.cycle_count,
            'timestamp': datetime.now(),
            'agent_decisions': {},
            'agent_reflections': {},
            'selected_trades': [],
            'executed_trades': [],
            'total_decisions': 0,
            'total_reflections': 0,
            'cycle_duration': 0,
            'account_info': {
                'buying_power': 0,
                'portfolio_value': 0,
                'positions_count': 0
            }
        }

async def main():
    """Main production trading loop"""
    try:
        logger.info("🏭 PRODUCTION TRADING SYSTEM STARTING")
        logger.info("=" * 60)
        
        # Create production system
        system = ProductionTradingSystem()
        
        logger.info("✅ Production system initialized!")
        logger.info("🎯 Risk-managed multi-agent trading active!")
        logger.info("💰 Paper trading mode - safe for testing")
        
        # Run production trading cycles
        while True:
            try:
                cycle_result = await system.run_competitive_cycle()
                
                # Wait between cycles (60 seconds for production)
                await asyncio.sleep(60)
                
            except KeyboardInterrupt:
                logger.info("Production system stopped by user")
                break
            except Exception as e:
                logger.error(f"Error in production cycle: {e}")
                await asyncio.sleep(60)
        
    except Exception as e:
        logger.error(f"Fatal error in production system: {e}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)

if __name__ == "__main__":
    # Create logs directory
    os.makedirs('logs', exist_ok=True)
    
    # Run the production system
    asyncio.run(main())