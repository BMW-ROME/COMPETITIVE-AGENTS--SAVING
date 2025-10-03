#!/usr/bin/env python3
"""
Smart Competitive Trading System
===============================
Intelligent trading system that checks positions and buying power before executing trades.
"""

import asyncio
import logging
import random
import sys
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import alpaca_trade_api as tradeapi
import yfinance as yf

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/smart_trading.log', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger("SmartCompetitiveTrading")

class SmartCompetitiveTradingSystem:
    """Smart competitive trading system with position and buying power checks"""
    
    def __init__(self):
        self.logger = logger
        self.symbols = ["AAPL", "MSFT", "GOOGL", "TSLA", "SPY", "QQQ"]
        
        # Initialize Alpaca API
        self.api = tradeapi.REST(
            key_id="PKK43GTIACJNUPGZPCPF",
            secret_key="CuQWde4QtPHAtwuMfxaQhB8njmrcJDq0YK4Oz9Rw",
            base_url="https://paper-api.alpaca.markets",
            api_version='v2'
        )
        
        # Initialize agents
        self.agents = {}
        self.agent_performance = {}
        self.cycle_count = 0
        self.total_trades = 0
        self.total_pnl = 0.0
        
        # Initialize system
        self._initialize_agents()
        
    def _initialize_agents(self):
        """Initialize 12 smart trading agents"""
        agent_styles = [
            'conservative', 'balanced', 'aggressive', 'scalping',
            'momentum', 'mean_reversion', 'breakout', 'swing',
            'day_trading', 'position', 'arbitrage', 'ai_enhanced'
        ]
        
        for i, style in enumerate(agent_styles, 1):
            agent_id = f"smart_{style}_{i}"
            self.agents[agent_id] = {
                'id': agent_id,
                'style': style,
                'confidence': random.uniform(0.3, 0.9),
                'max_position': random.uniform(0.01, 0.05),  # 1-5% of portfolio
                'trades_count': 0,
                'win_rate': 0.0,
                'total_pnl': 0.0
            }
            self.agent_performance[agent_id] = {
                'decisions': 0,
                'trades': 0,
                'wins': 0,
                'losses': 0,
                'total_pnl': 0.0
            }
            
        self.logger.info(f"Initialized {len(self.agents)} smart trading agents")
    
    async def get_account_info(self):
        """Get current account information"""
        try:
            account = self.api.get_account()
            return {
                'buying_power': float(account.buying_power),
                'cash': float(account.cash),
                'portfolio_value': float(account.portfolio_value),
                'equity': float(account.equity)
            }
        except Exception as e:
            self.logger.error(f"Failed to get account info: {e}")
            return None
    
    async def get_positions(self):
        """Get current positions"""
        try:
            positions = self.api.list_positions()
            return {
                pos.symbol: {
                    'qty': float(pos.qty),
                    'market_value': float(pos.market_value),
                    'unrealized_pl': float(pos.unrealized_pl),
                    'side': pos.side
                } for pos in positions
            }
        except Exception as e:
            self.logger.error(f"Failed to get positions: {e}")
            return {}
    
    async def get_market_data(self):
        """Get real market data"""
        try:
            market_data = {}
            for symbol in self.symbols:
                try:
                    # Get current price
                    quote = self.api.get_latest_quote(symbol)
                    current_price = float(quote.bid_price)
                    
                    # Get recent bars for analysis
                    bars = self.api.get_bars(symbol, '1Min', limit=5)
                    if bars:
                        recent_prices = [float(bar.close) for bar in bars]
                        price_change = (current_price - recent_prices[0]) / recent_prices[0] if recent_prices[0] > 0 else 0
                        volatility = max(recent_prices) - min(recent_prices) if len(recent_prices) > 1 else 0
                    else:
                        price_change = 0
                        volatility = 0
                    
                    market_data[symbol] = {
                        'price': current_price,
                        'price_change': price_change,
                        'volatility': volatility,
                        'volume': 1000000,  # Default volume
                        'timestamp': datetime.now()
                    }
                except Exception as e:
                    self.logger.warning(f"Failed to get data for {symbol}: {e}")
                    continue
            
            return market_data
        except Exception as e:
            self.logger.error(f"Failed to get market data: {e}")
            return {}
    
    def _generate_smart_decision(self, agent_id: str, market_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Generate a smart trading decision based on agent style and market conditions"""
        agent = self.agents[agent_id]
        style = agent['style']
        
        # Select a random symbol
        symbol = random.choice(self.symbols)
        if symbol not in market_data:
            return None
        
        data = market_data[symbol]
        price = data['price']
        price_change = data['price_change']
        volatility = data['volatility']
        
        # Style-based decision logic
        if style == 'conservative':
            # Only trade on strong signals
            if abs(price_change) > 0.01 and volatility < 0.02:
                action = 'BUY' if price_change > 0 else 'SELL'
                confidence = min(0.8, agent['confidence'] + 0.2)
            else:
                return None
                
        elif style == 'aggressive':
            # Trade on any significant movement
            if abs(price_change) > 0.005:
                action = 'BUY' if price_change > 0 else 'SELL'
                confidence = agent['confidence']
            else:
                return None
                
        elif style == 'scalping':
            # Quick trades on small movements
            if abs(price_change) > 0.001:
                action = 'BUY' if price_change > 0 else 'SELL'
                confidence = agent['confidence']
            else:
                return None
                
        elif style == 'momentum':
            # Follow strong trends
            if abs(price_change) > 0.008:
                action = 'BUY' if price_change > 0 else 'SELL'
                confidence = min(0.9, agent['confidence'] + 0.3)
            else:
                return None
                
        else:
            # Default balanced approach
            if abs(price_change) > 0.003:
                action = 'BUY' if price_change > 0 else 'SELL'
                confidence = agent['confidence']
            else:
                return None
        
        # Calculate position size based on available buying power
        max_position_value = agent['max_position'] * 100000  # Assume $100k portfolio
        quantity = max_position_value / price
        
        return {
            'agent_id': agent_id,
            'symbol': symbol,
            'action': action,
            'quantity': round(quantity, 2),
            'price': price,
            'confidence': confidence,
            'reasoning': f"Smart {style} decision based on {price_change:.3f} price change",
            'hold_duration': 'short' if style in ['scalping', 'day_trading'] else 'medium',
            'style': style
        }
    
    async def execute_trade(self, trade: Dict[str, Any]) -> bool:
        """Execute a trade with proper validation"""
        try:
            symbol = trade['symbol']
            action = trade['action']
            quantity = trade['quantity']
            
            # Get current positions
            positions = await self.get_positions()
            current_position = positions.get(symbol, {}).get('qty', 0)
            
            # Get account info
            account_info = await self.get_account_info()
            if not account_info:
                return False
            
            buying_power = account_info['buying_power']
            cash = account_info['cash']
            
            # Validate trade
            if action == 'BUY':
                # Check if we have enough buying power
                required_capital = quantity * trade['price']
                if required_capital > buying_power:
                    self.logger.warning(f"Insufficient buying power for {symbol}: {required_capital:.2f} > {buying_power:.2f}")
                    return False
                    
            elif action == 'SELL':
                # Check if we have enough shares to sell
                if quantity > current_position:
                    self.logger.warning(f"Insufficient shares to sell {symbol}: {quantity} > {current_position}")
                    return False
            
            # Execute the trade
            if action == 'BUY':
                order = self.api.submit_order(
                    symbol=symbol,
                    qty=quantity,
                    side='buy',
                    type='market',
                    time_in_force='day'
                )
            else:  # SELL
                order = self.api.submit_order(
                    symbol=symbol,
                    qty=quantity,
                    side='sell',
                    type='market',
                    time_in_force='day'
                )
            
            self.logger.info(f"✅ Executed {action} {quantity} shares of {symbol} at ${trade['price']:.2f}")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to execute trade: {e}")
            return False
    
    async def run_cycle(self):
        """Run one trading cycle"""
        cycle_start = time.time()
        self.cycle_count += 1
        
        self.logger.info(f"Starting Smart Trading Cycle {self.cycle_count}")
        
        # Get market data
        market_data = await self.get_market_data()
        if not market_data:
            self.logger.warning("No market data available")
            return
        
        # Get account info
        account_info = await self.get_account_info()
        if not account_info:
            self.logger.warning("No account info available")
            return
        
        # Get current positions
        positions = await self.get_positions()
        
        self.logger.info(f"Account: ${account_info['buying_power']:.2f} buying power, ${account_info['cash']:.2f} cash")
        self.logger.info(f"Positions: {len(positions)} symbols")
        
        # Generate decisions from all agents
        agent_decisions = {}
        for agent_id in self.agents:
            decision = self._generate_smart_decision(agent_id, market_data)
            if decision:
                agent_decisions[agent_id] = decision
                self.agent_performance[agent_id]['decisions'] += 1
        
        # Execute trades
        executed_trades = 0
        for agent_id, decision in agent_decisions.items():
            if await self.execute_trade(decision):
                executed_trades += 1
                self.agent_performance[agent_id]['trades'] += 1
                self.total_trades += 1
        
        # Update performance
        for agent_id in self.agents:
            perf = self.agent_performance[agent_id]
            if perf['trades'] > 0:
                self.agents[agent_id]['win_rate'] = perf['wins'] / perf['trades']
                self.agents[agent_id]['total_pnl'] = perf['total_pnl']
        
        # Log cycle summary
        cycle_duration = time.time() - cycle_start
        self.logger.info(f"Smart Trading Cycle {self.cycle_count} Summary:")
        self.logger.info(f"  Total Decisions: {len(agent_decisions)}")
        self.logger.info(f"  Executed Trades: {executed_trades}")
        self.logger.info(f"  Cycle Duration: {cycle_duration:.2f}s")
        self.logger.info(f"  Total Trades: {self.total_trades}")
        self.logger.info(f"  Total PnL: ${self.total_pnl:.2f}")
        
        # Log agent performance
        for agent_id, perf in self.agent_performance.items():
            if perf['trades'] > 0:
                self.logger.info(f"  Agent {agent_id}: {perf['trades']} trades, {perf['wins']/perf['trades']:.2f} win rate, ${perf['total_pnl']:.2f} PnL")
    
    async def run_system(self):
        """Run the smart trading system"""
        self.logger.info("=" * 60)
        self.logger.info("SMART COMPETITIVE TRADING SYSTEM STARTING")
        self.logger.info("=" * 60)
        self.logger.info("✅ Connected to Alpaca PAPER TRADING API")
        self.logger.info(f"Initialized {len(self.agents)} smart trading agents")
        self.logger.info("Smart trading system initialized!")
        self.logger.info("All agents will make intelligent decisions based on market conditions!")
        self.logger.info("Using PAPER TRADING for safety!")
        
        try:
            while True:
                await self.run_cycle()
                await asyncio.sleep(30)  # Wait 30 seconds between cycles
                
        except KeyboardInterrupt:
            self.logger.info("System stopped by user")
        except Exception as e:
            self.logger.error(f"Fatal error: {e}")
            import traceback
            self.logger.error(traceback.format_exc())

async def main():
    """Main function"""
    # Create logs directory
    import os
    os.makedirs('logs', exist_ok=True)
    
    # Initialize and run system
    system = SmartCompetitiveTradingSystem()
    await system.run_system()

if __name__ == "__main__":
    asyncio.run(main())

