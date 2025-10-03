#!/usr/bin/env python3
"""
Fixed Ultra Aggressive Trading System
====================================
Fixes the issues with wash trades and insufficient buying power.
"""

import asyncio
import logging
import random
import sys
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import alpaca_trade_api as tradeapi

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/fixed_ultra_aggressive.log', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger("FixedUltraAggressive")

class FixedUltraAggressiveTradingSystem:
    """Fixed ultra aggressive trading system that checks positions and buying power"""
    
    def __init__(self):
        self.logger = logger
        self.symbols = ["AAPL", "MSFT", "GOOGL", "TSLA", "SPY", "QQQ"]
        
        # Initialize Alpaca API
        self.api = tradeapi.REST(
            key_id=os.getenv("APCA_API_KEY_ID", "PK5CVK31GSEWD3ZT0XH7"),
            secret_key=os.getenv("APCA_API_SECRET_KEY", "ryRQD6VmjY14UFE57ess9pvH1b5eorkXuwV3SDqk"),
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
        """Initialize 12 ultra aggressive agents"""
        for i in range(1, 13):
            agent_id = f"fixed_ultra_aggressive_{i}"
            self.agents[agent_id] = {
                'id': agent_id,
                'style': 'ultra_aggressive',
                'confidence': 0.99,
                'max_position': 0.02,  # 2% of portfolio
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
            
        self.logger.info(f"Initialized {len(self.agents)} FIXED ultra aggressive agents")
    
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
    
    def _generate_fixed_decision(self, agent_id: str, market_data: Dict[str, Any], positions: Dict[str, Any], account_info: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Generate a FIXED trading decision that checks positions and buying power"""
        # Select a random symbol
        symbol = random.choice(self.symbols)
        if symbol not in market_data:
            return None
        
        data = market_data[symbol]
        price = data['price']
        price_change = data['price_change']
        
        # Get current position for this symbol
        current_position = positions.get(symbol, {}).get('qty', 0)
        buying_power = account_info['buying_power']
        
        # Decide action based on position and buying power
        if current_position > 0:
            # We have shares, consider selling
            if random.random() < 0.7:  # 70% chance to sell
                action = 'SELL'
                quantity = min(current_position, random.uniform(0.1, 0.5))  # Sell 10-50% of position
            else:
                return None  # Don't trade
        else:
            # No position, consider buying if we have buying power
            if buying_power > price * 10:  # Only buy if we have enough buying power for at least 10 shares
                if random.random() < 0.8:  # 80% chance to buy
                    action = 'BUY'
                    max_shares = int(buying_power / price * 0.02)  # Use 2% of buying power
                    quantity = min(max_shares, random.uniform(1, 5))  # Buy 1-5 shares
                else:
                    return None
            else:
                return None  # Not enough buying power
        
        return {
            'agent_id': agent_id,
            'symbol': symbol,
            'action': action,
            'quantity': round(quantity, 2),
            'price': price,
            'confidence': 0.99,
            'reasoning': f"FIXED ULTRA AGGRESSIVE - {action} {quantity} shares of {symbol}",
            'hold_duration': 'short',
            'style': 'ultra_aggressive'
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
            
            # Final validation
            if action == 'BUY':
                required_capital = quantity * trade['price']
                if required_capital > buying_power:
                    self.logger.warning(f"Insufficient buying power for {symbol}: {required_capital:.2f} > {buying_power:.2f}")
                    return False
                    
            elif action == 'SELL':
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
        
        self.logger.info(f"Starting FIXED Ultra Aggressive Cycle {self.cycle_count}")
        
        # Get market data
        market_data = await self.get_market_data()
        if not market_data:
            self.logger.warning("No market data available")
            return
        
        # Get account info and positions
        account_info = await self.get_account_info()
        if not account_info:
            self.logger.warning("No account info available")
            return
        
        positions = await self.get_positions()
        
        self.logger.info(f"Account: ${account_info['buying_power']:.2f} buying power, ${account_info['cash']:.2f} cash")
        self.logger.info(f"Positions: {len(positions)} symbols")
        
        # Generate decisions from all agents
        agent_decisions = {}
        for agent_id in self.agents:
            decision = self._generate_fixed_decision(agent_id, market_data, positions, account_info)
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
        self.logger.info(f"FIXED Ultra Aggressive Cycle {self.cycle_count} Summary:")
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
        """Run the fixed ultra aggressive trading system"""
        self.logger.info("=" * 60)
        self.logger.info("FIXED ULTRA AGGRESSIVE TRADING SYSTEM STARTING")
        self.logger.info("=" * 60)
        self.logger.info("✅ Connected to Alpaca PAPER TRADING API")
        self.logger.info(f"Initialized {len(self.agents)} FIXED ultra aggressive agents")
        self.logger.info("FIXED ultra aggressive trading system initialized!")
        self.logger.info("All agents will make SMART decisions with position checks!")
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
    system = FixedUltraAggressiveTradingSystem()
    await system.run_system()

if __name__ == "__main__":
    asyncio.run(main())

