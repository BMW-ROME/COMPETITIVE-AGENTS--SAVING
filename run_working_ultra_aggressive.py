#!/usr/bin/env python3
"""
Working Ultra Aggressive Trading System
======================================
A completely new, working ultra aggressive trading system with proper logic.
"""

import asyncio
import logging
import random
import sys
import time
import os
from datetime import datetime
from typing import Dict, Optional, Any
import alpaca_trade_api as tradeapi

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/working_ultra_aggressive.log', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger("WorkingUltraAggressive")

class WorkingUltraAggressiveTradingSystem:
    """Working ultra aggressive trading system with proper logic"""
    
    def __init__(self):
        self.logger = logger
        self.symbols = ["AAPL", "MSFT", "GOOGL", "TSLA", "SPY", "QQQ"]
        
        # Initialize Alpaca API with error handling
        try:
            self.api = tradeapi.REST(
                key_id=os.getenv("ALPACA_API_KEY", "PKK43GTIACJNUPGZPCPF"),
                secret_key=os.getenv("ALPACA_SECRET_KEY", "CuQWde4QtPHAtwuMfxaQhB8njmrcJDq0YK4Oz9Rw"),
                base_url="https://paper-api.alpaca.markets",
                api_version='v2'
            )
            # Test connection
            account = self.api.get_account()
            self.logger.info(f"✅ Connected to Alpaca PAPER TRADING API - Account: {account.id}")
        except Exception as e:
            self.logger.error(f"❌ Failed to connect to Alpaca API: {e}")
            raise
        
        # Initialize agents
        self.agents = {}
        self.agent_performance = {}
        self.cycle_count = 0
        self.total_trades = 0
        self.total_pnl = 0.0
        self.max_retries = 3
        self.retry_delay = 3
        
        # Initialize system
        self._initialize_agents()
        
    def _initialize_agents(self):
        """Initialize 6 working ultra aggressive agents"""
        aggression_levels = [
            {'level': 'extreme', 'confidence': 0.95, 'max_position': 0.1, 'trade_probability': 0.8},
            {'level': 'ultra', 'confidence': 0.90, 'max_position': 0.08, 'trade_probability': 0.75},
            {'level': 'hyper', 'confidence': 0.85, 'max_position': 0.06, 'trade_probability': 0.70},
            {'level': 'maximum', 'confidence': 0.88, 'max_position': 0.07, 'trade_probability': 0.72},
            {'level': 'extreme', 'confidence': 0.92, 'max_position': 0.09, 'trade_probability': 0.78},
            {'level': 'ultra', 'confidence': 0.87, 'max_position': 0.05, 'trade_probability': 0.68}
        ]
        
        for i, config in enumerate(aggression_levels, 1):
            agent_id = f"working_ultra_{config['level']}_{i}"
            self.agents[agent_id] = {
                'id': agent_id,
                'style': 'ultra_aggressive',
                'aggression_level': config['level'],
                'confidence': config['confidence'],
                'max_position': config['max_position'],
                'trade_probability': config['trade_probability'],
                'trades_count': 0,
                'win_rate': 0.0,
                'total_pnl': 0.0,
                'last_trade_time': None,
                'cooldown_seconds': 60  # Longer cooldown to avoid wash trades
            }
            self.agent_performance[agent_id] = {
                'decisions': 0,
                'trades': 0,
                'wins': 0,
                'losses': 0,
                'total_pnl': 0.0,
                'success_rate': 0.0
            }
            
        self.logger.info(f"Initialized {len(self.agents)} WORKING ultra aggressive agents")
    
    async def get_account_info(self) -> Optional[Dict[str, Any]]:
        """Get current account information with retry logic"""
        for attempt in range(self.max_retries):
            try:
                account = self.api.get_account()
                return {
                    'buying_power': float(account.buying_power),
                    'cash': float(account.cash),
                    'portfolio_value': float(account.portfolio_value),
                    'equity': float(account.equity),
                    'day_trade_count': int(getattr(account, 'day_trade_count', 0)),
                    'pattern_day_trader': getattr(account, 'pattern_day_trader', False)
                }
            except Exception as e:
                self.logger.warning(f"Account info attempt {attempt + 1} failed: {e}")
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(self.retry_delay)
                else:
                    self.logger.error(f"Failed to get account info after {self.max_retries} attempts")
                    return None
    
    async def get_positions(self) -> Dict[str, Dict[str, Any]]:
        """Get current positions with retry logic"""
        for attempt in range(self.max_retries):
            try:
                positions = self.api.list_positions()
                return {
                    pos.symbol: {
                        'qty': float(pos.qty),
                        'market_value': float(pos.market_value),
                        'unrealized_pl': float(pos.unrealized_pl),
                        'side': pos.side,
                        'avg_fill_price': float(getattr(pos, 'avg_fill_price', 0.0))
                    } for pos in positions
                }
            except Exception as e:
                self.logger.warning(f"Positions attempt {attempt + 1} failed: {e}")
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(self.retry_delay)
                else:
                    self.logger.error(f"Failed to get positions after {self.max_retries} attempts")
                    return {}
    
    async def get_market_data(self) -> Dict[str, Dict[str, Any]]:
        """Get real market data with comprehensive error handling"""
        market_data = {}
        
        for symbol in self.symbols:
            try:
                # Get current price with retry logic
                for attempt in range(self.max_retries):
                    try:
                        quote = self.api.get_latest_quote(symbol)
                        current_price = float(quote.bid_price)
                        break
                    except Exception as e:
                        if attempt < self.max_retries - 1:
                            await asyncio.sleep(1)
                            continue
                        else:
                            raise e
                
                # Get recent bars for analysis
                try:
                    bars = self.api.get_bars(symbol, '1Min', limit=5)
                    if bars:
                        recent_prices = [float(bar.close) for bar in bars]
                        price_change = (current_price - recent_prices[0]) / recent_prices[0] if recent_prices[0] > 0 else 0
                        volatility = (max(recent_prices) - min(recent_prices)) / recent_prices[0] if recent_prices[0] > 0 else 0
                    else:
                        price_change = 0
                        volatility = 0
                except Exception as e:
                    self.logger.warning(f"Failed to get bars for {symbol}: {e}")
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
    
    def _generate_working_decision(self, agent_id: str, market_data: Dict[str, Any], 
                                 positions: Dict[str, Any], account_info: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Generate a WORKING trading decision with proper logic"""
        agent = self.agents[agent_id]
        
        # Check cooldown period
        if agent['last_trade_time']:
            time_since_last_trade = (datetime.now() - agent['last_trade_time']).total_seconds()
            if time_since_last_trade < agent['cooldown_seconds']:
                return None
        
        # WORKING: 80% chance to make a decision (reduced from 99%)
        if random.random() > agent['trade_probability']:
            return None
        
        # Select a random symbol
        symbol = random.choice(self.symbols)
        if symbol not in market_data:
            return None
        
        data = market_data[symbol]
        price = data['price']
        
        # Get current position for this symbol
        current_position = positions.get(symbol, {}).get('qty', 0)
        buying_power = account_info['buying_power']
        
        # WORKING decision logic - SIMPLIFIED AND FIXED
        action = None
        quantity = 0
        
        # Determine action based on current position and buying power
        if current_position > 0:
            # We have shares, consider selling (60% chance)
            if random.random() < 0.6:
                action = 'SELL'
                # FINAL FIX: Handle fractional shares properly
                if current_position >= 1.0:
                    quantity = 1  # Sell 1 whole share
                else:
                    # For fractional shares, don't try to sell
                    return None
                # Ensure we don't try to sell more than we have
                if quantity > current_position:
                    return None
        else:
            # No position, consider buying if we have buying power
            if buying_power > price * 2:  # Only buy if we have enough for at least 2 shares
                if random.random() < 0.7:  # 70% chance to buy
                    action = 'BUY'
                    # WORKING FIX: Buy only 1-2 shares at a time
                    max_affordable = int(buying_power / price)
                    quantity = min(2, max(1, random.randint(1, 2)))  # Buy 1-2 shares max
                    
                    # Ensure we don't exceed buying power
                    if quantity * price > buying_power * 0.5:  # Use max 50% of buying power
                        quantity = max(1, int(buying_power * 0.5 / price))
        
        if not action or quantity <= 0:
            return None
        
        # Final validation
        if action == 'BUY':
            required_capital = quantity * price
            if required_capital > buying_power * 0.5:  # Use max 50% of buying power
                return None
        elif action == 'SELL':
            if quantity > current_position:
                return None
        
        return {
            'agent_id': agent_id,
            'symbol': symbol,
            'action': action,
            'quantity': quantity,
            'price': price,
            'confidence': agent['confidence'],
            'reasoning': f"WORKING ULTRA AGGRESSIVE {agent['aggression_level']} - {action} {quantity} shares of {symbol}",
            'hold_duration': 'short',
            'style': 'ultra_aggressive'
        }
    
    async def execute_trade(self, trade: Dict[str, Any]) -> bool:
        """Execute a trade with comprehensive validation and error handling"""
        try:
            symbol = trade['symbol']
            action = trade['action']
            quantity = trade['quantity']
            
            # Get current positions and account info
            positions = await self.get_positions()
            account_info = await self.get_account_info()
            
            if not account_info:
                self.logger.error("Cannot execute trade: No account info")
                return False
            
            current_position = positions.get(symbol, {}).get('qty', 0)
            buying_power = account_info['buying_power']
            
            # Final validation
            if action == 'BUY':
                required_capital = quantity * trade['price']
                if required_capital > buying_power * 0.5:  # Use max 50% of buying power
                    self.logger.warning(f"Insufficient buying power for {symbol}: {required_capital:.2f} > {buying_power * 0.5:.2f}")
                    return False
                    
            elif action == 'SELL':
                if quantity > current_position:
                    self.logger.warning(f"Insufficient shares to sell {symbol}: {quantity} > {current_position}")
                    return False
            
            # Execute the trade with retry logic and delays
            for attempt in range(self.max_retries):
                try:
                    # Add delay to avoid wash trade detection
                    if attempt > 0:
                        await asyncio.sleep(2)  # Longer delay
                    
                    if action == 'BUY':
                        self.api.submit_order(
                            symbol=symbol,
                            qty=quantity,
                            side='buy',
                            type='market',
                            time_in_force='day'
                        )
                    else:  # SELL
                        self.api.submit_order(
                            symbol=symbol,
                            qty=quantity,
                            side='sell',
                            type='market',
                            time_in_force='day'
                        )
                    
                    self.logger.info(f"✅ WORKING ULTRA AGGRESSIVE: Executed {action} {quantity} shares of {symbol} at ${trade['price']:.2f}")
                    
                    # Update agent's last trade time
                    self.agents[trade['agent_id']]['last_trade_time'] = datetime.now()
                    
                    return True
                    
                except Exception as e:
                    self.logger.warning(f"Trade execution attempt {attempt + 1} failed: {e}")
                    if attempt < self.max_retries - 1:
                        await asyncio.sleep(3)  # Longer delay between retries
                    else:
                        raise e
            
        except Exception as e:
            self.logger.error(f"❌ Failed to execute WORKING ULTRA AGGRESSIVE trade: {e}")
            return False
    
    async def run_cycle(self):
        """Run one working ultra aggressive trading cycle"""
        cycle_start = time.time()
        self.cycle_count += 1
        
        self.logger.info(f"Starting WORKING Ultra Aggressive Cycle {self.cycle_count}")
        
        try:
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
                try:
                    decision = self._generate_working_decision(agent_id, market_data, positions, account_info)
                    if decision:
                        agent_decisions[agent_id] = decision
                        self.agent_performance[agent_id]['decisions'] += 1
                except Exception as e:
                    self.logger.warning(f"Decision generation failed for {agent_id}: {e}")
                    continue
            
            # Execute trades
            executed_trades = 0
            for agent_id, decision in agent_decisions.items():
                try:
                    if await self.execute_trade(decision):
                        executed_trades += 1
                        self.agent_performance[agent_id]['trades'] += 1
                        self.total_trades += 1
                except Exception as e:
                    self.logger.warning(f"Trade execution failed for {agent_id}: {e}")
                    continue
            
            # Update performance metrics
            for agent_id in self.agents:
                perf = self.agent_performance[agent_id]
                if perf['trades'] > 0:
                    self.agents[agent_id]['win_rate'] = perf['wins'] / perf['trades']
                    self.agents[agent_id]['total_pnl'] = perf['total_pnl']
                    perf['success_rate'] = perf['wins'] / perf['trades']
            
            # Log cycle summary
            cycle_duration = time.time() - cycle_start
            self.logger.info(f"WORKING Ultra Aggressive Cycle {self.cycle_count} Summary:")
            self.logger.info(f"  Total Decisions: {len(agent_decisions)}")
            self.logger.info(f"  Executed Trades: {executed_trades}")
            self.logger.info(f"  Cycle Duration: {cycle_duration:.2f}s")
            self.logger.info(f"  Total Trades: {self.total_trades}")
            self.logger.info(f"  Total PnL: ${self.total_pnl:.2f}")
            
            # Log top performing agents
            top_agents = sorted(self.agent_performance.items(), 
                             key=lambda x: x[1]['success_rate'], reverse=True)[:3]
            for agent_id, perf in top_agents:
                if perf['trades'] > 0:
                    self.logger.info(f"  Top Agent {agent_id}: {perf['trades']} trades, {perf['success_rate']:.2f} success rate, ${perf['total_pnl']:.2f} PnL")
                    
        except Exception as e:
            self.logger.error(f"Cycle {self.cycle_count} failed: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
    
    async def run_system(self):
        """Run the working ultra aggressive trading system"""
        self.logger.info("=" * 60)
        self.logger.info("WORKING ULTRA AGGRESSIVE TRADING SYSTEM STARTING")
        self.logger.info("=" * 60)
        self.logger.info("✅ Connected to Alpaca PAPER TRADING API")
        self.logger.info(f"Initialized {len(self.agents)} WORKING ultra aggressive agents")
        self.logger.info("WORKING ultra aggressive trading system initialized!")
        self.logger.info("All agents will make WORKING ULTRA AGGRESSIVE decisions with proper logic!")
        self.logger.info("Using PAPER TRADING for safety!")
        
        try:
            while True:
                await self.run_cycle()
                await asyncio.sleep(30)  # 30 second cycles for stability
                
        except KeyboardInterrupt:
            self.logger.info("System stopped by user")
        except Exception as e:
            self.logger.error(f"Fatal error: {e}")
            import traceback
            self.logger.error(traceback.format_exc())

async def main():
    """Main function"""
    # Create logs directory
    os.makedirs('logs', exist_ok=True)
    
    # Initialize and run system
    system = WorkingUltraAggressiveTradingSystem()
    await system.run_system()

if __name__ == "__main__":
    asyncio.run(main())
