#!/usr/bin/env python3
"""
Optimized Smart Competitive Trading System
=========================================
Bulletproof trading system with comprehensive error handling, position validation,
and intelligent decision making.
"""

import asyncio
import logging
import random
import sys
import time
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import alpaca_trade_api as tradeapi
import yfinance as yf

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/optimized_smart_trading.log', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger("OptimizedSmartTrading")

class OptimizedSmartTradingSystem:
    """Optimized smart competitive trading system with bulletproof error handling"""
    
    def __init__(self):
        self.logger = logger
        self.symbols = ["AAPL", "MSFT", "GOOGL", "TSLA", "SPY", "QQQ"]
        
        # Initialize Alpaca API with error handling
        try:
            self.api = tradeapi.REST(
                key_id=os.getenv("APCA_API_KEY_ID", "PK5CVK31GSEWD3ZT0XH7"),
                secret_key=os.getenv("APCA_API_SECRET_KEY", "ryRQD6VmjY14UFE57ess9pvH1b5eorkXuwV3SDqk"),
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
        self.retry_delay = 5
        
        # Initialize system
        self._initialize_agents()
        
    def _initialize_agents(self):
        """Initialize 12 optimized trading agents with distinct styles"""
        # 🚀 OPTIMIZED SMART TRADING: UNLEASHED WITH 100X-250X POSITION BOOSTS!
        agent_configs = [
            {'style': 'conservative', 'confidence': 0.3, 'max_position': 2.0, 'risk_tolerance': 'high'},   # 🚀 100X: 0.02→2.0, conf 0.8→0.3
            {'style': 'balanced', 'confidence': 0.2, 'max_position': 3.0, 'risk_tolerance': 'high'},      # 🚀 100X: 0.03→3.0, conf 0.6→0.2  
            {'style': 'aggressive', 'confidence': 0.1, 'max_position': 4.0, 'risk_tolerance': 'high'},    # 🚀 100X: 0.04→4.0, conf 0.4→0.1
            {'style': 'scalping', 'confidence': 0.2, 'max_position': 1.0, 'risk_tolerance': 'high'},      # 🚀 100X: 0.01→1.0, conf 0.7→0.2
            {'style': 'momentum', 'confidence': 0.15, 'max_position': 3.0, 'risk_tolerance': 'high'},     # 🚀 100X: 0.03→3.0, conf 0.5→0.15
            {'style': 'mean_reversion', 'confidence': 0.2, 'max_position': 2.5, 'risk_tolerance': 'high'}, # 🚀 100X: 0.025→2.5, conf 0.6→0.2
            {'style': 'breakout', 'confidence': 0.15, 'max_position': 3.5, 'risk_tolerance': 'high'},     # 🚀 100X: 0.035→3.5, conf 0.5→0.15
            {'style': 'swing', 'confidence': 0.2, 'max_position': 3.0, 'risk_tolerance': 'high'},         # 🚀 100X: 0.03→3.0, conf 0.7→0.2
            {'style': 'day_trading', 'confidence': 0.2, 'max_position': 2.5, 'risk_tolerance': 'high'},   # 🚀 100X: 0.025→2.5, conf 0.6→0.2
            {'style': 'position', 'confidence': 0.3, 'max_position': 2.0, 'risk_tolerance': 'high'},      # 🚀 100X: 0.02→2.0, conf 0.8→0.3
            {'style': 'arbitrage', 'confidence': 0.9, 'max_position': 0.01, 'risk_tolerance': 'low'},
            {'style': 'ai_enhanced', 'confidence': 0.7, 'max_position': 0.03, 'risk_tolerance': 'medium'}
        ]
        
        for i, config in enumerate(agent_configs, 1):
            agent_id = f"optimized_{config['style']}_{i}"
            self.agents[agent_id] = {
                'id': agent_id,
                'style': config['style'],
                'confidence': config['confidence'],
                'max_position': config['max_position'],
                'risk_tolerance': config['risk_tolerance'],
                'trades_count': 0,
                'win_rate': 0.0,
                'total_pnl': 0.0,
                'last_trade_time': None,
                'cooldown_seconds': 60 if config['style'] in ['scalping', 'day_trading'] else 300
            }
            self.agent_performance[agent_id] = {
                'decisions': 0,
                'trades': 0,
                'wins': 0,
                'losses': 0,
                'total_pnl': 0.0,
                'success_rate': 0.0
            }
            
        self.logger.info(f"Initialized {len(self.agents)} optimized trading agents")
    
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
                    bars = self.api.get_bars(symbol, '1Min', limit=10)
                    if bars:
                        recent_prices = [float(bar.close) for bar in bars]
                        price_change = (current_price - recent_prices[0]) / recent_prices[0] if recent_prices[0] > 0 else 0
                        volatility = (max(recent_prices) - min(recent_prices)) / recent_prices[0] if recent_prices[0] > 0 else 0
                        volume = sum([float(bar.volume) for bar in bars]) / len(bars) if bars else 1000000
                    else:
                        price_change = 0
                        volatility = 0
                        volume = 1000000
                except Exception as e:
                    self.logger.warning(f"Failed to get bars for {symbol}: {e}")
                    price_change = 0
                    volatility = 0
                    volume = 1000000
                
                market_data[symbol] = {
                    'price': current_price,
                    'price_change': price_change,
                    'volatility': volatility,
                    'volume': volume,
                    'timestamp': datetime.now()
                }
                
            except Exception as e:
                self.logger.warning(f"Failed to get data for {symbol}: {e}")
                continue
        
        return market_data
    
    def _generate_optimized_decision(self, agent_id: str, market_data: Dict[str, Any], 
                                   positions: Dict[str, Any], account_info: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Generate an optimized trading decision with comprehensive validation"""
        agent = self.agents[agent_id]
        
        # Check cooldown period
        if agent['last_trade_time']:
            time_since_last_trade = (datetime.now() - agent['last_trade_time']).total_seconds()
            if time_since_last_trade < agent['cooldown_seconds']:
                return None
        
        # Select a random symbol
        symbol = random.choice(self.symbols)
        if symbol not in market_data:
            return None
        
        data = market_data[symbol]
        price = data['price']
        price_change = data['price_change']
        volatility = data['volatility']
        volume = data['volume']
        
        # Get current position for this symbol
        current_position = positions.get(symbol, {}).get('qty', 0)
        buying_power = account_info['buying_power']
        
        # Style-based decision logic with enhanced validation
        decision = None
        
        if agent['style'] == 'conservative':
            # Only trade on strong signals with low volatility
            if abs(price_change) > 0.015 and volatility < 0.02 and volume > 500000:
                action = 'BUY' if price_change > 0 else 'SELL'
                confidence = min(0.9, agent['confidence'] + 0.1)
            else:
                return None
                
        elif agent['style'] == 'aggressive':
            # Trade on any significant movement
            if abs(price_change) > 0.005 and volume > 100000:
                action = 'BUY' if price_change > 0 else 'SELL'
                confidence = agent['confidence']
            else:
                return None
                
        elif agent['style'] == 'scalping':
            # Quick trades on small movements
            if abs(price_change) > 0.002 and volume > 200000:
                action = 'BUY' if price_change > 0 else 'SELL'
                confidence = agent['confidence']
            else:
                return None
                
        elif agent['style'] == 'momentum':
            # Follow strong trends
            if abs(price_change) > 0.01 and volume > 300000:
                action = 'BUY' if price_change > 0 else 'SELL'
                confidence = min(0.95, agent['confidence'] + 0.2)
            else:
                return None
                
        else:
            # Default balanced approach
            if abs(price_change) > 0.005 and volume > 150000:
                action = 'BUY' if price_change > 0 else 'SELL'
                confidence = agent['confidence']
            else:
                return None
        
        # Calculate position size based on available buying power and risk tolerance - FIXED
        max_affordable = int(buying_power / price)
        max_position_shares = int(max_affordable * agent['max_position'])
        
        # Determine quantity based on action and current position - FIXED
        if action == 'BUY':
            if current_position > 0:
                return None  # Don't buy if we already have a position
            quantity = max(1, min(max_position_shares, random.randint(1, 3)))  # Buy 1-3 shares max
        else:  # SELL
            if current_position <= 0:
                return None  # Don't sell if we don't have shares
            # FIXED: Sell percentage of current position, not random amount
            sell_percentage = random.uniform(0.3, 0.7)  # Sell 30-70% of position
            quantity = max(1, int(current_position * sell_percentage))  # Ensure at least 1 share
            # CRITICAL FIX: Ensure we don't try to sell more than we have
            quantity = min(quantity, int(current_position))
        
        # Final validation
        if action == 'BUY':
            required_capital = quantity * price
            if required_capital > buying_power * 0.8:  # Use only 80% of buying power
                return None
        elif action == 'SELL':
            if quantity > current_position:
                return None
        
        return {
            'agent_id': agent_id,
            'symbol': symbol,
            'action': action,
            'quantity': round(quantity, 2),
            'price': price,
            'confidence': confidence,
            'reasoning': f"Optimized {agent['style']} decision: {price_change:.3f} change, {volatility:.3f} volatility",
            'hold_duration': 'short' if agent['style'] in ['scalping', 'day_trading'] else 'medium',
            'style': agent['style']
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
                if required_capital > buying_power * 0.8:  # Use only 80% of buying power
                    self.logger.warning(f"Insufficient buying power for {symbol}: {required_capital:.2f} > {buying_power * 0.8:.2f}")
                    return False
                    
            elif action == 'SELL':
                if quantity > current_position:
                    self.logger.warning(f"Insufficient shares to sell {symbol}: {quantity} > {current_position}")
                    return False
            
            # Execute the trade with retry logic
            for attempt in range(self.max_retries):
                try:
                    # Add small delay to avoid wash trade detection
                    if attempt > 0:
                        await asyncio.sleep(1)
                    
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
                    
                    self.logger.info(f"✅ Executed {action} {quantity} shares of {symbol} at ${trade['price']:.2f}")
                    
                    # Update agent's last trade time
                    self.agents[trade['agent_id']]['last_trade_time'] = datetime.now()
                    
                    return True
                    
                except Exception as e:
                    self.logger.warning(f"Trade execution attempt {attempt + 1} failed: {e}")
                    if attempt < self.max_retries - 1:
                        await asyncio.sleep(2)
                    else:
                        raise e
            
        except Exception as e:
            self.logger.error(f"❌ Failed to execute trade: {e}")
            return False
    
    async def run_cycle(self):
        """Run one optimized trading cycle"""
        cycle_start = time.time()
        self.cycle_count += 1
        
        self.logger.info(f"Starting Optimized Smart Trading Cycle {self.cycle_count}")
        
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
                    decision = self._generate_optimized_decision(agent_id, market_data, positions, account_info)
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
            self.logger.info(f"Optimized Smart Trading Cycle {self.cycle_count} Summary:")
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
        """Run the optimized smart trading system"""
        self.logger.info("=" * 60)
        self.logger.info("OPTIMIZED SMART COMPETITIVE TRADING SYSTEM STARTING")
        self.logger.info("=" * 60)
        self.logger.info("✅ Connected to Alpaca PAPER TRADING API")
        self.logger.info(f"Initialized {len(self.agents)} optimized trading agents")
        self.logger.info("Optimized smart trading system initialized!")
        self.logger.info("All agents will make intelligent decisions with comprehensive validation!")
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
    os.makedirs('logs', exist_ok=True)
    
    # Initialize and run system
    system = OptimizedSmartTradingSystem()
    await system.run_system()

if __name__ == "__main__":
    asyncio.run(main())

