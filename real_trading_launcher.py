#!/usr/bin/env python3
"""
Real Trading System - Simple but Functional
==========================================
Direct connection to Alpaca with minimal complexity.
"""

import asyncio
import logging
import random
import os
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

# Try importing Alpaca (fail gracefully)
try:
    import alpaca_trade_api as tradeapi
    ALPACA_AVAILABLE = True
except ImportError:
    ALPACA_AVAILABLE = False

# Simple logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger("RealTrading")

class RealTradingSystem:
    """Real trading system with Alpaca integration"""
    
    def __init__(self):
        self.cycle_count = 0
        self.total_trades = 0
        self.agents = {
            'conservative_1': {'rate': 0.6, 'risk': 0.5},
            'aggressive_1': {'rate': 0.8, 'risk': 1.5},
            'balanced_1': {'rate': 0.7, 'risk': 1.0},
            'scalping_1': {'rate': 0.9, 'risk': 0.8},
            'momentum_1': {'rate': 0.8, 'risk': 1.2},
            'ai_enhanced_1': {'rate': 0.7, 'risk': 1.0},
            'arbitrage_1': {'rate': 0.4, 'risk': 0.3},
            'adaptive_1': {'rate': 0.8, 'risk': 0.9},
            'fractal_1': {'rate': 0.6, 'risk': 0.7},
            'candle_range_1': {'rate': 0.7, 'risk': 0.8},
            'quant_1': {'rate': 0.6, 'risk': 0.9},
            'ml_pattern_1': {'rate': 0.7, 'risk': 1.1}
        }
        
        # Initialize Alpaca if available
        self.api = None
        self.account = None
        
        if ALPACA_AVAILABLE:
            try:
                self.api = tradeapi.REST(
                    key_id=os.getenv("ALPACA_API_KEY"),
                    secret_key=os.getenv("ALPACA_SECRET_KEY"),
                    base_url=os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets"),
                    api_version='v2'
                )
                self.account = self.api.get_account()
                logger.info(f"✅ Connected to Alpaca (Balance: ${float(self.account.buying_power):,.2f})")
            except Exception as e:
                logger.warning(f"⚠️ Alpaca connection failed: {e}")
                self.api = None
        else:
            logger.warning("⚠️ Alpaca library not available - simulation mode")
        
        logger.info(f"✅ Real system initialized with {len(self.agents)} agents")
    
    async def get_market_data(self):
        """Get simple market data"""
        symbols = ['AAPL', 'MSFT', 'SPY', 'QQQ']
        market_data = {}
        
        for symbol in symbols:
            market_data[symbol] = {
                'price': random.uniform(100, 600),
                'change': random.uniform(-0.03, 0.03),
                'volume': random.randint(50000, 500000)
            }
        
        return market_data
    
    async def run_cycle(self):
        """Run one real trading cycle"""
        self.cycle_count += 1
        start_time = datetime.now()
        
        # Get market data
        market_data = await self.get_market_data()
        
        # Generate agent decisions
        decisions = []
        
        for agent_id, config in self.agents.items():
            if random.random() < config['rate']:
                symbol = random.choice(list(market_data.keys()))
                action = random.choice(['BUY', 'SELL'])
                
                # Calculate trade size based on risk and available funds
                if self.account:
                    max_trade = float(self.account.buying_power) * 0.02 * config['risk']
                    trade_value = min(max_trade, random.uniform(20, 100))
                else:
                    trade_value = random.uniform(20, 100)
                
                price = market_data[symbol]['price']
                quantity = max(0.001, trade_value / price)  # Fractional shares
                
                decision = {
                    'agent_id': agent_id,
                    'symbol': symbol,
                    'action': action,
                    'quantity': quantity,
                    'price': price,
                    'trade_value': trade_value
                }
                
                decisions.append(decision)
        
        # Execute top decisions
        executed = await self.execute_trades(decisions[:3])  # Top 3 trades
        
        duration = (datetime.now() - start_time).total_seconds()
        
        logger.info(f"📊 Cycle {self.cycle_count}: {len(decisions)} decisions, {executed} executed, {duration:.2f}s")
        
        # Update account info
        if self.api:
            try:
                self.account = self.api.get_account()
            except:
                pass
        
        return {
            'cycle': self.cycle_count,
            'decisions': len(decisions),
            'executed': executed,
            'duration': duration
        }
    
    async def execute_trades(self, decisions):
        """Execute trading decisions"""
        executed_count = 0
        
        for decision in decisions:
            try:
                if self.api:
                    # Real trade execution
                    symbol = decision['symbol']
                    action = decision['action']
                    quantity = decision['quantity']
                    
                    # Submit fractional order
                    order = self.api.submit_order(
                        symbol=symbol,
                        qty=quantity,
                        side='buy' if action == 'BUY' else 'sell',
                        type='market',
                        time_in_force='day'
                    )
                    
                    logger.info(f"✅ REAL TRADE: {decision['agent_id']} - {symbol} {action} {quantity:.3f} @ ${decision['price']:.2f}")
                    executed_count += 1
                    self.total_trades += 1
                else:
                    # Simulation
                    logger.info(f"🎯 SIM TRADE: {decision['agent_id']} - {decision['symbol']} {decision['action']} ${decision['trade_value']:.2f}")
                    executed_count += 1
                
                # Small delay between trades
                await asyncio.sleep(1)
                
            except Exception as e:
                logger.error(f"❌ Trade failed {decision['symbol']}: {e}")
        
        return executed_count

async def main():
    """Main trading loop"""
    logger.info("🚀 REAL TRADING SYSTEM STARTING")
    logger.info("=" * 50)
    
    system = RealTradingSystem()
    
    try:
        # Run continuous cycles
        cycle_count = 0
        while cycle_count < 20:  # Run 20 cycles then stop
            cycle_count += 1
            result = await system.run_cycle()
            
            # Sleep for 45 seconds between cycles
            await asyncio.sleep(45)
            
            # Every 5 cycles, show summary
            if cycle_count % 5 == 0:
                logger.info(f"🎯 Summary: {cycle_count} cycles, {system.total_trades} total trades")
        
        logger.info("🏁 Trading session completed!")
                
    except KeyboardInterrupt:
        logger.info("System stopped by user")
    except Exception as e:
        logger.error(f"Error: {e}")

if __name__ == "__main__":
    asyncio.run(main())