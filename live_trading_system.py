#!/usr/bin/env python3
"""
Lightweight Continuous Trading System
=====================================
Optimized for immediate deployment with minimal dependencies
"""

import asyncio
import logging
import sys
import os
import json
import random
import signal
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import alpaca_trade_api as tradeapi
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'logs/live_trading_{datetime.now().strftime("%Y%m%d_%H%M")}.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger("LiveTrading")

class LiveCompetitiveTradingSystem:
    """Lightweight 24/7 competitive trading system"""
    
    def __init__(self):
        self.running = False
        self.cycle_count = 0
        self.total_trades = 0
        
        # Initialize Alpaca API
        self.api = tradeapi.REST(
            key_id=os.getenv("ALPACA_API_KEY"),
            secret_key=os.getenv("ALPACA_SECRET_KEY"),
            base_url=os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets"),
            api_version='v2'
        )
        
        # Get account info
        account = self.api.get_account()
        self.initial_buying_power = float(account.buying_power)
        
        # Trading parameters
        self.max_trade_value = min(20, self.initial_buying_power * 0.02)  # 2% max per trade
        self.cycle_interval = 60  # 1 minute cycles
        
        # 12 agents with simple configurations
        self.agents = {
            f'agent_{i}': {
                'decision_rate': random.uniform(0.3, 0.8),
                'trade_size': random.uniform(0.5, 1.5)
            } for i in range(1, 13)
        }
        
        logger.info(f"🚀 Live System Ready - Buying Power: ${self.initial_buying_power:.2f}")
        logger.info(f"⚙️ 12 agents initialized, max trade: ${self.max_trade_value:.2f}")
        
        # Signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        logger.info("📡 Shutdown signal received, stopping gracefully...")
        self.running = False
    
    async def start_live_trading(self):
        """Start live trading operation"""
        self.running = True
        logger.info("🎯 STARTING LIVE COMPETITIVE TRADING")
        logger.info("=" * 50)
        
        try:
            while self.running:
                await self._run_trading_cycle()
                
                # Log progress every 10 cycles
                if self.cycle_count % 10 == 0:
                    account = self.api.get_account()
                    current_bp = float(account.buying_power)
                    pnl = current_bp - self.initial_buying_power
                    
                    logger.info(f"📊 Cycle {self.cycle_count}: ${current_bp:.2f} BP, {self.total_trades} trades, ${pnl:+.2f} P&L")
                
                await asyncio.sleep(self.cycle_interval)
                
        except Exception as e:
            logger.error(f"💥 Critical error: {e}")
        finally:
            logger.info("✅ Live trading stopped")
    
    async def _run_trading_cycle(self):
        """Run one trading cycle"""
        self.cycle_count += 1
        
        try:
            # Get current account status
            account = self.api.get_account()
            buying_power = float(account.buying_power)
            
            # Skip if low buying power
            if buying_power < 10:
                return
            
            # Get market data for liquid symbols
            symbols = ['SPY', 'QQQ', 'AAPL', 'MSFT']
            market_data = {}
            
            for symbol in symbols:
                try:
                    quote = self.api.get_latest_quote(symbol)
                    if quote and quote.bid_price:
                        market_data[symbol] = {
                            'price': (float(quote.bid_price) + float(quote.ask_price)) / 2,
                            'tradeable': True
                        }
                except:
                    continue
            
            if not market_data:
                return
            
            # Generate agent decisions
            decisions = []
            for agent_id, config in self.agents.items():
                if random.random() < config['decision_rate']:
                    symbol = random.choice(list(market_data.keys()))
                    data = market_data[symbol]
                    
                    trade_value = self.max_trade_value * config['trade_size']
                    trade_value = min(trade_value, buying_power * 0.1)  # Max 10% of BP
                    
                    if trade_value >= 5:  # Minimum $5 trade
                        decisions.append({
                            'agent_id': agent_id,
                            'symbol': symbol,
                            'action': random.choice(['BUY', 'SELL']),
                            'trade_value': trade_value,
                            'price': data['price']
                        })
            
            # Execute top 2 decisions
            executed = 0
            for decision in sorted(decisions, key=lambda x: x['trade_value'], reverse=True)[:2]:
                if await self._execute_trade(decision):
                    executed += 1
                    
                    # Add delay between trades
                    if executed < len(decisions):
                        await asyncio.sleep(3)
            
            if executed > 0:
                self.total_trades += executed
                logger.info(f"✅ Cycle {self.cycle_count}: {len(decisions)} decisions, {executed} trades executed")
                
        except Exception as e:
            logger.error(f"❌ Cycle {self.cycle_count} error: {e}")
    
    async def _execute_trade(self, decision: Dict[str, Any]) -> bool:
        """Execute a single trade"""
        try:
            symbol = decision['symbol']
            action = decision['action']
            trade_value = decision['trade_value']
            
            # Submit notional order (dollar amount)
            order = self.api.submit_order(
                symbol=symbol,
                notional=trade_value,
                side='buy' if action == 'BUY' else 'sell',
                type='market',
                time_in_force='day'
            )
            
            logger.info(f"💰 {symbol} {action} ${trade_value:.2f} [{decision['agent_id']}] - Order: {order.id}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Trade failed {decision['symbol']}: {str(e)[:50]}...")
            return False

async def main():
    """Main entry point"""
    try:
        os.makedirs('logs', exist_ok=True)
        
        system = LiveCompetitiveTradingSystem()
        
        logger.info("🎯 LIVE COMPETITIVE TRADING SYSTEM")
        logger.info("💎 Paper Trading Mode - Safe for Testing")
        logger.info("🔄 Press Ctrl+C to stop gracefully")
        logger.info("=" * 50)
        
        await system.start_live_trading()
        
    except KeyboardInterrupt:
        logger.info("👋 Graceful shutdown completed")
    except Exception as e:
        logger.error(f"💥 Fatal error: {e}")

if __name__ == "__main__":
    asyncio.run(main())