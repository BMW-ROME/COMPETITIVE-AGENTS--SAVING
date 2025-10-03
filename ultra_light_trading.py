#!/usr/bin/env python3
"""
Ultra-Light Trading System - No Heavy Dependencies
=================================================
Minimal competitive trading system for immediate deployment
"""

import asyncio
import logging
import sys
import os
import json
import random
import time
import signal
import requests
from datetime import datetime
from typing import Dict, List, Any, Optional

# Configure logging
log_file = f'logs/ultra_light_{datetime.now().strftime("%Y%m%d_%H%M")}.log'
os.makedirs('logs', exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger("UltraLight")

class UltraLightTradingSystem:
    """Ultra-lightweight trading system"""
    
    def __init__(self):
        self.running = False
        self.cycle_count = 0
        self.total_trades = 0
        
        # Alpaca API settings from environment
        self.api_key = "PKK43GTIACJNUPGZPCPF"
        self.secret_key = "CuQWde4QtPHAtwuMfxaQhB8njmrcJDq0YK4Oz9Rw"
        self.base_url = "https://paper-api.alpaca.markets"
        
        # Headers for API requests
        self.headers = {
            'APCA-API-KEY-ID': self.api_key,
            'APCA-API-SECRET-KEY': self.secret_key,
            'Content-Type': 'application/json'
        }
        
        # Get account info
        self.buying_power = self._get_buying_power()
        
        # Trading parameters
        self.max_trade_value = min(15, self.buying_power * 0.02) if self.buying_power > 0 else 15
        self.cycle_interval = 90  # 90 second cycles
        
        # Initialize 12 simple agents
        self.agents = {}
        for i in range(1, 13):
            self.agents[f'agent_{i}'] = {
                'decision_rate': random.uniform(0.2, 0.7),
                'trade_multiplier': random.uniform(0.5, 1.5),
                'total_trades': 0
            }
        
        logger.info(f"🚀 Ultra-Light System Ready")
        logger.info(f"💰 Buying Power: ${self.buying_power:.2f}")
        logger.info(f"⚙️ Max Trade: ${self.max_trade_value:.2f}")
        logger.info(f"🤖 12 agents initialized")
        
        # Signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        logger.info("📡 Shutdown signal received")
        self.running = False
    
    def _get_buying_power(self) -> float:
        """Get account buying power via direct API call"""
        try:
            url = f"{self.base_url}/v2/account"
            response = requests.get(url, headers=self.headers, timeout=10)
            
            if response.status_code == 200:
                account_data = response.json()
                return float(account_data.get('buying_power', 0))
            else:
                logger.error(f"Account API error: {response.status_code}")
                return 100.0  # Fallback value
                
        except Exception as e:
            logger.error(f"Error getting buying power: {e}")
            return 100.0  # Fallback value
    
    def _get_quote(self, symbol: str) -> Optional[Dict[str, float]]:
        """Get latest quote for symbol"""
        try:
            url = f"{self.base_url}/v2/stocks/{symbol}/quotes/latest"
            response = requests.get(url, headers=self.headers, timeout=5)
            
            if response.status_code == 200:
                data = response.json()
                quote = data.get('quote', {})
                
                bid_price = float(quote.get('bid_price', 0))
                ask_price = float(quote.get('ask_price', 0))
                
                if bid_price > 0 and ask_price > 0:
                    return {
                        'price': (bid_price + ask_price) / 2,
                        'bid': bid_price,
                        'ask': ask_price
                    }
            
            return None
            
        except Exception as e:
            logger.error(f"Quote error for {symbol}: {e}")
            return None
    
    def _submit_order(self, symbol: str, side: str, notional: float) -> bool:
        """Submit order via direct API call"""
        try:
            url = f"{self.base_url}/v2/orders"
            
            order_data = {
                "symbol": symbol,
                "side": side,
                "type": "market",
                "time_in_force": "day",
                "notional": str(notional)
            }
            
            response = requests.post(url, headers=self.headers, 
                                   data=json.dumps(order_data), timeout=10)
            
            if response.status_code == 201:
                order = response.json()
                logger.info(f"✅ Order submitted: {symbol} {side} ${notional:.2f} - ID: {order.get('id', 'unknown')}")
                return True
            else:
                error_msg = response.text[:100] if response.text else f"HTTP {response.status_code}"
                logger.error(f"❌ Order failed: {error_msg}")
                return False
                
        except Exception as e:
            logger.error(f"Order submission error: {str(e)[:50]}...")
            return False
    
    async def start_ultra_light_trading(self):
        """Start ultra-light trading operation"""
        self.running = True
        
        logger.info("🎯 STARTING ULTRA-LIGHT COMPETITIVE TRADING")
        logger.info("💎 Paper Trading - Safe Mode")
        logger.info("🔄 Press Ctrl+C to stop")
        logger.info("=" * 50)
        
        try:
            while self.running:
                await self._run_ultra_light_cycle()
                
                # Progress update every 5 cycles
                if self.cycle_count % 5 == 0:
                    current_bp = self._get_buying_power()
                    pnl = current_bp - self.buying_power if self.buying_power > 0 else 0
                    
                    logger.info(f"📊 Cycle {self.cycle_count}: ${current_bp:.2f} BP, {self.total_trades} trades, ${pnl:+.2f} P&L")
                
                await asyncio.sleep(self.cycle_interval)
                
        except KeyboardInterrupt:
            logger.info("👋 User requested shutdown")
        except Exception as e:
            logger.error(f"💥 Critical error: {e}")
        finally:
            logger.info("✅ Ultra-light trading stopped")
    
    async def _run_ultra_light_cycle(self):
        """Run one ultra-light trading cycle"""
        self.cycle_count += 1
        
        try:
            # Check buying power
            current_bp = self._get_buying_power()
            if current_bp < 5:
                return
            
            # Get quotes for liquid symbols
            symbols = ['SPY', 'QQQ', 'AAPL']
            quotes = {}
            
            for symbol in symbols:
                quote = self._get_quote(symbol)
                if quote:
                    quotes[symbol] = quote
            
            if not quotes:
                return
            
            # Generate agent decisions
            decisions = []
            active_agents = 0
            
            for agent_id, agent in self.agents.items():
                if random.random() < agent['decision_rate']:
                    active_agents += 1
                    
                    symbol = random.choice(list(quotes.keys()))
                    quote = quotes[symbol]
                    
                    trade_value = self.max_trade_value * agent['trade_multiplier']
                    trade_value = min(trade_value, current_bp * 0.08)  # Max 8% of BP
                    
                    if trade_value >= 5:  # Min $5 trade
                        decisions.append({
                            'agent_id': agent_id,
                            'symbol': symbol,
                            'side': random.choice(['buy', 'sell']),
                            'notional': trade_value,
                            'price': quote['price']
                        })
            
            # Execute top decision
            if decisions:
                # Sort by trade value and pick best
                best_decision = max(decisions, key=lambda x: x['notional'])
                
                if self._submit_order(best_decision['symbol'], 
                                    best_decision['side'], 
                                    best_decision['notional']):
                    self.total_trades += 1
                    self.agents[best_decision['agent_id']]['total_trades'] += 1
                    
                    logger.info(f"🎯 Cycle {self.cycle_count}: {active_agents} active, 1 trade executed")
            
        except Exception as e:
            logger.error(f"❌ Cycle error: {e}")
    
    def _get_status_summary(self):
        """Get system status summary"""
        most_active = max(self.agents.items(), key=lambda x: x[1]['total_trades'])
        
        return {
            'cycles': self.cycle_count,
            'trades': self.total_trades,
            'most_active_agent': most_active[0],
            'agent_trades': most_active[1]['total_trades']
        }

async def main():
    """Main entry point"""
    try:
        system = UltraLightTradingSystem()
        await system.start_ultra_light_trading()
        
    except Exception as e:
        logger.error(f"💥 System error: {e}")

if __name__ == "__main__":
    logger.info("🎯 ULTRA-LIGHT COMPETITIVE TRADING SYSTEM")
    logger.info("📦 Minimal dependencies, maximum reliability")
    logger.info("=" * 50)
    
    asyncio.run(main())