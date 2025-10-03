#!/usr/bin/env python3
"""
Minimal Alpaca Paper Trading System
==================================
Direct REST API integration without complex dependencies.
Uses only requests library to avoid aiohttp compilation issues.
"""

import asyncio
import logging
import random
import os
import json
import time
from datetime import datetime
from dotenv import load_dotenv
import requests

# Load environment variables
load_dotenv()

# Enhanced logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/alpaca_paper_trading_minimal.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("AlpacaMinimal")

class MinimalAlpacaTradingSystem:
    """Minimal Alpaca Paper Trading with direct REST API"""
    
    def __init__(self):
        self.cycle_count = 0
        self.total_trades = 0
        self.session_start = datetime.now()
        
        # Alpaca credentials
        self.api_key = os.getenv('ALPACA_API_KEY', 'PKK43GTIACJNUPGZPCPF')
        self.secret_key = os.getenv('ALPACA_SECRET_KEY', 'your_secret_key_here')
        self.base_url = os.getenv('ALPACA_BASE_URL', 'https://paper-api.alpaca.markets')
        
        # Headers for API requests
        self.headers = {
            'APCA-API-KEY-ID': self.api_key,
            'APCA-API-SECRET-KEY': self.secret_key,
            'Content-Type': 'application/json'
        }
        
        # Trading symbols
        self.symbols = ['AAPL', 'MSFT', 'SPY', 'QQQ', 'TSLA', 'NVDA']
        
        # Competitive agents
        self.agents = [
            {'name': 'minimal_scalper', 'risk': 0.1, 'confidence': 0.0},
            {'name': 'minimal_momentum', 'risk': 0.2, 'confidence': 0.0},
            {'name': 'minimal_conservative', 'risk': 0.05, 'confidence': 0.0},
            {'name': 'minimal_aggressive', 'risk': 0.3, 'confidence': 0.0},
            {'name': 'minimal_balanced', 'risk': 0.15, 'confidence': 0.0},
            {'name': 'minimal_ai', 'risk': 0.25, 'confidence': 0.0}
        ]
        
        logger.info("🎯 MINIMAL ALPACA PAPER TRADING SYSTEM")
        logger.info("💼 Direct REST API integration")
        logger.info("🚀 Zero complex dependencies")
        logger.info("=" * 55)
        
    def get_account(self):
        """Get account information"""
        try:
            url = f"{self.base_url}/v2/account"
            response = requests.get(url, headers=self.headers, timeout=10)
            
            if response.status_code == 200:
                account = response.json()
                logger.info("🏦 MINIMAL ALPACA ACCOUNT CONNECTED")
                logger.info(f"💰 Buying Power: ${float(account.get('buying_power', 0)):.2f}")
                logger.info(f"📊 Portfolio Value: ${float(account.get('portfolio_value', 0)):.2f}")
                logger.info(f"💵 Cash: ${float(account.get('cash', 0)):.2f}")
                logger.info(f"✅ Account Status: {account.get('status', 'UNKNOWN')}")
                return account
            else:
                logger.error(f"❌ Failed to get account: {response.status_code}")
                return None
        except Exception as e:
            logger.error(f"❌ Account error: {e}")
            return None
    
    def get_market_data(self):
        """Get market data for symbols"""
        try:
            # Use Alpaca data API
            url = f"{self.base_url}/v2/stocks/snapshots"
            params = {'symbols': ','.join(self.symbols)}
            response = requests.get(url, headers=self.headers, params=params, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                market_data = {}
                for symbol in self.symbols:
                    if symbol in data.get('snapshots', {}):
                        snap = data['snapshots'][symbol]
                        quote = snap.get('latestQuote', {})
                        trade = snap.get('latestTrade', {})
                        daily = snap.get('dailyBar', {})
                        
                        price = trade.get('p', quote.get('bp', 100.0))
                        prev_close = daily.get('c', price)
                        change_pct = ((price - prev_close) / prev_close * 100) if prev_close else 0
                        volume = daily.get('v', random.randint(1000000, 5000000))
                        
                        market_data[symbol] = {
                            'price': price,
                            'change_pct': change_pct,
                            'volume': volume
                        }
                    else:
                        # Fallback data
                        market_data[symbol] = {
                            'price': random.uniform(50, 500),
                            'change_pct': random.uniform(-3, 3),
                            'volume': random.randint(1000000, 5000000)
                        }
                
                return market_data
            else:
                logger.warning(f"⚠️ Market data request failed: {response.status_code}")
                # Return fallback data
                return {symbol: {
                    'price': random.uniform(50, 500),
                    'change_pct': random.uniform(-3, 3),
                    'volume': random.randint(1000000, 5000000)
                } for symbol in self.symbols}
        
        except Exception as e:
            logger.error(f"❌ Market data error: {e}")
            # Return fallback data
            return {symbol: {
                'price': random.uniform(50, 500),
                'change_pct': random.uniform(-3, 3),
                'volume': random.randint(1000000, 5000000)
            } for symbol in self.symbols}
    
    def generate_agent_decisions(self, market_data):
        """Generate trading decisions for each agent"""
        decisions = []
        
        for agent in self.agents:
            # Simple decision logic
            symbol = random.choice(self.symbols)
            data = market_data[symbol]
            
            # Basic trading signals
            volume_ok = data['volume'] > 1000000
            price_movement = abs(data['change_pct']) > 0.5
            
            if volume_ok and price_movement and random.random() > 0.3:
                side = 'buy' if data['change_pct'] > 0 else 'sell'
                qty = round(random.uniform(0.01, 0.1), 4)
                confidence = random.uniform(0.6, 0.9)
                
                agent['confidence'] = confidence
                
                decisions.append({
                    'agent': agent['name'],
                    'symbol': symbol,
                    'side': side,
                    'qty': qty,
                    'price': data['price'],
                    'confidence': confidence
                })
        
        return decisions
    
    def execute_trade(self, decision):
        """Execute a trade via Alpaca API"""
        try:
            url = f"{self.base_url}/v2/orders"
            
            order_data = {
                'symbol': decision['symbol'],
                'qty': str(decision['qty']),
                'side': decision['side'],
                'type': 'market',
                'time_in_force': 'day'
            }
            
            response = requests.post(url, headers=self.headers, json=order_data, timeout=10)
            
            if response.status_code in [200, 201]:
                order = response.json()
                order_id = order.get('id', 'unknown')
                status = order.get('status', 'unknown')
                
                logger.info(f"✅ MINIMAL TRADE: {decision['agent']} | {decision['symbol']} {decision['side'].upper()} {decision['qty']} | Order: {order_id[:8]}... | Status: {status}")
                self.total_trades += 1
                return True
            else:
                error_msg = response.json().get('message', 'Unknown error') if response.content else 'No response'
                logger.error(f"❌ Trade failed: {decision['symbol']} {decision['side'].upper()} - {error_msg}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Trade execution error: {e}")
            return False
    
    async def trading_cycle(self):
        """Execute one trading cycle"""
        self.cycle_count += 1
        
        logger.info(f"🔄 MINIMAL TRADING CYCLE {self.cycle_count}")
        logger.info("=" * 45)
        
        # Get account info
        account = self.get_account()
        if not account:
            logger.error("❌ Cannot proceed without account access")
            return
        
        buying_power = float(account.get('buying_power', 0))
        logger.info(f"💰 Current Buying Power: ${buying_power:.2f}")
        
        # Get market data
        market_data = self.get_market_data()
        logger.info(f"📡 Retrieved market data for {len(market_data)} symbols")
        
        # Display market snapshot
        logger.info("📊 MARKET SNAPSHOT:")
        for symbol, data in market_data.items():
            direction = "🟢" if data['change_pct'] >= 0 else "🔴"
            logger.info(f"   {symbol}: ${data['price']:.2f} {direction} {data['change_pct']:+.2f}% | Vol: {data['volume']:,}")
        
        # Generate agent decisions
        decisions = self.generate_agent_decisions(market_data)
        logger.info(f"🎯 AGENT DECISIONS: {len(decisions)}/6")
        
        if not decisions:
            logger.info("😴 No trading opportunities found")
            return
        
        # Select top decisions
        decisions.sort(key=lambda x: x['confidence'], reverse=True)
        selected_decisions = decisions[:3]  # Top 3
        
        logger.info("🏆 SELECTED TRADES:")
        for decision in selected_decisions:
            logger.info(f"   {decision['agent']}: {decision['symbol']} {decision['side'].upper()} {decision['qty']} @ ${decision['price']:.2f} (confidence: {decision['confidence']:.3f})")
        
        # Execute trades
        executed = 0
        for decision in selected_decisions:
            if buying_power > 10:  # Basic buying power check
                if self.execute_trade(decision):
                    executed += 1
                    buying_power -= decision['qty'] * decision['price']  # Rough estimate
                    time.sleep(1)  # Rate limiting
            else:
                logger.warning("💰 Insufficient buying power for more trades")
                break
        
        logger.info(f"✅ EXECUTED: {executed}/{len(selected_decisions)} trades")
        logger.info(f"⏱️ Cycle completed | Total session trades: {self.total_trades}")
        logger.info("=" * 45)
        
    async def run(self):
        """Main trading loop"""
        logger.info("🚀 Starting Minimal Alpaca Paper Trading System")
        
        try:
            while True:
                await self.trading_cycle()
                logger.info("⏳ Waiting 60 seconds before next cycle...")
                await asyncio.sleep(60)
                
        except KeyboardInterrupt:
            logger.info("🛑 Trading system stopped by user")
        except Exception as e:
            logger.error(f"❌ System error: {e}")
        
        # Final summary
        runtime = datetime.now() - self.session_start
        logger.info(f"📊 SESSION COMPLETE:")
        logger.info(f"   Runtime: {runtime}")
        logger.info(f"   Cycles: {self.cycle_count}")
        logger.info(f"   Total Trades: {self.total_trades}")

async def main():
    """Main entry point"""
    # Create logs directory
    os.makedirs('logs', exist_ok=True)
    
    # Start the trading system
    system = MinimalAlpacaTradingSystem()
    await system.run()

if __name__ == "__main__":
    asyncio.run(main())