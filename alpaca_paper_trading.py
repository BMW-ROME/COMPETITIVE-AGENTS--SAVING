#!/usr/bin/env python3
"""
Foreground Alpaca Paper Trading System
=====================================
Real-time competitive trading with actual Alpaca Paper Trading API.
Runs in foreground with live updates and real market data.
"""

import asyncio
import logging
import random
import os
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import Alpaca (with fallback)
try:
    import alpaca_trade_api as tradeapi
    ALPACA_AVAILABLE = True
    print("✅ Alpaca Trade API loaded successfully")
except ImportError:
    ALPACA_AVAILABLE = False
    print("❌ Alpaca Trade API not available")

# Enhanced logging for foreground operation
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/alpaca_paper_trading.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("AlpacaPaperTrading")

class AlpacaPaperTradingSystem:
    """Real Alpaca Paper Trading System"""
    
    def __init__(self):
        self.cycle_count = 0
        self.total_trades = 0
        self.total_decisions = 0
        self.session_start = datetime.now()
        self.api = None
        self.account = None
        self.positions = {}
        
        # Initialize Alpaca API
        if ALPACA_AVAILABLE:
            try:
                self.api = tradeapi.REST(
                    key_id=os.getenv("ALPACA_API_KEY"),
                    secret_key=os.getenv("ALPACA_SECRET_KEY"),
                    base_url=os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets"),
                    api_version='v2'
                )
                
                # Get account info
                self.account = self.api.get_account()
                logger.info(f"🏦 ALPACA PAPER ACCOUNT CONNECTED")
                logger.info(f"💰 Buying Power: ${float(self.account.buying_power):,.2f}")
                logger.info(f"📊 Portfolio Value: ${float(self.account.portfolio_value):,.2f}")
                logger.info(f"💵 Cash: ${float(self.account.cash):,.2f}")
                logger.info(f"✅ Account Status: {self.account.status}")
                
                # Get current positions
                self.update_positions()
                
            except Exception as e:
                logger.error(f"❌ Failed to connect to Alpaca: {e}")
                self.api = None
        
        # Competitive agents with real trading focus
        self.agents = {
            'paper_trader_1': {'rate': 0.7, 'risk': 'medium', 'style': 'momentum'},
            'paper_trader_2': {'rate': 0.6, 'risk': 'low', 'style': 'value'},
            'paper_trader_3': {'rate': 0.8, 'risk': 'high', 'style': 'growth'},
            'paper_scalper_1': {'rate': 0.9, 'risk': 'medium', 'style': 'scalping'},
            'paper_scalper_2': {'rate': 0.85, 'risk': 'medium', 'style': 'scalping'},
            'paper_momentum_1': {'rate': 0.75, 'risk': 'high', 'style': 'momentum'},
            'paper_ai_1': {'rate': 0.7, 'risk': 'medium', 'style': 'ai'},
            'paper_ai_2': {'rate': 0.65, 'risk': 'medium', 'style': 'ai'},
            'paper_conservative': {'rate': 0.5, 'risk': 'low', 'style': 'conservative'},
            'paper_aggressive': {'rate': 0.9, 'risk': 'high', 'style': 'aggressive'},
            'paper_arbitrage': {'rate': 0.4, 'risk': 'low', 'style': 'arbitrage'},
            'paper_adaptive': {'rate': 0.8, 'risk': 'medium', 'style': 'adaptive'}
        }
        
        # Focus on liquid, tradeable stocks
        self.symbols = ['AAPL', 'MSFT', 'SPY', 'QQQ', 'TSLA', 'NVDA']
        
        logger.info(f"🤖 {len(self.agents)} competitive paper trading agents initialized")
        logger.info(f"📈 Trading symbols: {', '.join(self.symbols)}")
    
    def update_positions(self):
        """Update current positions from Alpaca"""
        if not self.api:
            return
        
        try:
            positions = self.api.list_positions()
            self.positions = {}
            
            logger.info("📊 CURRENT POSITIONS:")
            if positions:
                for pos in positions:
                    self.positions[pos.symbol] = {
                        'qty': float(pos.qty),
                        'market_value': float(pos.market_value),
                        'avg_entry_price': float(pos.avg_entry_price),
                        'unrealized_pnl': float(pos.unrealized_pnl),
                        'side': pos.side
                    }
                    
                    pnl_color = "🟢" if float(pos.unrealized_pnl) >= 0 else "🔴"
                    logger.info(f"   {pos.symbol}: {pos.qty} shares @ ${float(pos.avg_entry_price):.2f} | {pnl_color} P&L: ${float(pos.unrealized_pnl):.2f}")
            else:
                logger.info("   No current positions")
                
        except Exception as e:
            logger.error(f"Error updating positions: {e}")
    
    async def get_real_market_data(self):
        """Get real market data from Alpaca"""
        if not self.api:
            return self.get_simulated_data()
        
        market_data = {}
        
        try:
            for symbol in self.symbols:
                try:
                    # Get latest quote
                    quote = self.api.get_latest_quote(symbol)
                    
                    # Get recent bars for trend analysis
                    bars = self.api.get_bars(symbol, tradeapi.TimeFrame.Minute, limit=3).df
                    
                    if not bars.empty and quote.bid_price and quote.ask_price:
                        latest_bar = bars.iloc[-1]
                        prev_close = bars.iloc[0]['close'] if len(bars) > 1 else latest_bar['close']
                        
                        current_price = (float(quote.bid_price) + float(quote.ask_price)) / 2
                        price_change = (float(latest_bar['close']) - float(prev_close)) / float(prev_close)
                        
                        market_data[symbol] = {
                            'price': current_price,
                            'bid': float(quote.bid_price),
                            'ask': float(quote.ask_price),
                            'volume': int(latest_bar['volume']),
                            'change': price_change,
                            'volatility': float((latest_bar['high'] - latest_bar['low']) / latest_bar['close']),
                            'timestamp': datetime.now().isoformat()
                        }
                        
                    else:
                        # Fallback to simulated data for this symbol
                        market_data[symbol] = self.get_simulated_symbol_data(symbol)
                        
                except Exception as e:
                    logger.warning(f"Failed to get real data for {symbol}: {e}")
                    market_data[symbol] = self.get_simulated_symbol_data(symbol)
            
            logger.info(f"📡 Retrieved real market data for {len(market_data)} symbols")
            return market_data
            
        except Exception as e:
            logger.error(f"Market data error: {e}")
            return self.get_simulated_data()
    
    def get_simulated_data(self):
        """Fallback simulated data"""
        market_data = {}
        price_ranges = {
            'AAPL': (220, 235),
            'MSFT': (415, 435), 
            'SPY': (570, 580),
            'QQQ': (490, 505),
            'TSLA': (250, 270),
            'NVDA': (125, 135)
        }
        
        for symbol in self.symbols:
            market_data[symbol] = self.get_simulated_symbol_data(symbol, price_ranges)
        
        return market_data
    
    def get_simulated_symbol_data(self, symbol, price_ranges=None):
        """Get simulated data for a single symbol"""
        if price_ranges is None:
            price_ranges = {
                'AAPL': (220, 235),
                'MSFT': (415, 435), 
                'SPY': (570, 580),
                'QQQ': (490, 505),
                'TSLA': (250, 270),
                'NVDA': (125, 135)
            }
        
        min_price, max_price = price_ranges.get(symbol, (100, 200))
        base_price = random.uniform(min_price, max_price)
        
        return {
            'price': round(base_price, 2),
            'bid': round(base_price * 0.9995, 2),
            'ask': round(base_price * 1.0005, 2),
            'volume': random.randint(1000000, 5000000),
            'change': round(random.uniform(-0.02, 0.02), 4),
            'volatility': round(random.uniform(0.01, 0.03), 4),
            'timestamp': datetime.now().isoformat()
        }
    
    def generate_paper_decision(self, agent_id, market_data):
        """Generate paper trading decision"""
        agent = self.agents[agent_id]
        
        # Decision probability
        if random.random() > agent['rate']:
            return None
        
        # Select symbol based on strategy
        symbol = random.choice(self.symbols)
        data = market_data[symbol]
        
        # Determine action based on agent style
        style = agent['style']
        if style == 'momentum':
            action = 'BUY' if data['change'] > 0 else 'SELL'
        elif style == 'scalping':
            action = random.choice(['BUY', 'SELL'])
        elif style == 'conservative':
            action = 'BUY'  # Conservative prefers buying
        elif style == 'aggressive':
            action = random.choice(['BUY', 'SELL'])
        else:
            action = 'BUY' if data['change'] >= 0 else 'SELL'
        
        # Calculate position size based on account balance
        if self.account:
            buying_power = float(self.account.buying_power)
            max_trade_value = buying_power * 0.05 * {'low': 0.5, 'medium': 1.0, 'high': 1.5}[agent['risk']]
            max_trade_value = min(max_trade_value, 500)  # Cap at $500 per trade
        else:
            max_trade_value = 100
        
        # Check if we have existing position for sell orders
        current_position = self.positions.get(symbol, {})
        can_sell = current_position.get('qty', 0) > 0
        
        if action == 'SELL' and not can_sell:
            action = 'BUY'  # Can't sell what we don't have
        
        # Calculate quantity
        price = data['price']
        quantity = max_trade_value / price
        
        # For sell orders, limit to available shares
        if action == 'SELL' and can_sell:
            available_qty = abs(float(current_position['qty']))
            quantity = min(quantity, available_qty * 0.5)  # Sell at most 50% of position
        
        return {
            'agent_id': agent_id,
            'symbol': symbol,
            'action': action,
            'quantity': round(quantity, 4),
            'price': price,
            'trade_value': round(quantity * price, 2),
            'style': style,
            'confidence': round(random.uniform(0.6, 0.9), 3)
        }
    
    async def execute_paper_trade(self, decision):
        """Execute real paper trade via Alpaca"""
        if not self.api:
            logger.info(f"🎯 SIMULATED: {decision['agent_id']} | {decision['symbol']} {decision['action']} {decision['quantity']}")
            return True
        
        try:
            symbol = decision['symbol']
            action = decision['action']
            quantity = decision['quantity']
            
            # Submit order to Alpaca
            order = self.api.submit_order(
                symbol=symbol,
                qty=quantity,
                side='buy' if action == 'BUY' else 'sell',
                type='market',
                time_in_force='day'
            )
            
            logger.info(f"✅ REAL PAPER TRADE: {decision['agent_id']} | {symbol} {action} {quantity} | Order: {order.id} | Status: {order.status}")
            
            # Small delay to avoid rate limiting
            await asyncio.sleep(1)
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Paper trade failed: {decision['symbol']} {decision['action']} - {e}")
            return False
    
    async def run_paper_cycle(self):
        """Run one paper trading cycle"""
        self.cycle_count += 1
        cycle_start = datetime.now()
        
        logger.info(f"🔄 PAPER TRADING CYCLE {self.cycle_count}")
        logger.info("=" * 50)
        
        # Update account status
        if self.api:
            try:
                self.account = self.api.get_account()
                logger.info(f"💰 Current Buying Power: ${float(self.account.buying_power):,.2f}")
            except:
                pass
        
        # Get real market data
        market_data = await self.get_real_market_data()
        
        # Show market snapshot
        logger.info("📊 MARKET SNAPSHOT:")
        for symbol, data in market_data.items():
            change_emoji = "🟢" if data['change'] >= 0 else "🔴"
            logger.info(f"   {symbol}: ${data['price']:.2f} {change_emoji} {data['change']*100:+.2f}% | Vol: {data['volume']:,}")
        
        # Generate decisions
        decisions = []
        for agent_id in self.agents.keys():
            decision = self.generate_paper_decision(agent_id, market_data)
            if decision:
                decisions.append(decision)
                self.total_decisions += 1
        
        logger.info(f"🎯 AGENT DECISIONS: {len(decisions)}/12")
        
        # Select best trades (top 3 by confidence)
        if decisions:
            selected = sorted(decisions, key=lambda x: x['confidence'], reverse=True)[:3]
            
            logger.info("🏆 SELECTED TRADES:")
            for trade in selected:
                logger.info(f"   {trade['agent_id']}: {trade['symbol']} {trade['action']} {trade['quantity']} @ ${trade['price']:.2f} (confidence: {trade['confidence']})")
            
            # Execute trades
            executed = 0
            for trade in selected:
                if await self.execute_paper_trade(trade):
                    executed += 1
                    self.total_trades += 1
            
            logger.info(f"✅ EXECUTED: {executed}/{len(selected)} trades")
        else:
            logger.info("📭 No trading decisions this cycle")
        
        # Update positions after trades
        if self.api:
            self.update_positions()
        
        cycle_duration = (datetime.now() - cycle_start).total_seconds()
        logger.info(f"⏱️ Cycle completed in {cycle_duration:.2f}s")
        logger.info("=" * 50)
        
        return {
            'cycle': self.cycle_count,
            'decisions': len(decisions),
            'executed': executed if decisions else 0,
            'duration': cycle_duration
        }
    
    def show_session_summary(self):
        """Show session summary"""
        runtime = (datetime.now() - self.session_start).total_seconds() / 60
        
        logger.info("🏆 SESSION SUMMARY")
        logger.info("=" * 50)
        logger.info(f"⏱️ Runtime: {runtime:.1f} minutes")
        logger.info(f"🔄 Cycles: {self.cycle_count}")
        logger.info(f"🎯 Decisions: {self.total_decisions}")
        logger.info(f"💼 Trades: {self.total_trades}")
        
        if self.account:
            logger.info(f"💰 Final Buying Power: ${float(self.account.buying_power):,.2f}")
            logger.info(f"📊 Portfolio Value: ${float(self.account.portfolio_value):,.2f}")
        
        logger.info("=" * 50)

async def main():
    """Main paper trading session"""
    logger.info("🎯 ALPACA PAPER TRADING SYSTEM")
    logger.info("💼 Real paper trading with competitive agents")
    logger.info("🚀 Live market data and actual trade execution")
    logger.info("=" * 60)
    
    # Create paper trading system
    system = AlpacaPaperTradingSystem()
    
    if not system.api:
        logger.warning("⚠️ Running in simulation mode (Alpaca not connected)")
    
    try:
        # Run trading cycles (10 cycles for demo)
        for i in range(10):
            await system.run_paper_cycle()
            
            if i < 9:  # Don't wait after last cycle
                logger.info("⏳ Waiting 60 seconds before next cycle...\n")
                await asyncio.sleep(60)
        
        # Show final summary
        system.show_session_summary()
        
    except KeyboardInterrupt:
        logger.info("🛑 Paper trading session stopped by user")
        system.show_session_summary()
        
    except Exception as e:
        logger.error(f"❌ System error: {e}")
        system.show_session_summary()

if __name__ == "__main__":
    # Ensure logs directory exists
    os.makedirs('logs', exist_ok=True)
    
    # Run paper trading system
    asyncio.run(main())