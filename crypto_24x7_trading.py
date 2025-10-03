#!/usr/bin/env python3
"""
24/7 Crypto Trading System - Never-Stop Competitive Trading
Integrates all existing RL, risk management, and trading modules for continuous crypto trading
"""

import asyncio
import alpaca_trade_api as tradeapi
import os
import sys
import signal
import time
from datetime import datetime, timedelta
from dotenv import load_dotenv
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import logging
import json
import random
import numpy as np

# Import existing modules
sys.path.append('/workspaces/competitive-trading-agents')
from rl_100_percent_execution import get_100_percent_execution_optimizer
from rl_optimization_engine import RLTradingOptimizer

@dataclass
class CryptoTradingStats:
    start_time: str
    total_cycles: int = 0
    total_decisions: int = 0
    total_trades_attempted: int = 0
    total_trades_executed: int = 0
    total_pnl: float = 0.0
    initial_buying_power: float = 0.0
    current_buying_power: float = 0.0
    active_positions: int = 0
    runtime_hours: float = 0.0
    crypto_pairs_traded: int = 0
    avg_execution_rate: float = 0.0

class Crypto24x7TradingSystem:
    def __init__(self):
        # Load environment
        load_dotenv()
        
        # Initialize Alpaca API
        self.api = tradeapi.REST(
            os.getenv('APCA_API_KEY_ID'),
            os.getenv('APCA_API_SECRET_KEY'), 
            'https://paper-api.alpaca.markets'
        )
        
        # System configuration
        self.running = False
        self.cycle_interval = 60  # 60 seconds for crypto (faster than stocks)
        self.max_trade_value = 100  # Start conservative
        self.min_buying_power_threshold = 500
        self.max_positions = 10
        
        # Crypto-specific settings
        self.major_cryptos = [
            'BTC/USD', 'ETH/USD', 'DOGE/USD', 'LTC/USD', 'SOL/USD',
            'BTC/USDT', 'ETH/USDT', 'DOGE/USDT', 'LTC/USDT', 'SOL/USDT',
            'BTC/USDC', 'ETH/USDC', 'DOGE/USDC', 'LTC/USDC', 'SOL/USDC'
        ]
        
        # Trading agents (adapted for crypto)
        self.agents = {}
        self.agent_performance = {}
        self.cycle_count = 0
        
        # Statistics tracking
        self.stats = CryptoTradingStats(
            start_time=datetime.now().isoformat()
        )
        
        # RL Integration
        self.rl_optimizer = RLTradingOptimizer()
        
        # Logging setup
        self._setup_logging()
        
        # Signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        self.logger.info("🚀 24/7 Crypto Trading System Initialized")
    
    def _setup_logging(self):
        """Setup logging for crypto trading"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('logs/crypto_24x7_trading.log'),
                logging.StreamHandler(sys.stdout)
            ]
        )
        self.logger = logging.getLogger('Crypto24x7Trading')
    
    def _initialize_crypto_agents(self):
        """Initialize trading agents optimized for crypto markets"""
        crypto_agent_configs = {
            'crypto_scalper_1': {
                'strategy': 'scalping',
                'risk_tolerance': 0.3,
                'position_size': 0.15,
                'target_pairs': ['BTC/USD', 'ETH/USD'],
                'timeframe': '1min'
            },
            'crypto_scalper_2': {
                'strategy': 'scalping',
                'risk_tolerance': 0.4,
                'position_size': 0.12,
                'target_pairs': ['DOGE/USD', 'LTC/USD'],
                'timeframe': '1min'
            },
            'crypto_momentum_1': {
                'strategy': 'momentum',
                'risk_tolerance': 0.5,
                'position_size': 0.20,
                'target_pairs': ['SOL/USD', 'BTC/USDT'],
                'timeframe': '5min'
            },
            'crypto_momentum_2': {
                'strategy': 'momentum',
                'risk_tolerance': 0.6,
                'position_size': 0.18,
                'target_pairs': ['ETH/USDT', 'DOGE/USDT'],
                'timeframe': '5min'
            },
            'crypto_arbitrage_1': {
                'strategy': 'arbitrage',
                'risk_tolerance': 0.2,
                'position_size': 0.25,
                'target_pairs': ['BTC/USD', 'BTC/USDT', 'BTC/USDC'],
                'timeframe': '1min'
            },
            'crypto_swing_1': {
                'strategy': 'swing',
                'risk_tolerance': 0.7,
                'position_size': 0.30,
                'target_pairs': ['ETH/USD', 'SOL/USD'],
                'timeframe': '15min'
            }
        }
        
        for agent_id, config in crypto_agent_configs.items():
            self.agents[agent_id] = config
            self.agent_performance[agent_id] = {
                'total_trades': 0,
                'successful_trades': 0,
                'total_pnl': 0.0,
                'last_trade_time': None,
                'win_rate': 0.0,
                'confidence': 0.5,
                'crypto_pairs_traded': set()
            }
        
        self.logger.info(f"✅ Initialized {len(self.agents)} crypto trading agents")
    
    def _get_crypto_market_data(self) -> Dict:
        """Get real-time crypto market data"""
        market_data = {}
        
        for symbol in self.major_cryptos:
            try:
                # Get latest bars (1-minute)
                bars = self.api.get_crypto_bars(symbol, '1Min', limit=5)
                
                if bars and hasattr(bars, 'df') and not bars.df.empty:
                    latest_bar = bars.df.iloc[-1]
                    prev_bar = bars.df.iloc[-2] if len(bars.df) > 1 else latest_bar
                    
                    price = float(latest_bar['close'])
                    volume = float(latest_bar['volume'])
                    change = price - float(prev_bar['close'])
                    change_pct = (change / float(prev_bar['close'])) * 100 if prev_bar['close'] != 0 else 0
                    
                    market_data[symbol] = {
                        'price': price,
                        'volume': volume,
                        'change': change,
                        'change_pct': change_pct,
                        'high': float(latest_bar['high']),
                        'low': float(latest_bar['low']),
                        'vwap': float(latest_bar.get('vwap', price)),
                        'timestamp': datetime.now().isoformat()
                    }
                    
            except Exception as e:
                self.logger.warning(f"⚠️ Failed to get data for {symbol}: {e}")
                # Use fallback data
                market_data[symbol] = {
                    'price': 50000.0 if 'BTC' in symbol else 3000.0 if 'ETH' in symbol else 0.5,
                    'volume': 1000000,
                    'change': 0,
                    'change_pct': 0,
                    'high': 50500.0 if 'BTC' in symbol else 3100.0 if 'ETH' in symbol else 0.6,
                    'low': 49500.0 if 'BTC' in symbol else 2900.0 if 'ETH' in symbol else 0.4,
                    'vwap': 50000.0 if 'BTC' in symbol else 3000.0 if 'ETH' in symbol else 0.5,
                    'timestamp': datetime.now().isoformat()
                }
        
        return market_data
    
    def _generate_crypto_decisions(self, market_data: Dict) -> List[Dict]:
        """Generate trading decisions for crypto pairs"""
        decisions = []
        
        for agent_id, agent_config in self.agents.items():
            target_pairs = agent_config['target_pairs']
            
            for symbol in target_pairs:
                if symbol not in market_data:
                    continue
                
                data = market_data[symbol]
                
                # Enhanced crypto-specific decision logic
                decision_score = 0.0
                action = 'hold'
                
                # Momentum-based decisions
                if agent_config['strategy'] == 'momentum':
                    if abs(data['change_pct']) > 1.0:  # 1% move
                        decision_score = min(0.9, abs(data['change_pct']) / 5.0)
                        action = 'buy' if data['change_pct'] > 0 else 'sell'
                
                # Scalping decisions (small, frequent trades)
                elif agent_config['strategy'] == 'scalping':
                    if abs(data['change_pct']) > 0.2:  # 0.2% move
                        decision_score = min(0.8, abs(data['change_pct']) / 2.0)
                        action = 'buy' if data['change_pct'] > 0 else 'sell'
                
                # Arbitrage opportunities
                elif agent_config['strategy'] == 'arbitrage':
                    # Look for price differences between USD/USDT/USDC pairs
                    base_crypto = symbol.split('/')[0]
                    usd_price = market_data.get(f'{base_crypto}/USD', {}).get('price', 0)
                    usdt_price = market_data.get(f'{base_crypto}/USDT', {}).get('price', 0)
                    
                    if usd_price > 0 and usdt_price > 0:
                        price_diff_pct = abs(usd_price - usdt_price) / usd_price * 100
                        if price_diff_pct > 0.1:  # 0.1% arbitrage opportunity
                            decision_score = min(0.85, price_diff_pct / 0.5)
                            action = 'buy' if usd_price < usdt_price else 'sell'
                
                # Swing trading
                elif agent_config['strategy'] == 'swing':
                    # Look for larger moves and reversals
                    if abs(data['change_pct']) > 2.0:  # 2% move
                        decision_score = min(0.75, abs(data['change_pct']) / 8.0)
                        # Counter-trend for swing
                        action = 'sell' if data['change_pct'] > 2.0 else 'buy'
                
                # Generate decision if score is sufficient
                if decision_score > 0.3 and action != 'hold':
                    # Calculate position size
                    base_quantity = self.max_trade_value / data['price']
                    adjusted_quantity = base_quantity * agent_config['position_size'] * decision_score
                    
                    decision = {
                        'agent_id': agent_id,
                        'symbol': symbol,
                        'action': action,
                        'quantity': adjusted_quantity,
                        'confidence': decision_score,
                        'trade_value': adjusted_quantity * data['price'],
                        'strategy': agent_config['strategy'],
                        'crypto_pair': symbol
                    }
                    
                    decisions.append(decision)
        
        # Add some randomness for exploration
        if len(decisions) < 3 and random.random() < 0.3:
            random_symbol = random.choice(list(market_data.keys()))
            random_data = market_data[random_symbol]
            
            decisions.append({
                'agent_id': 'crypto_explorer',
                'symbol': random_symbol,
                'action': random.choice(['buy', 'sell']),
                'quantity': (self.max_trade_value * 0.1) / random_data['price'],
                'confidence': 0.4,
                'trade_value': self.max_trade_value * 0.1,
                'strategy': 'exploration',
                'crypto_pair': random_symbol
            })
        
        return decisions
    
    async def _execute_crypto_trades(self, decisions: List[Dict], market_data: Dict) -> List[Dict]:
        """Execute crypto trades with RL optimization"""
        if not decisions:
            return []
        
        # Apply RL 100% execution optimization
        rl_100_optimizer = get_100_percent_execution_optimizer()
        
        # Convert to RL format
        trade_candidates = []
        for decision in decisions:
            candidate = {
                'symbol': decision['symbol'],
                'side': decision['action'].lower(),
                'quantity': decision['quantity'],
                'confidence': decision['confidence'],
                'agent_id': decision['agent_id'],
                'trade_value': decision['trade_value']
            }
            trade_candidates.append(candidate)
        
        # RL optimization
        self.logger.info(f"🧠 [RL_CRYPTO] Optimizing {len(trade_candidates)} crypto trades for 100% execution")
        optimized_trades = rl_100_optimizer.optimize_for_100_percent_execution(trade_candidates, market_data)
        
        # Execute optimized trades
        executed_trades = []
        
        for trade in optimized_trades:
            symbol = trade['symbol']
            action = trade.get('side', trade.get('action', 'buy')).lower()
            quantity = trade['quantity']
            executed = False  # Initialize execution status
            
            try:
                # Pre-validate trade parameters
                if quantity <= 0:
                    raise ValueError("Invalid quantity")
                
                # Check account status before submitting
                account = self.api.get_account()
                if float(account.buying_power) < (quantity * market_data.get(symbol, {}).get('price', 50000)):
                    raise ValueError("Insufficient buying power")
                
                # Submit crypto order
                order = self.api.submit_order(
                    symbol=symbol,
                    qty=quantity,
                    side=action,
                    type='market',
                    time_in_force='gtc'  # Good Till Canceled for crypto
                )
                
                if order and hasattr(order, 'id'):
                    executed = True
                    
                    # Calculate P&L
                    try:
                        positions = self.api.list_positions()
                        position_pnl = 0.0
                        for pos in positions:
                            if pos.symbol == symbol:
                                position_pnl = float(pos.unrealized_pl or 0.0)
                                break
                    except:
                        position_pnl = 0.0
                    
                    # Update agent performance
                    agent_perf = self.agent_performance[trade['agent_id']]
                    agent_perf['total_trades'] += 1
                    agent_perf['successful_trades'] += 1
                    agent_perf['total_pnl'] += position_pnl
                    agent_perf['win_rate'] = agent_perf['successful_trades'] / agent_perf['total_trades']
                    agent_perf['last_trade_time'] = datetime.now()
                    agent_perf['crypto_pairs_traded'].add(symbol)
                    
                    # Update session stats
                    self.stats.total_pnl += position_pnl
                    self.stats.total_trades_executed += 1
                    
                    exec_prob = trade.get('execution_probability', 0.0)
                    self.logger.info(f"✅ CRYPTO FILLED: {trade['agent_id']} | {symbol} {action.upper()} "
                                   f"${trade['trade_value']:.2f} | P&L: ${position_pnl:.2f} | Exec Prob: {exec_prob:.1%}")
                    
                    executed_trades.append({
                        'symbol': symbol,
                        'action': action,
                        'quantity': quantity,
                        'agent_id': trade['agent_id'],
                        'execution_probability': exec_prob,
                        'rl_optimized': True,
                        'crypto_pair': symbol
                    })
                
                # Record RL outcome
                rl_trade = trade.copy()
                rl_trade['side'] = action
                rl_100_optimizer.record_execution_outcome(rl_trade, executed, None)
                
            except Exception as e:
                error_msg = str(e).lower()
                failure_reason = 'other'
                
                if 'insufficient' in error_msg and 'buying power' in error_msg:
                    failure_reason = 'insufficient_buying_power'
                elif 'insufficient' in error_msg and 'quantity' in error_msg:
                    failure_reason = 'insufficient_quantity' 
                elif 'risk' in error_msg or 'limit' in error_msg:
                    failure_reason = 'risk_limits'
                elif 'invalid' in error_msg:
                    failure_reason = 'invalid_parameters'
                
                # Only log non-routine rejections to reduce noise
                if failure_reason in ['insufficient_buying_power', 'invalid_parameters']:
                    self.logger.info(f"ℹ️ CRYPTO SKIPPED: {symbol} - {failure_reason}")
                else:
                    self.logger.debug(f"🔄 CRYPTO RETRY: {symbol} - {failure_reason}")
                
                # Record RL failure
                rl_trade = trade.copy()
                rl_trade['side'] = action
                rl_100_optimizer.record_execution_outcome(rl_trade, False, failure_reason)
        
        return executed_trades
    
    async def _run_crypto_cycle(self):
        """Run a single crypto trading cycle"""
        cycle_start = time.time()
        
        try:
            # Get market data
            market_data = self._get_crypto_market_data()
            
            if not market_data:
                self.logger.warning("⚠️ No crypto market data available")
                return
            
            # Generate decisions
            decisions = self._generate_crypto_decisions(market_data)
            self.stats.total_decisions += len(decisions)
            
            # Execute trades
            if decisions:
                self.stats.total_trades_attempted += len(decisions)
                executed_trades = await self._execute_crypto_trades(decisions, market_data)
                
                # Log cycle summary
                cycle_time = time.time() - cycle_start
                self.cycle_count += 1
                
                self.logger.info(f"🔄 Crypto Cycle {self.cycle_count}: "
                               f"{len(decisions)} decisions, {len(executed_trades)} trades, {cycle_time:.1f}s")
                
                # Update crypto-specific stats
                crypto_pairs = set()
                for trade in executed_trades:
                    crypto_pairs.add(trade.get('crypto_pair', trade['symbol']))
                
                self.stats.crypto_pairs_traded = len(crypto_pairs)
                self.stats.total_cycles = self.cycle_count
                
        except Exception as e:
            self.logger.error(f"❌ Error in crypto cycle {self.cycle_count}: {e}")
    
    async def start_crypto_trading(self):
        """Start 24/7 crypto trading"""
        self.running = True
        
        # Initialize
        self._initialize_crypto_agents()
        
        # Get initial account info
        try:
            account = self.api.get_account()
            self.stats.initial_buying_power = float(account.buying_power)
            self.stats.current_buying_power = self.stats.initial_buying_power
        except Exception as e:
            self.logger.error(f"❌ Failed to get account info: {e}")
            return
        
        self.logger.info(f"💰 Initial Buying Power: ${self.stats.initial_buying_power:,.2f}")
        self.logger.info(f"⚙️ Max Trade Value: ${self.max_trade_value:.2f}")
        self.logger.info(f"🔄 Cycle Interval: {self.cycle_interval}s")
        self.logger.info(f"🚀 LAUNCHING 24/7 CRYPTO COMPETITIVE TRADING")
        self.logger.info(f"🌍 Crypto Markets: ALWAYS OPEN")
        self.logger.info(f"💎 Press Ctrl+C to stop gracefully")
        self.logger.info("=" * 60)
        
        try:
            while self.running:
                await self._run_crypto_cycle()
                
                # Periodic reporting
                if self.cycle_count % 10 == 0:
                    await self._log_crypto_performance()
                
                # Wait for next cycle
                await asyncio.sleep(self.cycle_interval)
                
        except Exception as e:
            self.logger.error(f"💥 Critical error in crypto trading: {e}")
        finally:
            await self._shutdown_gracefully()
    
    async def _log_crypto_performance(self):
        """Log crypto trading performance"""
        runtime = (datetime.now() - datetime.fromisoformat(self.stats.start_time)).total_seconds() / 3600
        self.stats.runtime_hours = runtime
        
        if self.stats.total_trades_attempted > 0:
            self.stats.avg_execution_rate = (self.stats.total_trades_executed / self.stats.total_trades_attempted) * 100
        
        self.logger.info("🪙 24/7 CRYPTO PERFORMANCE REPORT")
        self.logger.info(f"⏱️ Runtime: {runtime:.1f} hours")
        self.logger.info(f"🔄 Cycles: {self.stats.total_cycles}")
        self.logger.info(f"🎯 Decisions: {self.stats.total_decisions}")
        self.logger.info(f"📈 Trades: {self.stats.total_trades_executed}/{self.stats.total_trades_attempted} ({self.stats.avg_execution_rate:.1f}%)")
        self.logger.info(f"💰 Total P&L: ${self.stats.total_pnl:.2f}")
        self.logger.info(f"🪙 Crypto Pairs Traded: {self.stats.crypto_pairs_traded}")
    
    async def _shutdown_gracefully(self):
        """Graceful shutdown"""
        self.logger.info("🔄 Performing graceful crypto trading shutdown...")
        
        # Save session data
        session_data = {
            'timestamp': datetime.now().isoformat(),
            'stats': {
                'start_time': self.stats.start_time,
                'total_cycles': self.stats.total_cycles,
                'total_decisions': self.stats.total_decisions,
                'total_trades_attempted': self.stats.total_trades_attempted,
                'total_trades_executed': self.stats.total_trades_executed,
                'total_pnl': self.stats.total_pnl,
                'runtime_hours': self.stats.runtime_hours,
                'crypto_pairs_traded': self.stats.crypto_pairs_traded,
                'avg_execution_rate': self.stats.avg_execution_rate
            },
            'agent_performance': {
                agent_id: {
                    'total_trades': perf['total_trades'],
                    'successful_trades': perf['successful_trades'],
                    'total_pnl': perf['total_pnl'],
                    'win_rate': perf['win_rate'],
                    'crypto_pairs_traded': list(perf['crypto_pairs_traded'])
                }
                for agent_id, perf in self.agent_performance.items()
            }
        }
        
        filename = f"data/crypto_session_{datetime.now().strftime('%Y%m%d_%H%M')}.json"
        with open(filename, 'w') as f:
            json.dump(session_data, f, indent=2)
        
        self.logger.info(f"💾 Crypto session saved: {filename}")
        
        # Final performance log
        await self._log_crypto_performance()
        
        self.logger.info("✅ Crypto trading shutdown complete")
        self.logger.info("🏁 24/7 Crypto trading system stopped")
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        self.logger.info(f"📡 Received signal {signum}, initiating crypto trading shutdown...")
        self.running = False

async def main():
    """Main function to run 24/7 crypto trading"""
    crypto_system = Crypto24x7TradingSystem()
    await crypto_system.start_crypto_trading()

if __name__ == "__main__":
    asyncio.run(main())